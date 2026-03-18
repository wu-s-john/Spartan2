// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Support for generating R1CS using bellpepper.
use crate::{
  Blind, Commitment, CommitmentKey, MULTIROUND_COMMITMENT_WIDTH, PCS, VerifierKey,
  bellpepper::{shape_cs::ShapeCS, solver::SatisfyingAssignment},
  errors::SpartanError,
  r1cs::{
    R1CSWitness, SparseMatrix, SplitMultiRoundR1CSInstance, SplitMultiRoundR1CSShape,
    SplitR1CSInstance, SplitR1CSShape,
  },
  small_constraint_system::{SmallCoeff, SmallSatisfyingAssignment},
  spartan::SpartanPrepSNARK,
  start_span,
  traits::{
    Engine,
    circuit::{MultiRoundCircuit, SmallSpartanCircuit, SpartanCircuit},
    pcs::PCSEngineTrait,
    transcript::TranscriptEngineTrait,
  },
};
use bellpepper::gadgets::num::AllocatedNum;
use bellpepper_core::{ConstraintSystem, Index, LinearCombination, Variable};
use ff::{Field, PrimeField};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

/// `SpartanShape` provides methods for acquiring `SplitR1CSShape` from implementers.
pub trait SpartanShape<E: Engine> {
  /// Return an appropriate `SplitR1CSShape`
  fn r1cs_shape<C: SpartanCircuit<E>>(circuit: &C) -> Result<SplitR1CSShape<E>, SpartanError>;
}

/// Defines rerandomization behavior for preprocessed state
pub trait RerandomizationTrait<E: Engine> {
  /// Returns a rerandomized version of self.
  fn rerandomize(&self, ck: &CommitmentKey<E>, S: &SplitR1CSShape<E>) -> Result<Self, SpartanError>
  where
    Self: Sized;

  /// Returns a rerandomized version of self, reusing shared commitments and blinds if provided.
  fn rerandomize_with_shared(
    &self,
    ck: &CommitmentKey<E>,
    S: &SplitR1CSShape<E>,
    shared: &Option<WitnessCommitment<E>>,
  ) -> Result<Self, SpartanError>
  where
    Self: Sized;
}

/// `SpartanWitness` provide a method for acquiring an `SplitR1CSInstance` and `R1CSWitness` from implementers.
pub trait SpartanWitness<E: Engine> {
  /// Holds the state of the prover after committing to the precommitted portions of the witness.
  type PrecommittedState: Send
    + Sync
    + Serialize
    + for<'de> Deserialize<'de>
    + RerandomizationTrait<E>;

  /// Return partial commitments (to shared variables) and the shared state
  fn shared_witness<C: SpartanCircuit<E>>(
    S: &SplitR1CSShape<E>,
    ck: &CommitmentKey<E>,
    circuit: &C,
    is_small: bool,
  ) -> Result<Self::PrecommittedState, SpartanError>;

  /// Return partial commitments (to shared and precommitted variables) and the partial witness vector.
  fn precommitted_witness<C: SpartanCircuit<E>>(
    ps: &mut Self::PrecommittedState,
    S: &SplitR1CSShape<E>,
    ck: &CommitmentKey<E>,
    circuit: &C,
    is_small: bool,
  ) -> Result<(), SpartanError>;

  /// Return an instance and witness, given a shape and ck.
  fn r1cs_instance_and_witness<C: SpartanCircuit<E>>(
    ps: &mut Self::PrecommittedState,
    S: &SplitR1CSShape<E>,
    ck: &CommitmentKey<E>,
    circuit: &C,
    is_small: bool,
    transcript: &mut E::TE,
  ) -> Result<(SplitR1CSInstance<E>, R1CSWitness<E>), SpartanError>;
}

/// `SmallSpartanWitness` provides methods for witness preparation on the small-value (integer) path.
///
/// Mirrors `SpartanWitness` but for `SmallSatisfyingAssignment<W>`. The small path always
/// uses `commit_witness` (bool MSM), so there is no `is_small` flag.
///
/// `Coeff` is the matrix-coefficient type stored in `SplitR1CSShape<E, Coeff>`.
pub trait SmallSpartanWitness<E: Engine, W, Coeff> {
  /// Holds the pre-processed state for the small-value proving path.
  type SmallPrepState;

  /// Synthesizes the shared witness and commits to it.
  ///
  /// Returns a `SmallPrepState` with `comm_precommitted: None`.
  fn shared_witness<C: SmallSpartanCircuit<E, W>>(
    S: &SplitR1CSShape<E, Coeff>,
    ck: &CommitmentKey<E>,
    circuit: &C,
  ) -> Result<Self::SmallPrepState, SpartanError>;

  /// Synthesizes the precommitted witness and commits to it, mutating `prep` in-place.
  fn precommitted_witness<C: SmallSpartanCircuit<E, W>>(
    prep: &mut Self::SmallPrepState,
    S: &SplitR1CSShape<E, Coeff>,
    ck: &CommitmentKey<E>,
    circuit: &C,
  ) -> Result<(), SpartanError>;
}

/// `MultiRoundSpartanShape` provides methods for acquiring `SplitMultiRoundR1CSShape` and `CommitmentKey` from implementers.
pub trait MultiRoundSpartanShape<E: Engine> {
  /// Return an appropriate `SplitMultiRoundR1CSShape` and `CommitmentKey` structs.
  fn multiround_r1cs_shape<C: MultiRoundCircuit<E>>(
    circuit: &C,
  ) -> Result<
    (
      SplitMultiRoundR1CSShape<E>,
      CommitmentKey<E>,
      VerifierKey<E>,
    ),
    SpartanError,
  >;
}

/// `MultiRoundSpartanWitness` provide a method for acquiring an `SplitMultiRoundR1CSInstance` and `R1CSWitness` from implementers.
pub trait MultiRoundSpartanWitness<E: Engine> {
  /// Holds the state of the prover across multiple rounds.
  type MultiRoundState: Send + Sync + Serialize + for<'de> Deserialize<'de>;

  /// Initialize the multi-round witness process and return the initial state.
  fn initialize_multiround_witness(
    s: &SplitMultiRoundR1CSShape<E>,
  ) -> Result<Self::MultiRoundState, SpartanError>;

  /// Process a specific round and update the state, returning the challenges generated after the commitment to the round.
  fn process_round<C: MultiRoundCircuit<E>>(
    state: &mut Self::MultiRoundState,
    s: &SplitMultiRoundR1CSShape<E>,
    ck: &CommitmentKey<E>,
    circuit: &C,
    round_index: usize,
    transcript: &mut E::TE,
  ) -> Result<Vec<E::Scalar>, SpartanError>;

  /// Finalize the multi-round witness and return the instance and witness.
  fn finalize_multiround_witness(
    state: &mut Self::MultiRoundState,
    s: &SplitMultiRoundR1CSShape<E>,
  ) -> Result<(SplitMultiRoundR1CSInstance<E>, R1CSWitness<E>), SpartanError>;
}

impl<E: Engine> SpartanShape<E> for ShapeCS<E> {
  fn r1cs_shape<C: SpartanCircuit<E>>(circuit: &C) -> Result<SplitR1CSShape<E>, SpartanError> {
    let num_challenges = circuit.num_challenges();

    let mut cs: Self = Self::new();

    // allocate shared variables
    let shared = circuit
      .shared(&mut cs)
      .map_err(|e| SpartanError::SynthesisError {
        reason: format!("Unable to allocate shared variables: {e}"),
      })?;

    // determine the number of shared variables
    let num_shared = cs.num_aux();
    debug!("num_shared: {}", num_shared);
    debug!("shared.len(): {}", shared.len());
    if shared.len() > num_shared {
      return Err(SpartanError::SynthesisError {
        reason: "Shared variables are not allocated correctly".to_string(),
      });
    }

    // allocate precommitted variables
    let precommitted =
      circuit
        .precommitted(&mut cs, &shared)
        .map_err(|e| SpartanError::SynthesisError {
          reason: format!("Unable to allocate precommitted variables: {e}"),
        })?;

    let num_precommitted = cs.num_aux() - num_shared;
    debug!("num_precommitted: {}", num_precommitted);
    debug!("precommitted.len(): {}", precommitted.len());
    if precommitted.len() > num_precommitted {
      return Err(SpartanError::SynthesisError {
        reason: "Precommitted variables are not allocated correctly".to_string(),
      });
    }

    // synthesize the circuit
    circuit
      .synthesize(&mut cs, &shared, &precommitted, None)
      .map_err(|e| SpartanError::SynthesisError {
        reason: format!("Unable to synthesize circuit: {e}"),
      })?;

    let mut A = SparseMatrix::<E::Scalar>::empty();
    let mut B = SparseMatrix::<E::Scalar>::empty();
    let mut C = SparseMatrix::<E::Scalar>::empty();

    let mut num_cons_added = 0;
    let mut X = (&mut A, &mut B, &mut C, &mut num_cons_added);
    let num_inputs = cs.num_inputs();
    let num_constraints = cs.num_constraints();
    let num_vars = cs.num_aux();

    for constraint in cs.constraints.iter() {
      add_constraint(
        &mut X,
        num_vars,
        &constraint.0,
        &constraint.1,
        &constraint.2,
      );
    }
    assert_eq!(num_cons_added, num_constraints);
    assert!(num_inputs > num_challenges);

    A.cols = num_vars + num_inputs;
    B.cols = num_vars + num_inputs;
    C.cols = num_vars + num_inputs;

    let num_rest = num_vars - num_shared - num_precommitted;

    debug!("num_constraints: {}", num_constraints);
    debug!("num_vars: {}", num_vars);
    debug!("num_inputs: {}", num_inputs);
    debug!("num_shared: {}", num_shared);
    debug!("num_precommitted: {}", num_precommitted);
    debug!("num_rest: {}", num_rest);

    // Don't count One as an input for shape's purposes.
    let S = SplitR1CSShape::new(
      num_constraints,
      num_shared,
      num_precommitted,
      num_rest,
      num_inputs - 1 - num_challenges,
      num_challenges,
      A,
      B,
      C,
    )
    .unwrap();

    Ok(S)
  }
}

/// Extract a `SplitR1CSShape<E, Coeff>` from a `SmallSpartanCircuit`.
///
/// Uses `SmallShapeCS<Coeff>` to record small-coefficient constraints directly,
/// without creating any field elements.
pub fn small_r1cs_shape<E: Engine, Coeff: SmallCoeff, Circuit: SmallSpartanCircuit<E, Coeff>>(
  circuit: &Circuit,
) -> Result<SplitR1CSShape<E, Coeff>, SpartanError> {
  use crate::small_constraint_system::SmallShapeCS;

  let num_challenges = circuit.num_challenges();
  let mut cs = SmallShapeCS::<Coeff>::new();

  let shared = circuit
    .shared(&mut cs)
    .map_err(|e| SpartanError::SynthesisError {
      reason: format!("small_r1cs_shape: shared: {e}"),
    })?;
  let num_shared = cs.num_vars();

  let precommitted =
    circuit
      .precommitted(&mut cs, &shared)
      .map_err(|e| SpartanError::SynthesisError {
        reason: format!("small_r1cs_shape: precommitted: {e}"),
      })?;
  let num_precommitted = cs.num_vars() - num_shared;

  circuit
    .synthesize(&mut cs, &shared, &precommitted, None)
    .map_err(|e| SpartanError::SynthesisError {
      reason: format!("small_r1cs_shape: synthesize: {e}"),
    })?;

  let num_vars = cs.num_vars();
  let num_inputs = cs.num_inputs(); // includes ONE at index 0
  let num_rest = num_vars - num_shared - num_precommitted;
  let num_public = num_inputs - 1 - num_challenges; // subtract ONE and challenges

  // Convert SmallShapeCS constraints to SparseMatrix<Coeff>
  let (mut A, mut B, mut C) = cs.to_matrices();
  A.cols = num_vars + num_inputs;
  B.cols = num_vars + num_inputs;
  C.cols = num_vars + num_inputs;

  let num_constraints = cs.num_constraints();

  SplitR1CSShape::<E, Coeff>::new_int(
    num_constraints,
    num_shared,
    num_precommitted,
    num_rest,
    num_public,
    num_challenges,
    A,
    B,
    C,
  )
  .map_err(|e| SpartanError::SynthesisError {
    reason: format!("small_r1cs_shape: new_int: {e:?}"),
  })
}

/// A commitment to a witness segment together with its blinding factor.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct WitnessCommitment<E: Engine> {
  pub(crate) comm: Commitment<E>,
  pub(crate) blind: Blind<E>,
}

/// A type that holds the pre-processed state for proving with the small-value (integer) path.
///
/// Contains the witness assignment, partial commitments, and the full witness vector.
/// The type parameter `W` controls the witness value type (e.g., `i8` for binary witnesses).
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SmallPrepSNARK<E: Engine, W> {
  pub(crate) cs: SmallSatisfyingAssignment<W>,
  pub(crate) shared: Vec<Variable>,
  pub(crate) precommitted: Vec<Variable>,
  pub(crate) comm_shared: Option<WitnessCommitment<E>>,
  pub(crate) comm_precommitted: Option<WitnessCommitment<E>>,
  pub(crate) W: Vec<W>,
}

pub(crate) fn add_constraint<S: PrimeField>(
  X: &mut (
    &mut SparseMatrix<S>,
    &mut SparseMatrix<S>,
    &mut SparseMatrix<S>,
    &mut usize,
  ),
  num_vars: usize,
  a_lc: &LinearCombination<S>,
  b_lc: &LinearCombination<S>,
  c_lc: &LinearCombination<S>,
) {
  let (A, B, C, nn) = X;
  let n = **nn;
  assert_eq!(n + 1, A.indptr.len(), "A: invalid shape");
  assert_eq!(n + 1, B.indptr.len(), "B: invalid shape");
  assert_eq!(n + 1, C.indptr.len(), "C: invalid shape");

  let add_constraint_component = |index: Index, coeff: &S, M: &mut SparseMatrix<S>| {
    // we add constraints to the matrix only if the associated coefficient is non-zero
    if *coeff != S::ZERO {
      match index {
        Index::Input(idx) => {
          // Inputs come last, with input 0, representing 'one',
          // at position num_vars within the witness vector.
          let idx = idx + num_vars;
          M.data.push(*coeff);
          M.indices.push(idx);
        }
        Index::Aux(idx) => {
          M.data.push(*coeff);
          M.indices.push(idx);
        }
      }
    }
  };

  for (index, coeff) in a_lc.iter() {
    add_constraint_component(index.0, coeff, A);
  }
  A.indptr.push(A.indices.len());

  for (index, coeff) in b_lc.iter() {
    add_constraint_component(index.0, coeff, B)
  }
  B.indptr.push(B.indices.len());

  for (index, coeff) in c_lc.iter() {
    add_constraint_component(index.0, coeff, C)
  }
  C.indptr.push(C.indices.len());

  **nn += 1;
}

impl<E: Engine> SpartanWitness<E> for SatisfyingAssignment<E> {
  type PrecommittedState = SpartanPrepSNARK<E>;

  fn shared_witness<C: SpartanCircuit<E>>(
    S: &SplitR1CSShape<E>,
    ck: &CommitmentKey<E>,
    circuit: &C,
    is_small: bool,
  ) -> Result<Self::PrecommittedState, SpartanError> {
    let (_synth_span, synth_t) = start_span!("shared_witness_synthesize");
    let mut cs: Self = Self::new();

    let num_vars = S.num_shared + S.num_precommitted + S.num_rest;
    debug!("num_vars: {}", num_vars);

    let mut W = vec![E::Scalar::ZERO; num_vars];

    // produce shared witness variables and commit to them
    let shared = circuit
      .shared(&mut cs)
      .map_err(|e| SpartanError::SynthesisError {
        reason: format!("Unable to allocate shared variables: {e}"),
      })?;

    // we know that W is large enough to hold all shared variables
    if cs.aux_assignment.len() < S.num_shared_unpadded {
      return Err(SpartanError::SynthesisError {
        reason: "Shared variables are not allocated correctly".to_string(),
      });
    }
    W[..S.num_shared_unpadded].copy_from_slice(&cs.aux_assignment[..S.num_shared_unpadded]);

    // partial commitment to shared witness variables
    let (_commit_span, commit_t) = start_span!("commit_witness_shared");
    let (comm_W_shared, r_W_shared) = if S.num_shared_unpadded > 0 {
      let r_W_shared = PCS::<E>::blind(ck, S.num_shared);
      let comm_W_shared = PCS::<E>::commit(ck, &W[0..S.num_shared], &r_W_shared, is_small)?;
      (Some(comm_W_shared), Some(r_W_shared))
    } else {
      (None, None)
    };
    info!(elapsed_ms = %commit_t.elapsed().as_millis(), "commit_witness_shared");
    info!(elapsed_ms = %synth_t.elapsed().as_millis(), "shared_witness_synthesize");

    Ok(SpartanPrepSNARK {
      cs,
      shared,
      precommitted: vec![],
      comm_shared: comm_W_shared.map(|comm| WitnessCommitment {
        comm,
        blind: r_W_shared.unwrap(),
      }),
      comm_precommitted: None,
      W,
    })
  }

  fn precommitted_witness<C: SpartanCircuit<E>>(
    ps: &mut Self::PrecommittedState,
    S: &SplitR1CSShape<E>,
    ck: &CommitmentKey<E>,
    circuit: &C,
    is_small: bool,
  ) -> Result<(), SpartanError> {
    let (_synth_span, synth_t) = start_span!("precommitted_witness_synthesize");
    // produce precommitted witness variables
    let precommitted =
      circuit
        .precommitted(&mut ps.cs, &ps.shared)
        .map_err(|e| SpartanError::SynthesisError {
          reason: format!("Unable to allocate precommitted variables: {e}"),
        })?;

    if ps.cs.aux_assignment[S.num_shared_unpadded..].len() < S.num_precommitted_unpadded {
      return Err(SpartanError::SynthesisError {
        reason: "Precommitted variables are not allocated correctly".to_string(),
      });
    }
    ps.W[S.num_shared..S.num_shared + S.num_precommitted_unpadded].copy_from_slice(
      &ps.cs.aux_assignment
        [S.num_shared_unpadded..S.num_shared_unpadded + S.num_precommitted_unpadded],
    );
    info!(elapsed_ms = %synth_t.elapsed().as_millis(), "precommitted_witness_synthesize");

    // partial commitment to precommitted witness variables
    let (_commit_precommitted_span, commit_precommitted_t) =
      start_span!("commit_witness_precommitted");
    let (comm_W_precommitted, r_W_precommitted) = if S.num_precommitted_unpadded > 0 {
      let r_W_precommitted = PCS::<E>::blind(ck, S.num_precommitted);
      let comm_W_precommitted = PCS::<E>::commit(
        ck,
        &ps.W[S.num_shared..S.num_shared + S.num_precommitted],
        &r_W_precommitted,
        is_small,
      )?;
      (Some(comm_W_precommitted), Some(r_W_precommitted))
    } else {
      (None, None)
    };
    info!(elapsed_ms = %commit_precommitted_t.elapsed().as_millis(), "commit_witness_precommitted");

    // update the preprocessed state
    ps.comm_precommitted = comm_W_precommitted.map(|comm| WitnessCommitment {
      comm,
      blind: r_W_precommitted.unwrap(),
    });
    ps.precommitted = precommitted;

    Ok(())
  }

  fn r1cs_instance_and_witness<C: SpartanCircuit<E>>(
    ps: &mut Self::PrecommittedState,
    S: &SplitR1CSShape<E>,
    ck: &CommitmentKey<E>,
    circuit: &C,
    is_small: bool,
    transcript: &mut E::TE,
  ) -> Result<(SplitR1CSInstance<E>, R1CSWitness<E>), SpartanError> {
    let (_synth_span, synth_t) = start_span!("circuit_synthesize_rest");

    // absorb partial commitments into transcript
    if let Some(wc) = &ps.comm_shared {
      transcript.absorb(b"comm_W_shared", &wc.comm);
    }
    if let Some(wc) = &ps.comm_precommitted {
      transcript.absorb(b"comm_W_precommitted", &wc.comm);
    }

    let challenges = (0..S.num_challenges)
      .map(|_| transcript.squeeze(b"challenge"))
      .collect::<Result<Vec<E::Scalar>, SpartanError>>()?;

    circuit
      .synthesize(&mut ps.cs, &ps.shared, &ps.precommitted, Some(&challenges))
      .map_err(|e| SpartanError::SynthesisError {
        reason: format!("Unable to synthesize witness: {e}"),
      })?;

    ps.W
      [S.num_shared + S.num_precommitted..S.num_shared + S.num_precommitted + S.num_rest_unpadded]
      .copy_from_slice(
        &ps.cs.aux_assignment[S.num_shared_unpadded + S.num_precommitted_unpadded
          ..S.num_shared_unpadded + S.num_precommitted_unpadded + S.num_rest_unpadded],
      );

    // commit to the rest with partial commitment
    let (_commit_rest_span, commit_rest_t) = start_span!("commit_witness_rest");
    let r_W_rest = PCS::<E>::blind(ck, S.num_rest);
    let comm_W_rest = PCS::<E>::commit(
      ck,
      &ps.W[S.num_shared + S.num_precommitted..S.num_shared + S.num_precommitted + S.num_rest],
      &r_W_rest,
      is_small,
    )?;
    info!(elapsed_ms = %commit_rest_t.elapsed().as_millis(), "commit_witness_rest");
    transcript.absorb(b"comm_W_rest", &comm_W_rest); // add commitment to transcript

    let public_values = ps.cs.input_assignment[1..].to_vec()[..S.num_public].to_vec();
    let U = SplitR1CSInstance::<E>::new(
      S,
      ps.comm_shared.as_ref().map(|wc| wc.comm.clone()),
      ps.comm_precommitted.as_ref().map(|wc| wc.comm.clone()),
      comm_W_rest,
      public_values,
      challenges,
    )?;

    let mut blinds = Vec::with_capacity(3);
    if let Some(wc) = &ps.comm_shared {
      blinds.push(wc.blind.clone());
    }
    if let Some(wc) = &ps.comm_precommitted {
      blinds.push(wc.blind.clone());
    }
    blinds.push(r_W_rest);

    let r_W = PCS::<E>::combine_blinds(&blinds)?;

    let W = R1CSWitness::<E>::new_unchecked(ps.W.clone(), r_W, is_small)?;

    info!(elapsed_ms = %synth_t.elapsed().as_millis(), "circuit_synthesize_rest");

    Ok((U, W))
  }
}

impl<E: Engine> RerandomizationTrait<E> for SpartanPrepSNARK<E> {
  fn rerandomize(&self, ck: &CommitmentKey<E>, S: &SplitR1CSShape<E>) -> Result<Self, SpartanError>
  where
    Self: Sized,
  {
    // generate new blinds for shared and precommitted commitments and rerandomize commitments
    let comm_shared_new = if let Some(wc) = &self.comm_shared {
      let r_new = PCS::<E>::blind(ck, S.num_shared);
      Some(WitnessCommitment {
        comm: PCS::<E>::rerandomize_commitment(ck, &wc.comm, &wc.blind, &r_new)?,
        blind: r_new,
      })
    } else {
      None
    };
    let comm_precommitted_new = if let Some(wc) = &self.comm_precommitted {
      let r_new = PCS::<E>::blind(ck, S.num_precommitted);
      Some(WitnessCommitment {
        comm: PCS::<E>::rerandomize_commitment(ck, &wc.comm, &wc.blind, &r_new)?,
        blind: r_new,
      })
    } else {
      None
    };

    Ok(SpartanPrepSNARK {
      cs: self.cs.clone(),
      shared: self.shared.clone(),
      precommitted: self.precommitted.clone(),
      comm_shared: comm_shared_new,
      comm_precommitted: comm_precommitted_new,
      W: self.W.clone(),
    })
  }

  fn rerandomize_with_shared(
    &self,
    ck: &CommitmentKey<E>,
    S: &SplitR1CSShape<E>,
    shared: &Option<WitnessCommitment<E>>,
  ) -> Result<Self, SpartanError>
  where
    Self: Sized,
  {
    // generate new blinds for precommitted commitments and rerandomize commitments
    let comm_precommitted_new = if let Some(wc) = &self.comm_precommitted {
      let r_new = PCS::<E>::blind(ck, S.num_precommitted);
      Some(WitnessCommitment {
        comm: PCS::<E>::rerandomize_commitment(ck, &wc.comm, &wc.blind, &r_new)?,
        blind: r_new,
      })
    } else {
      None
    };

    Ok(SpartanPrepSNARK {
      cs: self.cs.clone(),
      shared: self.shared.clone(),
      precommitted: self.precommitted.clone(),
      comm_shared: shared.clone(),
      comm_precommitted: comm_precommitted_new,
      W: self.W.clone(),
    })
  }
}

impl<E: Engine> MultiRoundSpartanShape<E> for ShapeCS<E> {
  fn multiround_r1cs_shape<C: MultiRoundCircuit<E>>(
    circuit: &C,
  ) -> Result<
    (
      SplitMultiRoundR1CSShape<E>,
      CommitmentKey<E>,
      VerifierKey<E>,
    ),
    SpartanError,
  > {
    let num_rounds = circuit.num_rounds();
    let mut cs: Self = Self::new();

    // Collect variables and challenges per round
    let mut vars_per_round: Vec<Vec<AllocatedNum<E::Scalar>>> = Vec::new();
    let mut challenges_per_round: Vec<Vec<AllocatedNum<E::Scalar>>> = Vec::new();
    let mut num_vars_per_round: Vec<usize> = Vec::new();
    let mut num_challenges_per_round: Vec<usize> = Vec::new();

    // Process each round to collect shape information
    for round in 0..num_rounds {
      let num_challenges =
        circuit
          .num_challenges(round)
          .map_err(|e| SpartanError::SynthesisError {
            reason: format!("Unable to get num_challenges for round {round}: {e}"),
          })?;
      num_challenges_per_round.push(num_challenges);

      // For shape generation, we don't need actual challenge values, so pass None
      let prev_aux = cs.num_aux();
      let (round_vars, round_challenges) = circuit
        .rounds(&mut cs, round, &vars_per_round, &challenges_per_round, None)
        .map_err(|e| SpartanError::SynthesisError {
          reason: format!("Unable to synthesize round {round}: {e}"),
        })?;

      // Determine how many new auxiliary variables were allocated in this round.
      let new_aux = cs.num_aux();
      num_vars_per_round.push(new_aux - prev_aux);

      // Record the high-level variables/challenges that the round chose to return so
      // they can be fed into subsequent rounds.
      vars_per_round.push(round_vars);
      challenges_per_round.push(round_challenges);
    }

    let mut A = SparseMatrix::<E::Scalar>::empty();
    let mut B = SparseMatrix::<E::Scalar>::empty();
    let mut C = SparseMatrix::<E::Scalar>::empty();

    let mut num_cons_added = 0;
    let mut x = (&mut A, &mut B, &mut C, &mut num_cons_added);
    let num_inputs = cs.num_inputs();
    let num_constraints = cs.num_constraints();
    let total_vars = cs.num_aux();

    for constraint in cs.constraints.iter() {
      add_constraint(
        &mut x,
        total_vars,
        &constraint.0,
        &constraint.1,
        &constraint.2,
      );
    }
    assert_eq!(num_cons_added, num_constraints);

    A.cols = total_vars + num_inputs;
    B.cols = total_vars + num_inputs;
    C.cols = total_vars + num_inputs;

    // Don't count One as an input for shape's purposes.
    let num_public_values = num_inputs - 1 - num_challenges_per_round.iter().sum::<usize>();
    let S = SplitMultiRoundR1CSShape::new(
      MULTIROUND_COMMITMENT_WIDTH,
      num_constraints,
      num_vars_per_round,
      num_challenges_per_round,
      num_public_values,
      A,
      B,
      C,
    )?;

    let (ck, vk) = S.commitment_key();

    Ok((S, ck, vk))
  }
}

/// A type that holds the multi-round state for proving
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct MultiRoundState<E: Engine> {
  cs: SatisfyingAssignment<E>,
  vars_per_round: Vec<Vec<AllocatedNum<E::Scalar>>>,
  challenges_per_round: Vec<Vec<AllocatedNum<E::Scalar>>>,
  challenges: Vec<Vec<E::Scalar>>,
  comm_w_per_round: Vec<Commitment<E>>,
  pub(crate) r_w_per_round: Vec<Blind<E>>,
  w: Vec<E::Scalar>,
  current_round: usize,
  num_rounds: usize,
}

impl<E: Engine> MultiRoundSpartanWitness<E> for SatisfyingAssignment<E> {
  type MultiRoundState = MultiRoundState<E>;

  fn initialize_multiround_witness(
    s: &SplitMultiRoundR1CSShape<E>,
  ) -> Result<Self::MultiRoundState, SpartanError> {
    let cs = Self::new();
    let total_vars: usize = s.num_vars_per_round.iter().sum();
    let w = vec![E::Scalar::ZERO; total_vars];

    Ok(MultiRoundState {
      cs,
      vars_per_round: Vec::new(),
      challenges_per_round: Vec::new(),
      challenges: Vec::new(),
      comm_w_per_round: Vec::new(),
      r_w_per_round: Vec::new(),
      w,
      current_round: 0,
      num_rounds: s.num_rounds,
    })
  }

  fn process_round<C: MultiRoundCircuit<E>>(
    state: &mut Self::MultiRoundState,
    s: &SplitMultiRoundR1CSShape<E>,
    ck: &CommitmentKey<E>,
    circuit: &C,
    round_index: usize,
    transcript: &mut E::TE,
  ) -> Result<Vec<E::Scalar>, SpartanError> {
    if round_index != state.current_round {
      return Err(SpartanError::SynthesisError {
        reason: format!(
          "Expected round {}, got {}",
          state.current_round, round_index
        ),
      });
    }

    // Process this round, supplying references to variables/challenges from previous rounds
    let chals = if round_index == 0 {
      None
    } else {
      state.challenges.get(round_index - 1).map(|c| c.as_slice())
    };

    let (round_vars, round_challenges) = circuit
      .rounds(
        &mut state.cs,
        round_index,
        &state.vars_per_round,
        &state.challenges_per_round,
        chals,
      )
      .map_err(|e| SpartanError::SynthesisError {
        reason: format!("Unable to synthesize round {round_index}: {e}"),
      })?;

    // Update witness with new variables
    // We need to map the *unpadded* auxiliary variables produced by this round
    // into the appropriate *padded* segment of the global witness vector.
    // 1. `start_idx_unpadded` – offset within `aux_assignment` (contains only the
    //    actual variables produced so far).
    // 2. `start_idx_padded`   – offset within the global witness vector (which
    //    has per-round padding applied).
    let start_idx_unpadded: usize = s.num_vars_per_round_unpadded[..round_index].iter().sum();
    let start_idx_padded: usize = s.num_vars_per_round[..round_index].iter().sum();
    let round_vars_unpadded = s.num_vars_per_round_unpadded[round_index];

    // Copy only the actually-allocated variables; leave the padding slots as 0.
    if state.cs.aux_assignment.len() >= start_idx_unpadded + round_vars_unpadded {
      state.w[start_idx_padded..start_idx_padded + round_vars_unpadded].copy_from_slice(
        &state.cs.aux_assignment[start_idx_unpadded..start_idx_unpadded + round_vars_unpadded],
      );
    }

    // Commit to this round's variables
    let start_padded: usize = s.num_vars_per_round[..round_index].iter().sum();
    let r_w_per_round = PCS::<E>::blind(ck, s.num_vars_per_round[round_index]);
    let comm_w_round = PCS::<E>::commit(
      ck,
      &state.w[start_padded..start_padded + s.num_vars_per_round[round_index]],
      &r_w_per_round,
      false,
    )?;

    // absorb commitment to this round's variables
    transcript.absorb(b"comm_w_round", &comm_w_round);

    // Generate challenges for this round
    let challenges = (0..s.num_challenges_per_round[round_index])
      .map(|_| transcript.squeeze(b"challenge"))
      .collect::<Result<Vec<E::Scalar>, SpartanError>>()?;

    state.vars_per_round.push(round_vars);
    state.challenges_per_round.push(round_challenges);
    state.comm_w_per_round.push(comm_w_round);
    state.r_w_per_round.push(r_w_per_round);
    state.challenges.push(challenges.clone());
    state.current_round += 1;

    // Return the challenges that were generated for this round
    Ok(challenges)
  }

  fn finalize_multiround_witness(
    state: &mut Self::MultiRoundState,
    s: &SplitMultiRoundR1CSShape<E>,
  ) -> Result<(SplitMultiRoundR1CSInstance<E>, R1CSWitness<E>), SpartanError> {
    if state.current_round != state.num_rounds {
      return Err(SpartanError::SynthesisError {
        reason: format!(
          "Expected {} rounds, processed {}",
          state.num_rounds, state.current_round
        ),
      });
    }

    let num_challenges: usize = s.num_challenges_per_round.iter().sum();

    // collect public values
    let public_values = state.cs.input_assignment[1 + num_challenges..].to_vec();

    let u = SplitMultiRoundR1CSInstance::<E>::new(
      s,
      state.comm_w_per_round.clone(),
      public_values,
      state.challenges.clone(),
    )?;

    let r_w = PCS::<E>::combine_blinds(&state.r_w_per_round)?;
    let w = R1CSWitness::<E>::new_unchecked(state.w.clone(), r_w, false)?;

    Ok((u, w))
  }
}

impl<E: Engine, W, Coeff> SmallSpartanWitness<E, W, Coeff> for SmallSatisfyingAssignment<W>
where
  W: Copy + Default + PartialEq + From<bool>,
{
  type SmallPrepState = SmallPrepSNARK<E, W>;

  fn shared_witness<C: SmallSpartanCircuit<E, W>>(
    S: &SplitR1CSShape<E, Coeff>,
    ck: &CommitmentKey<E>,
    circuit: &C,
  ) -> Result<SmallPrepSNARK<E, W>, SpartanError> {
    use tracing::info;

    let (_synth_span, synth_t) = start_span!("shared_witness_synthesize");

    let mut cs = SmallSatisfyingAssignment::<W>::new();

    let shared = circuit
      .shared(&mut cs)
      .map_err(|e| SpartanError::SynthesisError {
        reason: format!("shared_witness: shared: {e}"),
      })?;

    let num_vars = S.num_shared + S.num_precommitted + S.num_rest;
    let mut witness = vec![W::default(); num_vars];
    let shared_copy = cs.aux_assignment.len().min(S.num_shared_unpadded);
    witness[..shared_copy].copy_from_slice(&cs.aux_assignment[..shared_copy]);

    let zero_w = W::default();
    let comm_shared = if S.num_shared_unpadded > 0 {
      let blind = PCS::<E>::blind(ck, S.num_shared);
      let w_bool: Vec<bool> = witness[..S.num_shared]
        .iter()
        .map(|v| *v != zero_w)
        .collect();
      let comm = PCS::<E>::commit_witness(ck, &w_bool, &blind)?;
      Some(WitnessCommitment { comm, blind })
    } else {
      None
    };

    info!(elapsed_ms = %synth_t.elapsed().as_millis(), "shared_witness_synthesize");

    Ok(SmallPrepSNARK {
      cs,
      shared,
      precommitted: vec![],
      comm_shared,
      comm_precommitted: None,
      W: witness,
    })
  }

  fn precommitted_witness<C: SmallSpartanCircuit<E, W>>(
    prep: &mut SmallPrepSNARK<E, W>,
    S: &SplitR1CSShape<E, Coeff>,
    ck: &CommitmentKey<E>,
    circuit: &C,
  ) -> Result<(), SpartanError> {
    use tracing::info;

    let (_synth_span, synth_t) = start_span!("precommitted_witness_synthesize");

    let precommitted = circuit
      .precommitted(&mut prep.cs, &prep.shared)
      .map_err(|e| SpartanError::SynthesisError {
        reason: format!("precommitted_witness: precommitted: {e}"),
      })?;

    let precommitted_start_aux = S.num_shared_unpadded;
    let precommitted_copy = (prep
      .cs
      .aux_assignment
      .len()
      .saturating_sub(precommitted_start_aux))
    .min(S.num_precommitted_unpadded);
    let dst_start = S.num_shared;
    prep.W[dst_start..dst_start + precommitted_copy].copy_from_slice(
      &prep.cs.aux_assignment[precommitted_start_aux..precommitted_start_aux + precommitted_copy],
    );

    info!(elapsed_ms = %synth_t.elapsed().as_millis(), "precommitted_witness_synthesize");

    let (_commit_pre_span, commit_pre_t) = start_span!("commit_witness_precommitted");
    let zero_w = W::default();
    prep.comm_precommitted = if S.num_precommitted_unpadded > 0 {
      let blind = PCS::<E>::blind(ck, S.num_precommitted);
      let w_bool: Vec<bool> = prep.W[S.num_shared..S.num_shared + S.num_precommitted]
        .iter()
        .map(|v| *v != zero_w)
        .collect();
      let comm = PCS::<E>::commit_witness(ck, &w_bool, &blind)?;
      Some(WitnessCommitment { comm, blind })
    } else {
      None
    };
    info!(elapsed_ms = %commit_pre_t.elapsed().as_millis(), "commit_witness_precommitted");

    prep.precommitted = precommitted;

    Ok(())
  }
}
