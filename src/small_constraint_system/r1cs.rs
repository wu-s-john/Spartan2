// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! R1CS witness preparation for the small-value (integer) proving path.
//!
//! Contains `SmallSpartanWitness` (trait), `SmallPrepSNARK` (struct),
//! `small_r1cs_shape` (fn), and the impl of `SmallSpartanWitness` for
//! `SmallSatisfyingAssignment`.

use bellpepper_core::Variable;
use tracing::info;

use crate::{
  Blind, CommitmentKey, PCS,
  bellpepper::r1cs::WitnessCommitment,
  errors::SpartanError,
  r1cs::{SplitR1CSInstance, SplitR1CSShape},
  small_constraint_system::{
    SmallCoeff, SmallSatisfyingAssignment, SmallShapeCS,
    circuit::SmallSpartanCircuit,
  },
  start_span,
  traits::{Engine, pcs::PCSEngineTrait, transcript::TranscriptEngineTrait},
};

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

  /// Synthesizes the rest of the circuit, commits the rest segment, and returns
  /// the R1CS instance and combined blind.
  ///
  /// Mirrors `SpartanWitness::r1cs_instance_and_witness` for the integer path:
  /// absorbs commitments → squeezes challenges → synthesizes rest → commits rest → combines blinds.
  ///
  /// `public_values_field` are the field-typed public values (already absorbed into transcript).
  fn r1cs_instance_and_witness<C: SmallSpartanCircuit<E, W>>(
    prep: &mut Self::SmallPrepState,
    S: &SplitR1CSShape<E, Coeff>,
    ck: &CommitmentKey<E>,
    circuit: &C,
    public_values_field: Vec<E::Scalar>,
    transcript: &mut E::TE,
  ) -> Result<(SplitR1CSInstance<E>, Blind<E>), SpartanError>;
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

/// Extract a `SplitR1CSShape<E, Coeff>` from a `SmallSpartanCircuit`.
///
/// Uses `SmallShapeCS<Coeff>` to record small-coefficient constraints directly,
/// without creating any field elements.
pub fn small_r1cs_shape<E: Engine, Coeff: SmallCoeff, Circuit: SmallSpartanCircuit<E, Coeff>>(
  circuit: &Circuit,
) -> Result<SplitR1CSShape<E, Coeff>, SpartanError> {
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

  fn r1cs_instance_and_witness<C: SmallSpartanCircuit<E, W>>(
    prep: &mut SmallPrepSNARK<E, W>,
    S: &SplitR1CSShape<E, Coeff>,
    ck: &CommitmentKey<E>,
    circuit: &C,
    public_values_field: Vec<E::Scalar>,
    transcript: &mut E::TE,
  ) -> Result<(SplitR1CSInstance<E>, Blind<E>), SpartanError> {
    let (_sat_span, sat_t) = start_span!("r1cs_instance_and_witness");
    let zero_w = W::default();

    // Absorb shared/precommitted commitments into transcript
    if let Some(ref wc) = prep.comm_shared {
      transcript.absorb(b"comm_W_shared", &wc.comm);
    }
    if let Some(ref wc) = prep.comm_precommitted {
      transcript.absorb(b"comm_W_precommitted", &wc.comm);
    }

    // Squeeze challenges from transcript
    let challenges: Vec<E::Scalar> = (0..C::num_challenges(circuit))
      .map(|_| transcript.squeeze(b"c"))
      .collect::<Result<Vec<_>, _>>()?;

    // Synthesize rest of the circuit
    circuit
      .synthesize(
        &mut prep.cs,
        &prep.shared,
        &prep.precommitted,
        Some(&challenges),
      )
      .map_err(|e| SpartanError::SynthesisError {
        reason: format!("r1cs_instance_and_witness: synthesize: {e}"),
      })?;

    // Copy rest witness into W
    let rest_start_aux = S.num_shared_unpadded + S.num_precommitted_unpadded;
    let rest_copy =
      (prep.cs.aux_assignment.len().saturating_sub(rest_start_aux)).min(S.num_rest_unpadded);
    let dst_rest = S.num_shared + S.num_precommitted;
    prep.W[dst_rest..dst_rest + rest_copy]
      .copy_from_slice(&prep.cs.aux_assignment[rest_start_aux..rest_start_aux + rest_copy]);
    info!(elapsed_ms = %sat_t.elapsed().as_millis(), "r1cs_instance_and_witness");

    // Commit rest segment
    let (_commit_rest_span, commit_rest_t) = start_span!("commit_witness_rest");
    let r_W_rest = PCS::<E>::blind(ck, S.num_rest);
    let w_rest_bool: Vec<bool> = prep.W[S.num_shared + S.num_precommitted..]
      .iter()
      .map(|v| *v != zero_w)
      .collect();
    let comm_W_rest = PCS::<E>::commit_witness(ck, &w_rest_bool, &r_W_rest)?;
    info!(elapsed_ms = %commit_rest_t.elapsed().as_millis(), "commit_witness_rest");
    transcript.absorb(b"comm_W_rest", &comm_W_rest);

    // Combine blinds
    let mut blinds = Vec::with_capacity(3);
    if let Some(ref wc) = prep.comm_shared {
      blinds.push(wc.blind.clone());
    }
    if let Some(ref wc) = prep.comm_precommitted {
      blinds.push(wc.blind.clone());
    }
    blinds.push(r_W_rest);
    let r_W = PCS::<E>::combine_blinds(&blinds)?;

    // Build instance
    let U = SplitR1CSInstance::<E> {
      comm_W_shared: prep.comm_shared.as_ref().map(|wc| wc.comm.clone()),
      comm_W_precommitted: prep.comm_precommitted.as_ref().map(|wc| wc.comm.clone()),
      comm_W_rest,
      public_values: public_values_field,
      challenges,
    };

    Ok((U, r_W))
  }
}
