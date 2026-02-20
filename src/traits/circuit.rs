// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! This module defines traits that a circuit provider must implement to be used with Spartan.
use crate::traits::Engine;
use crate::{CommitmentKey, errors::SpartanError, r1cs::{R1CSInstance, R1CSWitness}};
use crate::small_r1cs::{SmallCS, SynthesisError as SmallSynthesisError};
use bellpepper_core::{ConstraintSystem, SynthesisError, num::AllocatedNum};

/// A helper trait for defining a randomized circuit that Spartan proves.
/// The circuit contains a set of variables that are shared with other circuits.
/// For simplicity, we only allow one round of interaction, so there is a single list of precommitted variables.
/// (1) the prover commits to precommitted variables (and variables that will be made public via inputize)
/// (2) the verifier challenges with a random value
/// (3) the prover commits to aux variables
/// (4) The circuit checks a set of constraints over witness = (shared, precommitted, aux)
/// The public IO includes the challenge and other things made public by the circuit
pub trait SpartanCircuit<E: Engine>: Send + Sync + Clone {
  /// Returns the public values of the circuit, which is the list of values that will be made public
  /// The circuit must make public these values followed by the challenges generated via the transcript
  fn public_values(&self) -> Result<Vec<E::Scalar>, SynthesisError>;

  /// Allocated variables in the circuit that are shared with other circuits
  fn shared<CS: ConstraintSystem<E::Scalar>>(
    &self,
    cs: &mut CS,
  ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError>;

  /// Allocates precommitted variables
  fn precommitted<CS: ConstraintSystem<E::Scalar>>(
    &self,
    cs: &mut CS,
    shared: &[AllocatedNum<E::Scalar>],
  ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError>;

  /// Returns the number of challenges that the circuit expects from the verifier
  /// for randomized checks added in synthesize
  fn num_challenges(&self) -> usize;

  /// Allocate the rest of the variables and constraints in the circuit.
  /// The `shared` and `precommitted` variables are already allocated.
  /// The `challenges` are the challenges from the verifier, if any (these must be allocated as public IO prior to other public IO).
  /// The circuit should use these challenges to synthesize the rest of the circuit.
  /// The circuit should return an error if it cannot synthesize the circuit.
  fn synthesize<CS: ConstraintSystem<E::Scalar>>(
    &self,
    cs: &mut CS,
    shared: &[AllocatedNum<E::Scalar>],
    precommitted: &[AllocatedNum<E::Scalar>],
    challenges: Option<&[E::Scalar]>, // challenges from the verifier
  ) -> Result<(), SynthesisError>;
}

/// A helper trait for defining a multi-round randomized circuit that Spartan proves.
/// Unlike the standard SpartanCircuit, this trait allows the circuit to be processed in multiple rounds,
/// where each round can allocate different constraints and witness variables based on the round index.
/// The public IO includes the challenges and other things made public by the circuit across all rounds.
pub trait MultiRoundCircuit<E: Engine>: Send + Sync + Clone {
  /// Returns the number of challenges that the circuit expects from the verifier
  /// for randomized checks added in the round `round_index`
  fn num_challenges(&self, round_index: usize) -> Result<usize, SynthesisError>;

  /// Processes a specific round of the circuit.
  /// The `round_index` determines which round is being processed, and the function branches
  /// based on this index to allocate the appropriate constraints and witness variables for that round.
  /// The `prior_round_vars` are variables allocated in rounds 0..round_index-1.
  /// The `prev_challenges` are challenges allocated in rounds 0..round_index-1.
  /// The `challenges` are the challenges for this round to be allocated by this round and returned.
  /// The circuit should return an error if it cannot synthesize the specified round.
  /// Returns a tuple of (round_vars, allocated_challenges) where:
  /// - round_vars: variables allocated in this round (excluding challenges)
  /// - allocated_challenges: challenge variables allocated in this round that will be passed to next round
  fn rounds<CS: ConstraintSystem<E::Scalar>>(
    &self,
    cs: &mut CS,
    round_index: usize,
    prior_round_vars: &[Vec<AllocatedNum<E::Scalar>>], // variables allocated in rounds 0..round_index-1 grouped per round
    prev_challenges: &[Vec<AllocatedNum<E::Scalar>>], // challenges allocated in rounds 0..round_index-1 grouped per round
    challenges: Option<&[E::Scalar]>, // challenges for this round to be allocated by this round and returned
  ) -> Result<(Vec<AllocatedNum<E::Scalar>>, Vec<AllocatedNum<E::Scalar>>), SynthesisError>;

  /// Returns the number of rounds in the circuit
  fn num_rounds(&self) -> usize;
}

/// A trait for circuits that synthesize into `SmallCS<i32, i64>` for native small-value proving.
///
/// This trait enables circuits to use native integer arithmetic during synthesis instead of
/// field element arithmetic, providing significant performance benefits for circuits with
/// small witness values (like SHA-256 where witnesses are bits).
///
/// # Type Parameters
/// - Witnesses are `i32` (bits, small integers)
/// - Matrix coefficients are `i64` (for optimized constraint batching)
///
/// # Example
/// ```ignore
/// impl<E: Engine> NativeSmallCircuit<E> for MySha256Circuit
/// where
///     E::Scalar: SmallValueField<i32>,
/// {
///     fn synthesize_i64(&self, cs: &mut SmallCS<i32, i64>) -> Result<(), SmallSynthesisError> {
///         // Allocate variables and constraints using native i32/i64 arithmetic
///     }
///
///     fn to_witness_and_instance_i64(&self, ck: &CommitmentKey<E>, num_rest_padded: usize)
///         -> Result<(R1CSWitness<E, i32>, R1CSInstance<E, i32>), SpartanError> {
///         // Synthesize and commit
///     }
/// }
/// ```
pub trait NativeSmallCircuit<E: Engine>: Send + Sync + Clone {
  /// Synthesize the circuit into `SmallCS<i32, i64>`.
  ///
  /// Uses native i32 witnesses and i64 coefficients for efficient constraint representation.
  fn synthesize_i64(&self, cs: &mut SmallCS<i32, i64>) -> Result<(), SmallSynthesisError>;

  /// Create witness and instance by synthesizing and committing.
  ///
  /// This method:
  /// 1. Synthesizes into `SmallCS<i32, i64>`
  /// 2. Extracts `Vec<i32>` witnesses
  /// 3. Commits using `commit_small_direct` (no field conversion)
  /// 4. Returns `R1CSWitness<E, i32>` and `R1CSInstance<E, i32>`
  fn to_witness_and_instance_i64(
    &self,
    ck: &CommitmentKey<E>,
    num_rest_padded: usize,
  ) -> Result<(R1CSWitness<E, i32>, R1CSInstance<E, i32>), SpartanError>;
}
