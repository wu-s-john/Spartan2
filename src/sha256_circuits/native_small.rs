// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Native small-value SHA-256 chain circuit.
//!
//! Uses `SmallCS<i32, i32>` + `small_gadgets/sha256` for true native small-value synthesis.
//! Produces `R1CSWitness<E, i32>` and `R1CSInstance<E, i32>` directly.

use sha2::{Digest, Sha256};

use crate::{
  CommitmentKey,
  errors::SpartanError,
  r1cs::{R1CSInstance, R1CSWitness, SplitR1CSShape},
  small_field::SmallValueField,
  small_gadgets::{Boolean, small_sha256_batched, small_sha256_batched_i64},
  small_r1cs::{SmallCS, SynthesisError as SmallSynthesisError},
  traits::{Engine, pcs::PCSEngineTrait},
};

/// Native small-value SHA-256 chain circuit.
///
/// Uses `SmallCS<i32, i32>` + `small_gadgets/sha256` for pure native integer synthesis.
/// No field conversions during synthesis - witness values are i32 throughout.
#[derive(Debug, Clone)]
pub struct NativeSmallSha256ChainCircuit {
  /// 32-byte (256-bit) input to start the chain
  pub input: [u8; 32],
  /// Number of SHA-256 hashes in the chain
  pub chain_length: usize,
}

impl NativeSmallSha256ChainCircuit {
  /// Create a new native SHA-256 chain circuit.
  pub fn new(input: [u8; 32], chain_length: usize) -> Self {
    Self { input, chain_length }
  }

  /// Compute the expected final hash by applying SHA-256 chain_length times.
  pub fn expected_output(&self) -> [u8; 32] {
    let mut current = self.input;
    for _ in 0..self.chain_length {
      let mut hasher = Sha256::new();
      hasher.update(current);
      current = hasher.finalize().into();
    }
    current
  }

  /// Synthesize into SmallCS<i32, i32>, producing i32 witnesses.
  ///
  /// This is pure native synthesis - no field conversions.
  pub fn synthesize(&self, cs: &mut SmallCS<i32, i32>) -> Result<(), SmallSynthesisError> {
    // Allocate input bytes as Boolean<i32, i32> bits (big-endian per byte)
    let mut current_bits: Vec<Boolean<i32, i32>> = Vec::with_capacity(256);
    for &byte in &self.input {
      for i in (0..8).rev() {
        let bit_val = ((byte >> i) & 1) != 0;
        current_bits.push(Boolean::alloc(cs, Some(bit_val))?);
      }
    }

    // Chain SHA-256 hashes (using batched version for reduced constraints)
    for _ in 0..self.chain_length {
      current_bits = small_sha256_batched(cs, &current_bits)?;
    }

    // Verify against expected output
    let expected = self.expected_output();
    let expected_bits: Vec<bool> = expected
      .iter()
      .flat_map(|&byte| (0..8).rev().map(move |i| ((byte >> i) & 1) != 0))
      .collect();

    for (i, (computed, &expected_bit)) in current_bits.iter().zip(expected_bits.iter()).enumerate()
    {
      let computed_val = computed.value.unwrap_or(false);
      assert_eq!(
        computed_val, expected_bit,
        "Hash bit {} mismatch: computed={}, expected={}",
        i, computed_val, expected_bit
      );
    }

    // Expose hash bits as public inputs
    for bit in &current_bits {
      bit.inputize(cs)?;
    }

    Ok(())
  }

  /// Build the R1CS shape with i32 coefficients.
  ///
  /// Returns `SplitR1CSShape<E, i32>` for use with `prove_native`.
  pub fn to_shape<E: Engine>(&self) -> SplitR1CSShape<E, i32> {
    let mut cs = SmallCS::<i32, i32>::new();
    self.synthesize(&mut cs).expect("Synthesis failed");
    cs.to_split_r1cs_shape()
  }

  /// Build R1CSWitness<E, i32> and R1CSInstance<E, i32> from synthesis.
  ///
  /// This is the main entry point for creating small-value witnesses and instances.
  /// Field conversion happens only at commitment time.
  ///
  /// The `num_rest_padded` parameter should match the padded witness size expected
  /// by the shape (typically `shape.num_rest`).
  pub fn to_witness_and_instance<E: Engine>(
    &self,
    ck: &CommitmentKey<E>,
    num_rest_padded: usize,
  ) -> Result<(R1CSWitness<E, i32>, R1CSInstance<E, i32>), SpartanError>
  where
    E::Scalar: SmallValueField<i32>,
  {
    let mut cs = SmallCS::<i32, i32>::new();
    self
      .synthesize(&mut cs)
      .map_err(|e| SpartanError::SynthesisError { reason: format!("{:?}", e) })?;

    let mut W: Vec<i32> = cs.witness_values();
    let X: Vec<i32> = cs.public_values();

    // Pad witness to match shape's padded size
    W.resize(num_rest_padded, 0);

    // Create blind and commit using commit_small_direct - passes i32 directly to MSM
    // This avoids the field conversion overhead entirely (no i32 → field → u64 roundtrip)
    let r_W = E::PCS::blind(ck, W.len());
    let comm_W = E::PCS::commit_small_direct(ck, &W, &r_W)?;

    Ok((
      R1CSWitness { W, r_W },
      R1CSInstance::new_unchecked_generic(comm_W, X),
    ))
  }

  // ========================================
  // i64 coefficient path (optimized constraints)
  // ========================================

  /// Synthesize into SmallCS<i32, i64> with optimized constraints.
  ///
  /// Uses full 35-bit addition and K=21 batching for optimal constraint count.
  /// Witnesses are i32, but matrix coefficients are i64.
  pub fn synthesize_i64(&self, cs: &mut SmallCS<i32, i64>) -> Result<(), SmallSynthesisError> {
    // Allocate input bytes as Boolean<i32, i64> bits (big-endian per byte)
    let mut current_bits: Vec<Boolean<i32, i64>> = Vec::with_capacity(256);
    for &byte in &self.input {
      for i in (0..8).rev() {
        let bit_val = ((byte >> i) & 1) != 0;
        current_bits.push(Boolean::alloc(cs, Some(bit_val))?);
      }
    }

    // Chain SHA-256 hashes (using i64 optimized version)
    for _ in 0..self.chain_length {
      current_bits = small_sha256_batched_i64(cs, &current_bits)?;
    }

    // Verify against expected output
    let expected = self.expected_output();
    let expected_bits: Vec<bool> = expected
      .iter()
      .flat_map(|&byte| (0..8).rev().map(move |i| ((byte >> i) & 1) != 0))
      .collect();

    for (i, (computed, &expected_bit)) in current_bits.iter().zip(expected_bits.iter()).enumerate()
    {
      let computed_val = computed.value.unwrap_or(false);
      assert_eq!(
        computed_val, expected_bit,
        "Hash bit {} mismatch: computed={}, expected={}",
        i, computed_val, expected_bit
      );
    }

    // Expose hash bits as public inputs
    for bit in &current_bits {
      bit.inputize(cs)?;
    }

    Ok(())
  }

  /// Build the R1CS shape with i64 coefficients.
  ///
  /// Returns `SplitR1CSShape<E, i64>` with optimized constraint count.
  pub fn to_shape_i64<E: Engine>(&self) -> SplitR1CSShape<E, i64> {
    let mut cs = SmallCS::<i32, i64>::new();
    self.synthesize_i64(&mut cs).expect("Synthesis failed");
    cs.to_split_r1cs_shape()
  }

  /// Build R1CSWitness<E, i32> and R1CSInstance<E, i32> using i64 coefficient path.
  ///
  /// Uses the optimized i64 constraint system but returns i32 witnesses
  /// (since witness values are still i32 bits).
  pub fn to_witness_and_instance_i64<E: Engine>(
    &self,
    ck: &CommitmentKey<E>,
    num_rest_padded: usize,
  ) -> Result<(R1CSWitness<E, i32>, R1CSInstance<E, i32>), SpartanError>
  where
    E::Scalar: SmallValueField<i32>,
  {
    let mut cs = SmallCS::<i32, i64>::new();
    self
      .synthesize_i64(&mut cs)
      .map_err(|e| SpartanError::SynthesisError { reason: format!("{:?}", e) })?;

    let mut W: Vec<i32> = cs.witness_values();
    let X: Vec<i32> = cs.public_values();

    // Pad witness to match shape's padded size
    W.resize(num_rest_padded, 0);

    // Create blind and commit using commit_small_direct - passes i32 directly to MSM
    let r_W = E::PCS::blind(ck, W.len());
    let comm_W = E::PCS::commit_small_direct(ck, &W, &r_W)?;

    Ok((
      R1CSWitness { W, r_W },
      R1CSInstance::new_unchecked_generic(comm_W, X),
    ))
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::Bn254Engine;
  use crate::small_r1cs::SmallConstraintSystem;

  type E = Bn254Engine;

  #[test]
  fn test_native_small_sha256_chain_synthesis() {
    let input = [0u8; 32];
    let circuit = NativeSmallSha256ChainCircuit::new(input, 1);

    let mut cs = SmallCS::<i32, i32>::new();
    circuit.synthesize(&mut cs).expect("Synthesis should succeed");

    // Check that constraints are satisfied
    assert!(cs.is_satisfied::<i64>(), "Constraints should be satisfied");

    // Check that we have the expected number of public inputs (256 bits)
    assert_eq!(cs.num_inputs() - 1, 256, "Should have 256 public input bits");

    println!("Native batched SHA-256 constraints: {}", cs.num_constraints());
  }

  #[test]
  fn test_native_small_sha256_chain_shape() {
    let input = [0u8; 32];
    let circuit = NativeSmallSha256ChainCircuit::new(input, 1);

    let shape: SplitR1CSShape<E, i32> = circuit.to_shape();

    // Verify shape has expected structure
    assert!(shape.num_cons > 0, "Should have constraints");
    assert_eq!(shape.num_public, 256, "Should have 256 public inputs");
  }
}
