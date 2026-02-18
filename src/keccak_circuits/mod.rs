// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT

//! Keccak-256 chain circuit for benchmarking small-value sumcheck.
//!
//! Uses bellpepper-keccak which already has small coefficients (0, 1, -1, 2).

use bellpepper_core::{
  Circuit, ConstraintSystem, SynthesisError,
  boolean::{AllocatedBit, Boolean},
  num::AllocatedNum,
};
use bellpepper_keccak::keccak256;
use ff::{Field, PrimeField, PrimeFieldBits};
use sha3::{Digest, Keccak256};
use std::marker::PhantomData;

use crate::traits::{Engine, circuit::SpartanCircuit};

/// Keccak-256 chain circuit using bellpepper-keccak.
///
/// Chains `chain_length` Keccak-256 hashes starting from a 64-byte input.
/// Hash[0] = Keccak256(input), Hash[i] = Keccak256(Hash[i-1] || zeros)
#[derive(Debug, Clone)]
pub struct KeccakChainCircuit<Scalar: PrimeField> {
  /// 64-byte (512-bit) input to start the chain
  pub input: [u8; 64],
  /// Number of Keccak-256 hashes in the chain
  pub chain_length: usize,
  _p: PhantomData<Scalar>,
}

impl<Scalar: PrimeField + PrimeFieldBits> KeccakChainCircuit<Scalar> {
  /// Create a new Keccak-256 chain circuit.
  pub fn new(input: [u8; 64], chain_length: usize) -> Self {
    Self {
      input,
      chain_length,
      _p: PhantomData,
    }
  }

  /// Compute the expected final hash by applying Keccak-256 chain_length times.
  pub fn expected_output(&self) -> [u8; 32] {
    let mut current: [u8; 32] = {
      let mut hasher = Keccak256::new();
      hasher.update(&self.input);
      let result = hasher.finalize();
      result.as_slice().try_into().unwrap()
    };

    for _ in 1..self.chain_length {
      // Pad 32 bytes to 64 bytes with zeros
      let mut padded = [0u8; 64];
      padded[..32].copy_from_slice(&current);
      let mut hasher = Keccak256::new();
      hasher.update(&padded);
      let result = hasher.finalize();
      current = result.as_slice().try_into().unwrap();
    }
    current
  }
}

/// Convert bytes to public value scalars (one field element per bit).
fn bytes_to_public_scalars<F: PrimeField>(bytes: &[u8]) -> Vec<F> {
  bytes
    .iter()
    .flat_map(|&byte| {
      (0..8).rev().map(move |i| {
        if (byte >> i) & 1 == 1 {
          F::ONE
        } else {
          F::ZERO
        }
      })
    })
    .collect()
}

/// Allocate bytes as witness bits (big-endian per byte).
fn alloc_input_bits<Scalar, CS>(
  cs: &mut CS,
  bytes: &[u8],
  prefix: &str,
) -> Result<Vec<Boolean>, SynthesisError>
where
  Scalar: PrimeField,
  CS: ConstraintSystem<Scalar>,
{
  bytes
    .iter()
    .enumerate()
    .flat_map(|(byte_idx, &byte)| {
      (0..8)
        .rev()
        .enumerate()
        .map(move |(bit_idx, i)| (byte_idx, bit_idx, (byte >> i) & 1 == 1))
    })
    .map(|(byte_idx, bit_idx, bit_val)| {
      AllocatedBit::alloc(
        cs.namespace(|| format!("{}_byte{}_bit{}", prefix, byte_idx, bit_idx)),
        Some(bit_val),
      )
      .map(Boolean::from)
    })
    .collect()
}

/// Pad 256 bits to 512 bits with zeros.
fn pad_256_to_512(bits: &[Boolean]) -> Vec<Boolean> {
  assert_eq!(bits.len(), 256);
  let mut padded = bits.to_vec();
  for _ in 0..256 {
    padded.push(Boolean::constant(false));
  }
  padded
}

/// Expose hash bits as public inputs.
fn expose_hash_bits_as_public<E, CS>(
  cs: &mut CS,
  hash_bits: &[Boolean],
) -> Result<(), SynthesisError>
where
  E: Engine,
  CS: ConstraintSystem<E::Scalar>,
{
  for (i, bit) in hash_bits.iter().enumerate() {
    let n = AllocatedNum::alloc_input(cs.namespace(|| format!("public_{i}")), || {
      Ok(
        if bit.get_value().ok_or(SynthesisError::AssignmentMissing)? {
          E::Scalar::ONE
        } else {
          E::Scalar::ZERO
        },
      )
    })?;

    cs.enforce(
      || format!("bit_eq_{i}"),
      |_| bit.lc(CS::one(), E::Scalar::ONE),
      |lc| lc + CS::one(),
      |lc| lc + n.get_variable(),
    );
  }
  Ok(())
}

impl<E: Engine> SpartanCircuit<E> for KeccakChainCircuit<E::Scalar>
where
  E::Scalar: PrimeFieldBits,
{
  fn public_values(&self) -> Result<Vec<E::Scalar>, SynthesisError> {
    Ok(bytes_to_public_scalars(&self.expected_output()))
  }

  fn shared<CS: ConstraintSystem<E::Scalar>>(
    &self,
    _: &mut CS,
  ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
    Ok(vec![])
  }

  fn precommitted<CS: ConstraintSystem<E::Scalar>>(
    &self,
    cs: &mut CS,
    _: &[AllocatedNum<E::Scalar>],
  ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
    // Allocate input bits (512 bits = 64 bytes)
    let mut current_bits = alloc_input_bits(cs, &self.input, "input")?;

    // Chain Keccak-256 hashes
    for chain_idx in 0..self.chain_length {
      let hash_bits = keccak256(
        cs.namespace(|| format!("keccak_{}", chain_idx)),
        &current_bits,
      )?;
      // Pad output (256 bits) to input size (512 bits)
      current_bits = pad_256_to_512(&hash_bits);
    }

    // Expose final hash as public (first 256 bits)
    expose_hash_bits_as_public::<E, _>(cs, &current_bits[..256])?;

    Ok(vec![])
  }

  fn num_challenges(&self) -> usize {
    0
  }

  fn synthesize<CS: ConstraintSystem<E::Scalar>>(
    &self,
    _: &mut CS,
    _: &[AllocatedNum<E::Scalar>],
    _: &[AllocatedNum<E::Scalar>],
    _: Option<&[E::Scalar]>,
  ) -> Result<(), SynthesisError> {
    Ok(())
  }
}

impl<Scalar: PrimeField + PrimeFieldBits> Circuit<Scalar> for KeccakChainCircuit<Scalar> {
  fn synthesize<CS: ConstraintSystem<Scalar>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
    let mut current_bits = alloc_input_bits(cs, &self.input, "input")?;

    for chain_idx in 0..self.chain_length {
      let hash_bits = keccak256(
        cs.namespace(|| format!("keccak_{}", chain_idx)),
        &current_bits,
      )?;
      current_bits = pad_256_to_512(&hash_bits);
    }

    Ok(())
  }
}
