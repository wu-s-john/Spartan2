// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT

//! Keccak-256 chain circuit for benchmarking small-value sumcheck.
//!
//! Uses bellpepper-keccak which already has small coefficients (0, 1, -1, 2).

use bellpepper_core::{
  Circuit, ConstraintSystem, SynthesisError,
  num::AllocatedNum,
};
use ff::{Field, PrimeField, PrimeFieldBits};
use sha3::{Digest, Keccak256};
use std::marker::PhantomData;

use crate::traits::{Engine, circuit::{SmallSpartanCircuit, SpartanCircuit}};
use crate::gadgets::{SmallBoolean, small_keccak256};
use crate::gadgets::small_boolean::{NegOne, SmallBit};
use crate::small_constraint_system::{SmallConstraintSystem, SmallToBellpepperCS};

/// Keccak-256 chain circuit using bellpepper-keccak.
///
/// Chains `chain_length` Keccak-256 hashes starting from an arbitrary-length input.
/// Hash[0] = Keccak256(input), Hash[i] = Keccak256(Hash[i-1])
#[derive(Debug, Clone)]
pub struct KeccakChainCircuit<Scalar: PrimeField> {
  /// Arbitrary-length byte input to start the chain
  pub input: Vec<u8>,
  /// Number of Keccak-256 hashes in the chain
  pub chain_length: usize,
  _p: PhantomData<Scalar>,
}

impl<Scalar: PrimeField + PrimeFieldBits> KeccakChainCircuit<Scalar> {
  /// Create a new Keccak-256 chain circuit.
  pub fn new(input: Vec<u8>, chain_length: usize) -> Self {
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
      let mut hasher = Keccak256::new();
      hasher.update(&current);
      let result = hasher.finalize();
      current = result.as_slice().try_into().unwrap();
    }
    current
  }
}

impl<E: Engine> SpartanCircuit<E> for KeccakChainCircuit<E::Scalar>
where
  E::Scalar: PrimeFieldBits,
{
  fn public_values(&self) -> Result<Vec<E::Scalar>, SynthesisError> {
    // LSB-first per byte, matching Keccak standard
    Ok(
      self
        .expected_output()
        .iter()
        .flat_map(|&byte| {
          (0..8).map(move |i| {
            if (byte >> i) & 1 == 1 {
              E::Scalar::ONE
            } else {
              E::Scalar::ZERO
            }
          })
        })
        .collect(),
    )
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
    // Use SmallToBellpepperCS so the field and integer paths produce the SAME shape.
    let mut small_cs = SmallToBellpepperCS::<E::Scalar, CS>::new(cs);
    let mut current_bits = alloc_input_small_bits::<i32, _>(&mut small_cs, &self.input, "input")?;

    for chain_idx in 0..self.chain_length {
      let mut ns =
        SmallConstraintSystem::<i32>::namespace(&mut small_cs, || format!("keccak_{chain_idx}"));
      let hash_bits = small_keccak256::<i32, _>(&mut ns, &current_bits)?;
      drop(ns);
      // After first hash, output is 256 bits (32 bytes); feed directly to next hash
      current_bits = hash_bits;
    }

    // Expose final hash as public inputs via the outer CS
    let outer_cs = small_cs.cs;
    for bit in &current_bits[..256] {
      outer_cs.alloc_input(
        || "hash_bit",
        || {
          Ok(if bit.get_value().unwrap_or(false) {
            E::Scalar::ONE
          } else {
            E::Scalar::ZERO
          })
        },
      )?;
    }

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
    let mut small_cs = SmallToBellpepperCS::<Scalar, CS>::new(cs);
    let mut current_bits = alloc_input_small_bits::<i32, _>(&mut small_cs, &self.input, "input")?;

    for chain_idx in 0..self.chain_length {
      let mut ns =
        SmallConstraintSystem::<i32>::namespace(&mut small_cs, || format!("keccak_{chain_idx}"));
      let hash_bits = small_keccak256::<i32, _>(&mut ns, &current_bits)?;
      drop(ns);
      current_bits = hash_bits;
    }

    Ok(())
  }
}

// ── SmallSpartanCircuit impls ─────────────────────────────────────────────

/// Allocate input bytes as SmallBoolean bits (LSB-first per byte, matching Keccak standard).
fn alloc_input_small_bits<V, CS>(
  cs: &mut CS,
  bytes: &[u8],
  prefix: &str,
) -> Result<Vec<SmallBoolean>, SynthesisError>
where
  V: Copy + From<bool> + NegOne,
  CS: SmallConstraintSystem<V>,
{
  let mut bits = Vec::with_capacity(bytes.len() * 8);
  for (byte_idx, &byte) in bytes.iter().enumerate() {
    for bit_idx in 0..8 {
      let val = (byte >> bit_idx) & 1 == 1;
      let bit = SmallBit::alloc(
        &mut cs.namespace(|| format!("{prefix}_byte{byte_idx}_bit{bit_idx}")),
        Some(val),
      )?;
      bits.push(SmallBoolean::Is(bit));
    }
  }
  Ok(bits)
}

/// Convert expected hash bytes to small public values (LSB-first per byte).
fn bytes_to_public_small<V: From<bool>>(bytes: &[u8]) -> Vec<V> {
  bytes
    .iter()
    .flat_map(|&byte| {
      (0..8).map(move |i| V::from((byte >> i) & 1 == 1))
    })
    .collect()
}

/// Keccak chain circuit: SmallSpartanCircuit<E, i8> — for shape extraction.
///
/// Uses `SmallShapeCS<i8>` to record i8-coefficient constraints.
impl<E: Engine> SmallSpartanCircuit<E, i8> for KeccakChainCircuit<E::Scalar>
where
  E::Scalar: PrimeFieldBits,
{
  fn public_values(&self) -> Result<Vec<i8>, SynthesisError> {
    Ok(bytes_to_public_small(&self.expected_output()))
  }

  fn shared<CS: SmallConstraintSystem<i8>>(
    &self,
    _cs: &mut CS,
  ) -> Result<Vec<bellpepper_core::Variable>, SynthesisError> {
    Ok(vec![])
  }

  fn precommitted<CS: SmallConstraintSystem<i8>>(
    &self,
    cs: &mut CS,
    _shared: &[bellpepper_core::Variable],
  ) -> Result<Vec<bellpepper_core::Variable>, SynthesisError> {
    let mut current_bits = alloc_input_small_bits(cs, &self.input, "input")?;

    for chain_idx in 0..self.chain_length {
      let mut ns = cs.namespace(|| format!("keccak_{chain_idx}"));
      let hash_bits = small_keccak256::<i8, _>(&mut ns, &current_bits)?;
      drop(ns);
      current_bits = hash_bits;
    }

    // Expose final hash as public inputs
    for bit in &current_bits[..256] {
      let val = bit.get_value().map(|b| if b { 1i8 } else { 0i8 });
      cs.alloc_input(|| "hash_bit", || val.ok_or(SynthesisError::AssignmentMissing))?;
    }

    Ok(vec![])
  }

  fn num_challenges(&self) -> usize { 0 }

  fn synthesize<CS: SmallConstraintSystem<i8>>(
    &self,
    _cs: &mut CS,
    _shared: &[bellpepper_core::Variable],
    _precommitted: &[bellpepper_core::Variable],
    _challenges: Option<&[E::Scalar]>,
  ) -> Result<(), SynthesisError> {
    Ok(())
  }
}

/// Keccak chain circuit: SmallSpartanCircuit<E, bool> — for witness generation.
///
/// Uses `SmallSatisfyingAssignment<bool>`. All enforce calls are no-ops.
impl<E: Engine> SmallSpartanCircuit<E, bool> for KeccakChainCircuit<E::Scalar>
where
  E::Scalar: PrimeFieldBits,
{
  fn public_values(&self) -> Result<Vec<bool>, SynthesisError> {
    Ok(
      self
        .expected_output()
        .iter()
        .flat_map(|&byte| (0..8).map(move |i| (byte >> i) & 1 == 1))
        .collect(),
    )
  }

  fn shared<CS: SmallConstraintSystem<bool>>(
    &self,
    _cs: &mut CS,
  ) -> Result<Vec<bellpepper_core::Variable>, SynthesisError> {
    Ok(vec![])
  }

  fn precommitted<CS: SmallConstraintSystem<bool>>(
    &self,
    cs: &mut CS,
    _shared: &[bellpepper_core::Variable],
  ) -> Result<Vec<bellpepper_core::Variable>, SynthesisError> {
    // Allocate input bits as bool witnesses
    let mut current_bits = alloc_input_small_bits(cs, &self.input, "input")?;

    for chain_idx in 0..self.chain_length {
      let mut ns = cs.namespace(|| format!("keccak_{chain_idx}"));
      let hash_bits = small_keccak256::<bool, _>(&mut ns, &current_bits)?;
      drop(ns);
      current_bits = hash_bits;
    }

    // Expose final hash as public inputs (bool)
    for bit in &current_bits[..256] {
      let val = bit.get_value();
      cs.alloc_input(|| "hash_bit", || val.ok_or(SynthesisError::AssignmentMissing))?;
    }

    Ok(vec![])
  }

  fn num_challenges(&self) -> usize { 0 }

  fn synthesize<CS: SmallConstraintSystem<bool>>(
    &self,
    _cs: &mut CS,
    _shared: &[bellpepper_core::Variable],
    _precommitted: &[bellpepper_core::Variable],
    _challenges: Option<&[E::Scalar]>,
  ) -> Result<(), SynthesisError> {
    Ok(())
  }
}
