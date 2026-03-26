// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! SHA-256 circuit using small_sha256 gadget (small-value compatible).

use super::hash_to_public_scalars;
use bellpepper_core::{Circuit, ConstraintSystem, SynthesisError, num::AllocatedNum};
use ff::{PrimeField, PrimeFieldBits};
use std::marker::PhantomData;

use crate::{
  gadgets::{NoBatchEq, SmallBoolean, small_sha256_int},
  small_constraint_system::{SmallConstraintSystem, SmallToBellpepperCS},
  traits::{
    Engine,
    circuit::{SmallSpartanCircuit, SpartanCircuit},
  },
};
use ff::Field;
#[cfg(debug_assertions)]
use sha2::{Digest, Sha256};

/// SHA-256 circuit using small_sha256 gadget (small-value compatible).
///
/// Uses `SmallMultiEq` to keep coefficients bounded for small-value sumcheck.
#[derive(Debug, Clone)]
pub struct SmallSha256Circuit<Scalar: PrimeField> {
  /// The preimage bytes to hash.
  pub preimage: Vec<u8>,
  /// If true, use BatchingEq<21> (i64 path); if false, use NoBatchEq (i32 path)
  pub use_batching: bool,
  _p: PhantomData<Scalar>,
}

impl<Scalar: PrimeField + PrimeFieldBits> SmallSha256Circuit<Scalar> {
  /// Create a new SHA-256 circuit.
  pub fn new(preimage: Vec<u8>, use_batching: bool) -> Self {
    Self {
      preimage,
      use_batching,
      _p: PhantomData,
    }
  }
}

impl<E: Engine> SpartanCircuit<E> for SmallSha256Circuit<E::Scalar>
where
  E::Scalar: PrimeFieldBits,
{
  fn public_values(&self) -> Result<Vec<E::Scalar>, SynthesisError> {
    Ok(hash_to_public_scalars(&self.preimage))
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
    let preimage_bits = alloc_preimage_small_bits::<i32, _>(&mut small_cs, &self.preimage)?;
    let mut eq = NoBatchEq::<i32, _>::new(&mut small_cs);
    let hash_bits = small_sha256_int::<i32, _>(&mut eq, &preimage_bits)?;
    drop(eq);

    // Verify against native SHA-256 (debug only)
    #[cfg(debug_assertions)]
    {
      let hash_expected = Sha256::digest(&self.preimage);
      for (i, bit) in hash_bits.iter().enumerate() {
        let byte_idx = i / 8;
        let bit_idx = 7 - (i % 8);
        let expected = (hash_expected[byte_idx] >> bit_idx) & 1 == 1;
        let computed = bit.get_value().unwrap_or(false);
        assert_eq!(computed, expected, "Hash bit {i} mismatch");
      }
    }

    // Expose hash bits as public inputs via the outer CS
    let outer_cs = small_cs.cs;
    for bit in &hash_bits {
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

// ── SmallSpartanCircuit impls ─────────────────────────────────────────────

/// Helper: allocate preimage bits as SmallBoolean variables.
fn alloc_preimage_small_bits<V, CS>(
  cs: &mut CS,
  preimage: &[u8],
) -> Result<Vec<SmallBoolean>, SynthesisError>
where
  V: Copy + From<bool> + crate::gadgets::small_boolean::NegOne,
  CS: SmallConstraintSystem<V>,
{
  let mut bits = Vec::with_capacity(preimage.len() * 8);
  for (byte_idx, &byte) in preimage.iter().enumerate() {
    for bit_idx in (0..8).rev() {
      let val = (byte >> bit_idx) & 1 == 1;
      let bit = crate::gadgets::small_boolean::SmallBit::alloc(
        &mut cs.namespace(|| format!("preimage_byte{byte_idx}_bit{bit_idx}")),
        Some(val),
      )?;
      bits.push(SmallBoolean::Is(bit));
    }
  }
  Ok(bits)
}

/// SHA-256 circuit: SmallSpartanCircuit<E, i32> — for shape extraction.
///
/// Uses `SmallShapeCS` (i32 coefficients). The whole circuit is in `precommitted`
/// since SHA-256 has no shared variables.
impl<E: Engine> SmallSpartanCircuit<E, i32> for SmallSha256Circuit<E::Scalar>
where
  E::Scalar: PrimeFieldBits,
{
  fn public_values(&self) -> Result<Vec<i32>, SynthesisError> {
    use crate::sha256_circuits::hash_to_public_scalars;
    let bits: Vec<E::Scalar> = hash_to_public_scalars(&self.preimage);
    Ok(
      bits
        .iter()
        .map(|b| if b.is_zero().into() { 0i32 } else { 1i32 })
        .collect(),
    )
  }

  fn shared<CS: SmallConstraintSystem<i32>>(
    &self,
    _cs: &mut CS,
  ) -> Result<Vec<bellpepper_core::Variable>, SynthesisError> {
    Ok(vec![])
  }

  fn precommitted<CS: SmallConstraintSystem<i32>>(
    &self,
    cs: &mut CS,
    _shared: &[bellpepper_core::Variable],
  ) -> Result<Vec<bellpepper_core::Variable>, SynthesisError> {
    let preimage_bits = alloc_preimage_small_bits(cs, &self.preimage)?;
    let mut eq = NoBatchEq::<i32, _>::new(cs);
    let hash_bits = small_sha256_int::<i32, _>(&mut eq, &preimage_bits)?;
    drop(eq);

    // Inputize hash bits as public values
    for bit in &hash_bits {
      let val = bit.get_value().map(|b| if b { 1i32 } else { 0i32 });
      cs.alloc_input(
        || "hash_bit",
        || val.ok_or(SynthesisError::AssignmentMissing),
      )?;
    }

    Ok(vec![])
  }

  fn num_challenges(&self) -> usize {
    0
  }

  fn synthesize<CS: SmallConstraintSystem<i32>>(
    &self,
    _cs: &mut CS,
    _shared: &[bellpepper_core::Variable],
    _precommitted: &[bellpepper_core::Variable],
    _challenges: Option<&[E::Scalar]>,
  ) -> Result<(), SynthesisError> {
    Ok(())
  }
}

/// SHA-256 circuit: SmallSpartanCircuit<E, i8> — for witness generation.
///
/// Uses `SmallSatisfyingAssignment<i8>`. All enforce calls are no-ops.
impl<E: Engine> SmallSpartanCircuit<E, i8> for SmallSha256Circuit<E::Scalar>
where
  E::Scalar: PrimeFieldBits,
{
  fn public_values(&self) -> Result<Vec<i8>, SynthesisError> {
    use crate::sha256_circuits::hash_to_public_scalars;
    let bits: Vec<E::Scalar> = hash_to_public_scalars(&self.preimage);
    Ok(
      bits
        .iter()
        .map(|b| if b.is_zero().into() { 0i8 } else { 1i8 })
        .collect(),
    )
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
    // Run SHA-256 gadget with i8 value type — allocates same vars as i32 shape path
    // but enforce is a no-op, so only witness values are recorded.
    let preimage_bits = alloc_preimage_small_bits(cs, &self.preimage)?;
    let mut eq = NoBatchEq::<i8, _>::new(cs);
    let hash_bits = small_sha256_int::<i8, _>(&mut eq, &preimage_bits)?;
    drop(eq);

    // Inputize hash bits as public values (i8)
    for bit in &hash_bits {
      let val = bit.get_value().map(|b| if b { 1i8 } else { 0i8 });
      cs.alloc_input(
        || "hash_bit",
        || val.ok_or(SynthesisError::AssignmentMissing),
      )?;
    }

    Ok(vec![])
  }

  fn num_challenges(&self) -> usize {
    0
  }

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

impl<Scalar: PrimeField + PrimeFieldBits> Circuit<Scalar> for SmallSha256Circuit<Scalar> {
  fn synthesize<CS: ConstraintSystem<Scalar>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
    let mut small_cs = SmallToBellpepperCS::<Scalar, CS>::new(cs);
    let preimage_bits = alloc_preimage_small_bits::<i32, _>(&mut small_cs, &self.preimage)?;
    let mut eq = NoBatchEq::<i32, _>::new(&mut small_cs);
    let _ = small_sha256_int::<i32, _>(&mut eq, &preimage_bits)?;
    Ok(())
  }
}
