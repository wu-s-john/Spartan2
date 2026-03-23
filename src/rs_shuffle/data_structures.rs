//! Data structures for the RS shuffle algorithm
//!
//! This module defines both native and circuit variable versions of
//! the shuffle witness data and ElGamal ciphertexts.

use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use ark_r1cs_std::{
  alloc::AllocVar, boolean::Boolean, fields::fp::FpVar, groups::CurveVar, prelude::*,
};
use ark_relations::r1cs::{Namespace, SynthesisError};
use std::borrow::Borrow;

// ============================================================================
// Native Data Structures
// ============================================================================

/// Row in the unsorted (witness) table for one level
#[derive(Clone, Copy, Debug)]
pub struct UnsortedRow {
  /// Split bit (0 or 1)
  pub bit: bool,
  /// Number of zeros seen before this row in its bucket
  pub num_zeros: u16,
  /// Number of ones seen before this row in its bucket
  pub num_ones: u16,
  /// Total zeros in this bucket (constant for all rows of bucket)
  pub num_zeros_in_bucket: u16,
  /// Length of this bucket (constant for all rows of bucket)
  pub bucket_length: u16,
  /// Stable original index
  pub idx: u16,
  /// Computed destination position for next level
  pub next_pos: u16,
  /// Bucket ID this row belongs to
  pub bucket_id: u16,
}

impl UnsortedRow {
  pub fn new(
    bit: bool,
    num_zeros: u16,
    num_ones: u16,
    num_zeros_in_bucket: u16,
    bucket_length: u16,
    idx: u16,
    next_pos: u16,
    bucket_id: u16,
  ) -> Self {
    Self {
      bit,
      num_zeros,
      num_ones,
      num_zeros_in_bucket,
      bucket_length,
      idx,
      next_pos,
      bucket_id,
    }
  }
}

/// Row in the next (sorted) array after placement
#[derive(Clone, Copy, Debug)]
pub struct SortedRow {
  /// Stable original index
  pub idx: u16,
  /// Length of the bucket this row enters (for next level)
  pub length: u16,
  /// Bucket index this row belongs to
  pub bucket: u16,
}

impl SortedRow {
  pub fn new_with_bucket(idx: u16, length: u16, bucket: u16) -> Self {
    Self {
      idx,
      length,
      bucket,
    }
  }
}

/// Witness data for all levels of the shuffle
#[derive(Clone, Debug)]
pub struct PermutationWitnessTrace<const N: usize, const LEVELS: usize> {
  /// Split bits matrix (LEVELS × N)
  pub bits_mat: [[bool; N]; LEVELS],
  /// Unsorted witness rows per level
  pub uns_levels: [[UnsortedRow; N]; LEVELS],
  /// Next-array per level (sorted rows after placement)
  pub next_levels: [[SortedRow; N]; LEVELS],
}

// ============================================================================
// Circuit Variable Versions
// ============================================================================

/// Circuit variable version of UnsortedRow for use in SNARK constraints
#[derive(Clone)]
pub struct UnsortedRowVar<F: PrimeField> {
  /// Split bit (0 or 1)
  pub bit: Boolean<F>,
  /// Number of zeros seen before this row in its bucket
  pub num_zeros: FpVar<F>,
  /// Number of ones seen before this row in its bucket
  pub num_ones: FpVar<F>,
  /// Total zeros in this bucket (constant for all rows of bucket)
  pub total_zeros_in_bucket: FpVar<F>,
  /// Length of this bucket (constant for all rows of bucket)
  pub bucket_length: FpVar<F>,
  /// Stable original index
  pub idx: FpVar<F>,
  /// Computed destination position for next level
  pub next_pos: FpVar<F>,
  /// Bucket ID this row belongs to
  pub bucket_id: FpVar<F>,
}

impl<F: PrimeField> AllocVar<UnsortedRow, F> for UnsortedRowVar<F> {
  fn new_variable<T: Borrow<UnsortedRow>>(
    cs: impl Into<Namespace<F>>,
    f: impl FnOnce() -> Result<T, SynthesisError>,
    mode: AllocationMode,
  ) -> Result<Self, SynthesisError> {
    let cs = cs.into().cs();
    let value = f()?;
    let row = value.borrow();

    Ok(Self {
      bit: Boolean::new_variable(cs.clone(), || Ok(row.bit), mode)?,
      num_zeros: FpVar::new_variable(cs.clone(), || Ok(F::from(row.num_zeros as u64)), mode)?,
      num_ones: FpVar::new_variable(cs.clone(), || Ok(F::from(row.num_ones as u64)), mode)?,
      total_zeros_in_bucket: FpVar::new_variable(
        cs.clone(),
        || Ok(F::from(row.num_zeros_in_bucket as u64)),
        mode,
      )?,
      bucket_length: FpVar::new_variable(
        cs.clone(),
        || Ok(F::from(row.bucket_length as u64)),
        mode,
      )?,
      idx: FpVar::new_variable(cs.clone(), || Ok(F::from(row.idx as u64)), mode)?,
      next_pos: FpVar::new_variable(cs.clone(), || Ok(F::from(row.next_pos as u64)), mode)?,
      bucket_id: FpVar::new_variable(cs.clone(), || Ok(F::from(row.bucket_id as u64)), mode)?,
    })
  }
}

/// Circuit variable version of SortedRow for use in SNARK constraints
#[derive(Clone)]
pub struct SortedRowVar<F: PrimeField> {
  /// Stable original index
  pub idx: FpVar<F>,
}

impl<F: PrimeField> AllocVar<SortedRow, F> for SortedRowVar<F> {
  fn new_variable<T: Borrow<SortedRow>>(
    cs: impl Into<Namespace<F>>,
    f: impl FnOnce() -> Result<T, SynthesisError>,
    mode: AllocationMode,
  ) -> Result<Self, SynthesisError> {
    let cs = cs.into().cs();
    let value = f()?;
    let row = value.borrow();

    Ok(Self {
      idx: FpVar::new_variable(cs.clone(), || Ok(F::from(row.idx as u64)), mode)?,
    })
  }
}

/// Circuit variable version of PermutationWitnessTrace
#[derive(Clone)]
pub struct PermutationWitnessTraceVar<F: PrimeField, const N: usize, const LEVELS: usize> {
  /// Split bits matrix (LEVELS × N) as Boolean variables
  pub bits_mat: [[Boolean<F>; N]; LEVELS],
  /// Unsorted witness rows per level
  pub uns_levels: [[UnsortedRowVar<F>; N]; LEVELS],
  /// Next-array per level
  pub sorted_levels: [[SortedRowVar<F>; N]; LEVELS],
}

impl<F: PrimeField, const N: usize, const LEVELS: usize>
  AllocVar<PermutationWitnessTrace<N, LEVELS>, F> for PermutationWitnessTraceVar<F, N, LEVELS>
{
  fn new_variable<T: Borrow<PermutationWitnessTrace<N, LEVELS>>>(
    cs: impl Into<Namespace<F>>,
    f: impl FnOnce() -> Result<T, SynthesisError>,
    mode: AllocationMode,
  ) -> Result<Self, SynthesisError> {
    let cs = cs.into().cs();
    let value = f()?;
    let witness_data = value.borrow();

    // Allocate bits matrix
    let bits_mat: [[Boolean<F>; N]; LEVELS] = std::array::from_fn(|level| {
      std::array::from_fn(|i| {
        Boolean::new_variable(cs.clone(), || Ok(witness_data.bits_mat[level][i]), mode)
          .expect("Failed to allocate bit")
      })
    });

    // Allocate unsorted rows
    let uns_levels: [[UnsortedRowVar<F>; N]; LEVELS] = std::array::from_fn(|level| {
      std::array::from_fn(|i| {
        UnsortedRowVar::new_variable(cs.clone(), || Ok(&witness_data.uns_levels[level][i]), mode)
          .expect("Failed to allocate unsorted row")
      })
    });

    // Allocate sorted rows
    let sorted_levels: [[SortedRowVar<F>; N]; LEVELS] = std::array::from_fn(|level| {
      std::array::from_fn(|i| {
        SortedRowVar::new_variable(cs.clone(), || Ok(&witness_data.next_levels[level][i]), mode)
          .expect("Failed to allocate sorted row")
      })
    });

    Ok(Self {
      bits_mat,
      uns_levels,
      sorted_levels,
    })
  }
}

// ============================================================================
// ElGamal Ciphertext Types
// ============================================================================

/// Native ElGamal ciphertext
#[derive(Clone, Debug)]
pub struct ElGamalCiphertext<C: CurveGroup> {
  /// First component: r·G
  pub c1: C,
  /// Second component: M + r·PK
  pub c2: C,
}

impl<C: CurveGroup> ElGamalCiphertext<C> {
  pub fn new(c1: C, c2: C) -> Self {
    Self { c1, c2 }
  }

  /// Encrypt a message (curve point) using ElGamal encryption
  /// Returns ElGamalCiphertext(r*G, M + r*PK)
  pub fn encrypt(message: C, randomness: C::ScalarField, public_key: C) -> Self {
    let identity = C::zero();
    let initial = Self::new(identity, message);
    initial.add_encryption_layer(randomness, &public_key)
  }

  /// Encrypt a scalar message by first converting it to a curve point (scalar * G)
  pub fn encrypt_scalar(
    message: C::ScalarField,
    randomness: C::ScalarField,
    public_key: C,
  ) -> Self {
    let generator = C::generator();
    let message_point = generator * message;
    Self::encrypt(message_point, randomness, public_key)
  }

  /// Add an encryption layer (for re-encryption)
  /// (c1, c2) → (c1 + r·G, c2 + r·PK)
  pub fn add_encryption_layer(&self, randomness: C::ScalarField, public_key: &C) -> Self {
    let generator = C::generator();
    Self {
      c1: self.c1 + generator * randomness,
      c2: self.c2 + (*public_key) * randomness,
    }
  }
}

/// Circuit variable for ElGamal ciphertext
#[derive(Clone, Debug)]
pub struct ElGamalCiphertextVar<C, CV>
where
  C: CurveGroup,
  C::BaseField: PrimeField,
  CV: CurveVar<C, C::BaseField>,
{
  pub c1: CV,
  pub c2: CV,
  _phantom: std::marker::PhantomData<C>,
}

impl<C, CV> ElGamalCiphertextVar<C, CV>
where
  C: CurveGroup,
  C::BaseField: PrimeField,
  CV: CurveVar<C, C::BaseField>,
{
  pub fn new(c1: CV, c2: CV) -> Self {
    Self {
      c1,
      c2,
      _phantom: std::marker::PhantomData,
    }
  }
}

impl<C, CV> AllocVar<ElGamalCiphertext<C>, C::BaseField> for ElGamalCiphertextVar<C, CV>
where
  C: CurveGroup,
  C::BaseField: PrimeField,
  CV: CurveVar<C, C::BaseField>,
{
  fn new_variable<T: Borrow<ElGamalCiphertext<C>>>(
    cs: impl Into<Namespace<C::BaseField>>,
    f: impl FnOnce() -> Result<T, SynthesisError>,
    mode: AllocationMode,
  ) -> Result<Self, SynthesisError> {
    let cs = cs.into().cs();
    let value = f()?;
    let ct = value.borrow();

    let c1 = CV::new_variable(cs.clone(), || Ok(ct.c1), mode)?;
    let c2 = CV::new_variable(cs.clone(), || Ok(ct.c2), mode)?;

    Ok(Self::new(c1, c2))
  }
}
