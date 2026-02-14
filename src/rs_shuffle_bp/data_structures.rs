//! Data structures for the RS shuffle algorithm (Bellpepper version)
//!
//! This module defines both native and circuit variable versions of
//! the shuffle witness data and ElGamal ciphertexts.

use crate::{gadgets::ecc::AllocatedPoint, traits::Engine};
use bellpepper_core::{
  boolean::AllocatedBit, num::AllocatedNum, ConstraintSystem, SynthesisError,
};
use ff::PrimeField;

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
    Self { idx, length, bucket }
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
  pub bit: AllocatedBit,
  /// Number of zeros seen before this row in its bucket
  pub num_zeros: AllocatedNum<F>,
  /// Number of ones seen before this row in its bucket
  pub num_ones: AllocatedNum<F>,
  /// Total zeros in this bucket (constant for all rows of bucket)
  pub total_zeros_in_bucket: AllocatedNum<F>,
  /// Length of this bucket (constant for all rows of bucket)
  pub bucket_length: AllocatedNum<F>,
  /// Stable original index
  pub idx: AllocatedNum<F>,
  /// Computed destination position for next level
  pub next_pos: AllocatedNum<F>,
  /// Bucket ID this row belongs to
  pub bucket_id: AllocatedNum<F>,
}

impl<F: PrimeField> UnsortedRowVar<F> {
  /// Allocate a new UnsortedRowVar from a native UnsortedRow
  pub fn alloc<CS: ConstraintSystem<F>>(
    mut cs: CS,
    row: &UnsortedRow,
  ) -> Result<Self, SynthesisError> {
    let bit = AllocatedBit::alloc(cs.namespace(|| "bit"), Some(row.bit))?;
    let num_zeros = AllocatedNum::alloc(cs.namespace(|| "num_zeros"), || {
      Ok(F::from(row.num_zeros as u64))
    })?;
    let num_ones = AllocatedNum::alloc(cs.namespace(|| "num_ones"), || {
      Ok(F::from(row.num_ones as u64))
    })?;
    let total_zeros_in_bucket =
      AllocatedNum::alloc(cs.namespace(|| "total_zeros_in_bucket"), || {
        Ok(F::from(row.num_zeros_in_bucket as u64))
      })?;
    let bucket_length = AllocatedNum::alloc(cs.namespace(|| "bucket_length"), || {
      Ok(F::from(row.bucket_length as u64))
    })?;
    let idx = AllocatedNum::alloc(cs.namespace(|| "idx"), || Ok(F::from(row.idx as u64)))?;
    let next_pos = AllocatedNum::alloc(cs.namespace(|| "next_pos"), || {
      Ok(F::from(row.next_pos as u64))
    })?;
    let bucket_id = AllocatedNum::alloc(cs.namespace(|| "bucket_id"), || {
      Ok(F::from(row.bucket_id as u64))
    })?;

    Ok(Self {
      bit,
      num_zeros,
      num_ones,
      total_zeros_in_bucket,
      bucket_length,
      idx,
      next_pos,
      bucket_id,
    })
  }
}

/// Circuit variable version of SortedRow for use in SNARK constraints
#[derive(Clone)]
pub struct SortedRowVar<F: PrimeField> {
  /// Stable original index
  pub idx: AllocatedNum<F>,
}

impl<F: PrimeField> SortedRowVar<F> {
  /// Allocate a new SortedRowVar from a native SortedRow
  pub fn alloc<CS: ConstraintSystem<F>>(
    mut cs: CS,
    row: &SortedRow,
  ) -> Result<Self, SynthesisError> {
    let idx = AllocatedNum::alloc(cs.namespace(|| "idx"), || Ok(F::from(row.idx as u64)))?;
    Ok(Self { idx })
  }
}

/// Circuit variable version of PermutationWitnessTrace
#[derive(Clone)]
pub struct PermutationWitnessTraceVar<F: PrimeField, const N: usize, const LEVELS: usize> {
  /// Split bits matrix (LEVELS × N) as AllocatedBit variables
  pub bits_mat: [[AllocatedBit; N]; LEVELS],
  /// Unsorted witness rows per level
  pub uns_levels: [[UnsortedRowVar<F>; N]; LEVELS],
  /// Next-array per level
  pub sorted_levels: [[SortedRowVar<F>; N]; LEVELS],
}

impl<F: PrimeField, const N: usize, const LEVELS: usize> PermutationWitnessTraceVar<F, N, LEVELS> {
  /// Allocate a new PermutationWitnessTraceVar from native data
  pub fn alloc<CS: ConstraintSystem<F>>(
    mut cs: CS,
    witness_data: &PermutationWitnessTrace<N, LEVELS>,
  ) -> Result<Self, SynthesisError> {
    // Allocate bits matrix
    let mut bits_mat_vec: Vec<[AllocatedBit; N]> = Vec::with_capacity(LEVELS);
    for level in 0..LEVELS {
      let mut level_bits: Vec<AllocatedBit> = Vec::with_capacity(N);
      for i in 0..N {
        let bit = AllocatedBit::alloc(
          cs.namespace(|| format!("bit_{}_{}", level, i)),
          Some(witness_data.bits_mat[level][i]),
        )?;
        level_bits.push(bit);
      }
      let level_arr: [AllocatedBit; N] = level_bits
        .try_into()
        .map_err(|_| SynthesisError::Unsatisfiable)?;
      bits_mat_vec.push(level_arr);
    }
    let bits_mat: [[AllocatedBit; N]; LEVELS] = bits_mat_vec
      .try_into()
      .map_err(|_| SynthesisError::Unsatisfiable)?;

    // Allocate unsorted rows
    let mut uns_levels_vec: Vec<[UnsortedRowVar<F>; N]> = Vec::with_capacity(LEVELS);
    for level in 0..LEVELS {
      let mut level_rows: Vec<UnsortedRowVar<F>> = Vec::with_capacity(N);
      for i in 0..N {
        let row = UnsortedRowVar::alloc(
          cs.namespace(|| format!("uns_{}_{}", level, i)),
          &witness_data.uns_levels[level][i],
        )?;
        level_rows.push(row);
      }
      let level_arr: [UnsortedRowVar<F>; N] = level_rows
        .try_into()
        .map_err(|_| SynthesisError::Unsatisfiable)?;
      uns_levels_vec.push(level_arr);
    }
    let uns_levels: [[UnsortedRowVar<F>; N]; LEVELS] = uns_levels_vec
      .try_into()
      .map_err(|_| SynthesisError::Unsatisfiable)?;

    // Allocate sorted rows
    let mut sorted_levels_vec: Vec<[SortedRowVar<F>; N]> = Vec::with_capacity(LEVELS);
    for level in 0..LEVELS {
      let mut level_rows: Vec<SortedRowVar<F>> = Vec::with_capacity(N);
      for i in 0..N {
        let row = SortedRowVar::alloc(
          cs.namespace(|| format!("sorted_{}_{}", level, i)),
          &witness_data.next_levels[level][i],
        )?;
        level_rows.push(row);
      }
      let level_arr: [SortedRowVar<F>; N] = level_rows
        .try_into()
        .map_err(|_| SynthesisError::Unsatisfiable)?;
      sorted_levels_vec.push(level_arr);
    }
    let sorted_levels: [[SortedRowVar<F>; N]; LEVELS] = sorted_levels_vec
      .try_into()
      .map_err(|_| SynthesisError::Unsatisfiable)?;

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

/// Native ElGamal ciphertext using Engine types
#[derive(Clone, Debug)]
pub struct ElGamalCiphertext<E: Engine> {
  /// First component: r·G (x, y coordinates)
  pub c1_x: E::Base,
  pub c1_y: E::Base,
  /// Second component: M + r·PK (x, y coordinates)
  pub c2_x: E::Base,
  pub c2_y: E::Base,
}

impl<E: Engine> ElGamalCiphertext<E> {
  pub fn new(c1_x: E::Base, c1_y: E::Base, c2_x: E::Base, c2_y: E::Base) -> Self {
    Self {
      c1_x,
      c1_y,
      c2_x,
      c2_y,
    }
  }

  /// Create from field elements (used when converting from external representation)
  pub fn from_coords(c1_x: E::Base, c1_y: E::Base, c2_x: E::Base, c2_y: E::Base) -> Self {
    Self::new(c1_x, c1_y, c2_x, c2_y)
  }
}

/// Circuit variable for ElGamal ciphertext using AllocatedPoint
#[derive(Clone)]
pub struct ElGamalCiphertextVar<E: Engine> {
  pub c1: AllocatedPoint<E>,
  pub c2: AllocatedPoint<E>,
}

impl<E: Engine> ElGamalCiphertextVar<E> {
  pub fn new(c1: AllocatedPoint<E>, c2: AllocatedPoint<E>) -> Self {
    Self { c1, c2 }
  }

  /// Allocate a new ElGamalCiphertextVar from native ciphertext
  pub fn alloc<CS: ConstraintSystem<E::Base>>(
    mut cs: CS,
    ct: &ElGamalCiphertext<E>,
  ) -> Result<Self, SynthesisError> {
    let c1 = AllocatedPoint::alloc(
      cs.namespace(|| "c1"),
      Some((ct.c1_x, ct.c1_y, false)), // not infinity
    )?;
    let c2 = AllocatedPoint::alloc(
      cs.namespace(|| "c2"),
      Some((ct.c2_x, ct.c2_y, false)), // not infinity
    )?;
    Ok(Self::new(c1, c2))
  }

  /// Allocate as public input
  pub fn alloc_input<CS: ConstraintSystem<E::Base>>(
    mut cs: CS,
    ct: &ElGamalCiphertext<E>,
  ) -> Result<Self, SynthesisError> {
    // Allocate points and then inputize the coordinates
    let c1 = AllocatedPoint::alloc(cs.namespace(|| "c1"), Some((ct.c1_x, ct.c1_y, false)))?;
    let c2 = AllocatedPoint::alloc(cs.namespace(|| "c2"), Some((ct.c2_x, ct.c2_y, false)))?;

    // Inputize the coordinates
    c1.x.inputize(cs.namespace(|| "c1_x_input"))?;
    c1.y.inputize(cs.namespace(|| "c1_y_input"))?;
    c2.x.inputize(cs.namespace(|| "c2_x_input"))?;
    c2.y.inputize(cs.namespace(|| "c2_y_input"))?;

    Ok(Self::new(c1, c2))
  }
}
