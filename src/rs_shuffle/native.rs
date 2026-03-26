//! Native RS shuffle execution (prover-side logic)
//!
//! This module provides the native (non-circuit) implementation of the RS shuffle,
//! used to generate witnesses for the SNARK prover.

use super::{
  bit_generation::derive_split_bits,
  data_structures::{PermutationWitnessTrace, SortedRow, UnsortedRow},
};
use ark_crypto_primitives::sponge::Absorb;
use ark_ff::PrimeField;
use std::collections::HashMap;

/// Output of the RS shuffle permutation operation
#[derive(Clone, Debug)]
pub struct RSShuffleTrace<T, const N: usize, const LEVELS: usize> {
  /// The witness trace for the shuffle
  pub witness_trace: PermutationWitnessTrace<N, LEVELS>,
  /// The number of samples used in bit generation
  pub num_samples: usize,
  /// The permuted array
  pub permuted_output: [T; N],
}

impl<T, const N: usize, const LEVELS: usize> RSShuffleTrace<T, N, LEVELS> {
  /// Extract the permutation array (0-indexed) from the witness trace
  pub fn extract_permutation_array(&self) -> [usize; N] {
    let final_sorted = &self.witness_trace.next_levels[LEVELS - 1];
    std::array::from_fn(|i| final_sorted[i].idx as usize)
  }
}

/// Run RS shuffle permutation on a collection
///
/// This function generates the witness trace for RS shuffle and applies the resulting
/// permutation to the input collection.
///
/// # Parameters
/// - `seed`: The seed for deterministic permutation generation
/// - `input`: The input array to be permuted
///
/// # Returns
/// An `RSShuffleTrace` containing the witness trace and permuted output
pub fn run_rs_shuffle_permutation<F, T, const N: usize, const LEVELS: usize>(
  seed: F,
  input: &[T; N],
) -> RSShuffleTrace<T, N, LEVELS>
where
  F: PrimeField + Absorb,
  T: Clone + std::fmt::Debug,
{
  // Generate witness trace
  let (witness_trace, num_samples) = prepare_rs_witness_trace::<F, N, LEVELS>(seed);

  // Extract final permutation from last level
  let final_sorted = &witness_trace.next_levels[LEVELS - 1];

  // Apply permutation to create output array
  let output: Vec<T> = final_sorted
    .iter()
    .map(|sorted_row| input[sorted_row.idx as usize].clone())
    .collect();

  let permuted_output: [T; N] = output
    .try_into()
    .expect("Permutation should preserve array size");

  RSShuffleTrace {
    witness_trace,
    num_samples,
    permuted_output,
  }
}

/// Prepare the witness trace for RS shuffle
///
/// # Returns
/// - The witness trace containing all level data
/// - The number of Poseidon samples used
pub fn prepare_rs_witness_trace<F, const N: usize, const LEVELS: usize>(
  seed: F,
) -> (PermutationWitnessTrace<N, LEVELS>, usize)
where
  F: PrimeField + Absorb,
{
  // Derive split bits from seed
  let (bits_mat, num_samples) = derive_split_bits::<F, N, LEVELS>(seed);

  // Initialize with level-0 rows (one bucket of full length)
  let prev: [SortedRow; N] =
    std::array::from_fn(|i| SortedRow::new_with_bucket(i as u16, N as u16, 0));

  // Process all levels
  let level_results: Vec<([UnsortedRow; N], [SortedRow; N])> = (0..LEVELS)
    .scan(prev, |prev_state, level| {
      let bits_level = &bits_mat[level];
      let (uns_array, nxt_array) = build_level::<N>(prev_state, bits_level);
      *prev_state = nxt_array.clone();
      Some((uns_array, nxt_array))
    })
    .collect();

  // Convert to arrays
  let uns_levels: [[UnsortedRow; N]; LEVELS] = std::array::from_fn(|i| level_results[i].0.clone());
  let next_levels: [[SortedRow; N]; LEVELS] = std::array::from_fn(|i| level_results[i].1.clone());

  (
    PermutationWitnessTrace {
      bits_mat,
      uns_levels,
      next_levels,
    },
    num_samples,
  )
}

/// Build witness tables for one level of the shuffle
///
/// This implements the stable-partition logic where elements are sorted
/// by their bit value while preserving relative order within each partition.
pub fn build_level<const N: usize>(
  prev_rows: &[SortedRow; N],
  bits_lvl: &[bool; N],
) -> ([UnsortedRow; N], [SortedRow; N]) {
  // Zip prev_rows with bits
  let rows_with_bits: Vec<(&SortedRow, bool)> =
    prev_rows.iter().zip(bits_lvl.iter().copied()).collect();

  // Track bucket statistics
  let mut bucket_zeros: HashMap<u16, u16> = HashMap::new();
  let mut bucket_ones: HashMap<u16, u16> = HashMap::new();
  let mut bucket_starts: HashMap<u16, u16> = HashMap::new();
  let mut bucket_lengths: HashMap<u16, u16> = HashMap::new();

  // First pass: compute bucket statistics
  let mut current_pos = 0u16;
  for (row, bit) in &rows_with_bits {
    let bucket = row.bucket;

    bucket_starts.entry(bucket).or_insert(current_pos);

    if !bit {
      *bucket_zeros.entry(bucket).or_insert(0) += 1;
    } else {
      *bucket_ones.entry(bucket).or_insert(0) += 1;
    }

    *bucket_lengths.entry(bucket).or_insert(0) += 1;
    current_pos += 1;
  }

  // Create unsorted rows with running counters
  let mut unsorted = Vec::new();
  let mut current_bucket = None;
  let mut num_zeros = 0u16;
  let mut num_ones = 0u16;

  for (row, bit) in rows_with_bits.iter() {
    let bucket = row.bucket;

    // Reset counters on bucket change
    if current_bucket != Some(bucket) {
      num_zeros = 0;
      num_ones = 0;
      current_bucket = Some(bucket);
    }

    let num_zeros_in_bucket = *bucket_zeros.get(&bucket).unwrap_or(&0);
    let bucket_length = *bucket_lengths.get(&bucket).unwrap_or(&0);
    let bucket_start = *bucket_starts.get(&bucket).unwrap_or(&0);

    // Compute destination position
    let offset = if !bit {
      num_zeros
    } else {
      num_zeros_in_bucket + num_ones
    };
    let next_pos = bucket_start + offset;

    unsorted.push(UnsortedRow::new(
      *bit,
      num_zeros,
      num_ones,
      num_zeros_in_bucket,
      bucket_length,
      row.idx,
      next_pos,
      bucket,
    ));

    // Update counters
    if !bit {
      num_zeros += 1;
    } else {
      num_ones += 1;
    }
  }

  // Create indexed tuples for sorting
  let mut sortable: Vec<(u16, u16, bool, u16)> = unsorted
    .iter()
    .zip(&rows_with_bits)
    .map(|(uns, (row, bit))| (uns.next_pos, row.idx, *bit, row.bucket))
    .collect();

  // Stable sort by next_pos
  sortable.sort_by_key(|&(next_pos, _, _, _)| next_pos);

  // Build the next array
  let next_arr: Vec<SortedRow> = sortable
    .into_iter()
    .map(|(_, idx, bit, parent_bucket)| {
      let num_zeros_in_bucket = *bucket_zeros.get(&parent_bucket).unwrap_or(&0);
      let num_ones_in_bucket = *bucket_ones.get(&parent_bucket).unwrap_or(&0);

      let next_bucket = if !bit {
        parent_bucket * 2
      } else {
        parent_bucket * 2 + 1
      };

      let next_bucket_length = if !bit {
        num_zeros_in_bucket
      } else {
        num_ones_in_bucket
      };

      SortedRow::new_with_bucket(idx, next_bucket_length, next_bucket)
    })
    .collect();

  // Convert to fixed-size arrays
  let unsorted_array: [UnsortedRow; N] = unsorted
    .try_into()
    .expect("Unsorted array should have exactly N elements");
  let next_array: [SortedRow; N] = next_arr
    .try_into()
    .expect("Next array should have exactly N elements");

  (unsorted_array, next_array)
}

#[cfg(test)]
mod tests {
  use super::*;
  use ark_bn254::Fr as TestField;

  const N: usize = 8;
  const LEVELS: usize = 3;

  #[test]
  fn test_run_rs_shuffle_permutation() {
    let seed = TestField::from(42u64);
    let input: [usize; N] = std::array::from_fn(|i| i);

    let trace = run_rs_shuffle_permutation::<TestField, _, N, LEVELS>(seed, &input);

    // Verify output is a permutation of input
    let mut sorted_output = trace.permuted_output.to_vec();
    sorted_output.sort();
    assert_eq!(sorted_output, vec![0, 1, 2, 3, 4, 5, 6, 7]);

    // Verify witness trace has correct dimensions
    assert_eq!(trace.witness_trace.bits_mat.len(), LEVELS);
    assert_eq!(trace.witness_trace.uns_levels.len(), LEVELS);
    assert_eq!(trace.witness_trace.next_levels.len(), LEVELS);
  }

  #[test]
  fn test_build_level_single_bucket() {
    // Single bucket with alternating bits
    let prev_rows: [SortedRow; N] =
      std::array::from_fn(|i| SortedRow::new_with_bucket(i as u16, N as u16, 0));

    let bits: [bool; N] = [false, true, false, true, false, true, false, true];
    let (_unsorted, next) = build_level::<N>(&prev_rows, &bits);

    // Verify zeros come before ones
    let zero_count = bits.iter().filter(|&&b| !b).count();
    for i in 0..zero_count {
      assert_eq!(next[i].bucket, 0); // Zeros go to bucket 0
    }
    for i in zero_count..N {
      assert_eq!(next[i].bucket, 1); // Ones go to bucket 1
    }

    // Verify all indices are accounted for
    let mut indices: Vec<u16> = next.iter().map(|r| r.idx).collect();
    indices.sort();
    assert_eq!(indices, vec![0, 1, 2, 3, 4, 5, 6, 7]);
  }
}
