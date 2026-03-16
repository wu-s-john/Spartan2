// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! # Sparse Matrices
//!
//! This module defines a custom implementation of CSR/CSC sparse matrices.
//! Specifically, we implement sparse matrix / dense vector multiplication
//! to compute the `A z`, `B z`, and `C z` in Spartan.
use crate::{
  errors::SpartanError,
  small_field::{DelayedReduction, ExtensionBound, SmallValueField, WideMul},
};
use ff::PrimeField;
use num_traits::{Bounded, One, Signed};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
  fmt::Debug,
  ops::{Div, Mul},
};

/// CSR format sparse matrix, We follow the names used by scipy.
/// Detailed explanation here: https://stackoverflow.com/questions/52299420/scipy-csr-matrix-understand-indptr
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparseMatrix<V> {
  /// all non-zero values in the matrix
  pub data: Vec<V>,
  /// column indices
  pub indices: Vec<usize>,
  /// row information
  pub indptr: Vec<usize>,
  /// number of columns
  pub cols: usize,
}

// ---- Generic methods (no bounds on V) ----

impl<V> SparseMatrix<V> {
  /// 0x0 empty matrix
  pub fn empty() -> Self {
    SparseMatrix {
      data: vec![],
      indices: vec![],
      indptr: vec![0],
      cols: 0,
    }
  }

  /// Retrieves the data for row slice [i..j] from `ptrs`.
  /// We assume that `ptrs` is indexed from `indptrs` and do not check if the
  /// returned slice is actually a valid row.
  pub fn get_row_unchecked(&self, ptrs: &[usize; 2]) -> impl Iterator<Item = (&V, &usize)> {
    self.data[ptrs[0]..ptrs[1]]
      .iter()
      .zip(&self.indices[ptrs[0]..ptrs[1]])
  }
}

// ---- Methods requiring V: Copy ----

impl<V: Copy> SparseMatrix<V> {
  /// returns a custom iterator
  pub fn iter(&self) -> Iter<'_, V> {
    let mut row = 0;
    while row + 1 < self.indptr.len() && self.indptr[row + 1] == 0 {
      row += 1;
    }
    let nnz = if self.indptr.is_empty() {
      0
    } else {
      self.indptr[self.indptr.len() - 1]
    };
    Iter {
      matrix: self,
      row,
      i: 0,
      nnz,
    }
  }
}

// ---- PrimeField-specific methods ----

impl<F: PrimeField> SparseMatrix<F> {
  /// Construct from the COO representation; Vec<usize(row), usize(col), F>.
  /// We assume that the rows are sorted during construction.
  #[cfg(test)]
  pub fn new(matrix: &[(usize, usize, F)], rows: usize, cols: usize) -> Self {
    let mut new_matrix = vec![vec![]; rows];
    for (row, col, val) in matrix {
      new_matrix[*row].push((*col, *val));
    }

    for row in new_matrix.iter() {
      assert!(row.windows(2).all(|w| w[0].0 < w[1].0));
    }

    let mut indptr = vec![0; rows + 1];
    for (i, col) in new_matrix.iter().enumerate() {
      indptr[i + 1] = indptr[i] + col.len();
    }

    let mut indices = vec![];
    let mut data = vec![];
    for col in new_matrix {
      let (idx, val): (Vec<_>, Vec<_>) = col.into_iter().unzip();
      indices.extend(idx);
      data.extend(val);
    }

    SparseMatrix {
      data,
      indices,
      indptr,
      cols,
    }
  }

  /// Multiply by a dense vector; uses rayon/gpu.
  ///
  /// # Errors
  /// Returns `SpartanError::InvalidInputLength` if the vector length doesn't match the matrix dimensions.
  pub fn multiply_vec(&self, vector: &[F]) -> Result<Vec<F>, SpartanError> {
    if self.cols != vector.len() {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "SparseMatrix multiply_vec: Expected {} elements in vector, got {}",
          self.cols,
          vector.len()
        ),
      });
    }

    Ok(self.multiply_vec_unchecked(vector))
  }

  /// Multiply by a dense vector; uses rayon/gpu.
  /// This does not check that the shape of the matrix/vector are compatible.
  pub fn multiply_vec_unchecked(&self, vector: &[F]) -> Vec<F> {
    self
      .indptr
      .par_windows(2)
      .map(|ptrs| {
        // par_windows(2) guarantees ptrs has exactly 2 elements
        let row_ptrs = [ptrs[0], ptrs[1]];
        self
          .get_row_unchecked(&row_ptrs)
          .map(|(val, col_idx)| *val * vector[*col_idx])
          .sum()
      })
      .collect()
  }

  /// Multiply by a dense small-value vector using delayed reduction.
  ///
  /// Uses `DelayedReduction::unreduced_multiply_accumulate` (field × small) with delayed reduction,
  /// then coerces the result back to small values with Lagrange extension bound check.
  ///
  /// # Type Parameters
  ///
  /// - `SmallValue`: The small value type (i32 or i64)
  /// - `D`: Polynomial degree for Lagrange extension (typically 2)
  ///
  /// # Arguments
  ///
  /// - `z`: Dense vector to multiply with
  /// - `lb`: Number of Lagrange extension rounds (determines growth factor of (D+1)^lb)
  ///
  /// # Errors
  /// Returns `SpartanError::InvalidInputLength` if vector length doesn't match.
  /// Returns `SpartanError::SmallValueOverflow` if any result exceeds the safe bound
  /// for Lagrange extension (see [`ExtensionBound`](crate::small_field::ExtensionBound)).
  pub fn multiply_vec_small<const D: usize, SmallValue>(
    &self,
    z: &[SmallValue],
    lb: usize,
  ) -> Result<Vec<SmallValue>, SpartanError>
  where
    F: DelayedReduction<SmallValue> + SmallValueField<SmallValue>,
    SmallValue: WideMul + Bounded + Copy + Send + Sync + Into<SmallValue::Product>,
    SmallValue::Product: Copy
      + Ord
      + Signed
      + Div<Output = SmallValue::Product>
      + Mul<Output = SmallValue::Product>
      + One
      + From<i32>,
  {
    if self.cols != z.len() {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "SparseMatrix multiply_vec_small: Expected {} elements, got {}",
          self.cols,
          z.len()
        ),
      });
    }

    // Compute bound once, use for all rows
    let bound = ExtensionBound::<SmallValue, D>::new(lb);

    self
      .indptr
      .par_windows(2)
      .enumerate()
      .map(|(row_idx, ptrs)| {
        let start = ptrs[0];
        let end = ptrs[1];

        // Accumulate using delayed reduction: acc += matrix_val × z[col]
        let mut acc = <F as DelayedReduction<SmallValue>>::Accumulator::default();
        for i in start..end {
          let matrix_val = &self.data[i];
          let col_idx = self.indices[i];
          <F as DelayedReduction<SmallValue>>::unreduced_multiply_accumulate(
            &mut acc,
            matrix_val,
            &z[col_idx],
          );
        }

        // Reduce to field element, then coerce to small value with extension bound check
        let field_result = <F as DelayedReduction<SmallValue>>::reduce(&acc);
        bound
          .try_to_small(&field_result)
          .ok_or_else(|| SpartanError::SmallValueOverflow {
            value: format!("{:?}", field_result),
            context: format!(
              "multiply_vec_small: row {} exceeds bound for D={}, lb={}",
              row_idx, D, lb
            ),
          })
      })
      .collect()
  }
}

// ---- Pure integer methods for small-coefficient matrices ----

impl<Coeff: Copy + Default + std::ops::AddAssign + Send + Sync> SparseMatrix<Coeff> {
  /// Pure integer matrix-vector multiply for the small-value path.
  ///
  /// Computes M × z where M has `Coeff` coefficients and z has small witness values.
  /// Since witnesses are bits (0/1), this is conditional addition: for each nonzero z[col],
  /// add data[i] to the accumulator. No multiply needed.
  ///
  /// For SHA-256 with NoBatchEq (max coeff ~2^18) and ~200 nonzeros per row,
  /// each row result is at most ~200 × 2^18 ≈ 2^26, well within i32 range.
  pub fn multiply_vec_witness<W>(&self, z: &[W]) -> Result<Vec<Coeff>, SpartanError>
  where
    W: Copy + Default + PartialEq + Send + Sync,
  {
    if self.cols != z.len() {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "SparseMatrix::multiply_vec_witness: Expected {} elements, got {}",
          self.cols,
          z.len()
        ),
      });
    }

    let zero_w = W::default();
    Ok(
      self
        .indptr
        .par_windows(2)
        .map(|ptrs| {
          let mut acc = Coeff::default();
          for i in ptrs[0]..ptrs[1] {
            if z[self.indices[i]] != zero_w {
              acc += self.data[i];
            }
          }
          acc
        })
        .collect(),
    )
  }
}

// ---- ±1 partitioning for i32 matrices ----

impl SparseMatrix<i32> {
  /// Reorder entries within each row so ±1 entries come first.
  /// Returns a `Vec<usize>` of per-row split points: entries in
  /// `[indptr[row]..unit_end[row])` are ±1, the rest are non-±1.
  pub fn partition_unit_entries(&mut self) -> Vec<usize> {
    let num_rows = self.indptr.len() - 1;
    let mut unit_end = Vec::with_capacity(num_rows);
    for row in 0..num_rows {
      let start = self.indptr[row];
      let end = self.indptr[row + 1];
      // Partition: ±1 entries first, others after
      let mut write = start;
      for read in start..end {
        if self.data[read] == 1 || self.data[read] == -1 {
          self.data.swap(write, read);
          self.indices.swap(write, read);
          write += 1;
        }
      }
      unit_end.push(write);
    }
    unit_end
  }
}

/// Iterator for sparse matrix
pub struct Iter<'a, V> {
  matrix: &'a SparseMatrix<V>,
  row: usize,
  i: usize,
  nnz: usize,
}

impl<V: Copy> Iterator for Iter<'_, V> {
  type Item = (usize, usize, V);

  fn next(&mut self) -> Option<Self::Item> {
    // are we at the end?
    if self.i == self.nnz {
      return None;
    }

    // compute current item
    let curr_item = (
      self.row,
      self.matrix.indices[self.i],
      self.matrix.data[self.i],
    );

    // advance the iterator
    self.i += 1;
    // edge case at the end
    if self.i == self.nnz {
      return Some(curr_item);
    }
    // if `i` has moved to next row
    while self.i >= self.matrix.indptr[self.row + 1] {
      self.row += 1;
    }

    Some(curr_item)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    provider::PallasHyraxEngine,
    traits::{Engine, Group},
  };
  use ff::PrimeField;
  use proptest::{
    prelude::*,
    strategy::{BoxedStrategy, Just, Strategy},
  };

  type G = <PallasHyraxEngine as Engine>::GE;
  type Fr = <G as Group>::Scalar;

  /// Wrapper struct around a field element that implements additional traits
  #[derive(Clone, Debug, PartialEq, Eq)]
  pub struct FWrap<F: PrimeField>(pub F);

  impl<F: PrimeField> Copy for FWrap<F> {}

  #[cfg(not(target_arch = "wasm32"))]
  /// Trait implementation for generating `FWrap<F>` instances with proptest
  impl<F: PrimeField> Arbitrary for FWrap<F> {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
      use rand::rngs::StdRng;
      use rand_core::SeedableRng;

      let strategy = any::<[u8; 32]>()
        .prop_map(|seed| FWrap(F::random(StdRng::from_seed(seed))))
        .no_shrink();
      strategy.boxed()
    }
  }

  #[test]
  fn test_matrix_creation() {
    let matrix_data = vec![
      (0, 1, Fr::from(2)),
      (1, 2, Fr::from(3)),
      (2, 0, Fr::from(4)),
    ];
    let sparse_matrix = SparseMatrix::<Fr>::new(&matrix_data, 3, 3);

    assert_eq!(
      sparse_matrix.data,
      vec![Fr::from(2), Fr::from(3), Fr::from(4)]
    );
    assert_eq!(sparse_matrix.indices, vec![1, 2, 0]);
    assert_eq!(sparse_matrix.indptr, vec![0, 1, 2, 3]);
  }

  #[test]
  fn test_matrix_vector_multiplication() {
    let matrix_data = vec![
      (0, 1, Fr::from(2)),
      (0, 2, Fr::from(7)),
      (1, 2, Fr::from(3)),
      (2, 0, Fr::from(4)),
    ];
    let sparse_matrix = SparseMatrix::<Fr>::new(&matrix_data, 3, 3);
    let vector = vec![Fr::from(1), Fr::from(2), Fr::from(3)];

    let result = sparse_matrix.multiply_vec(&vector);

    assert_eq!(
      result.unwrap(),
      vec![Fr::from(25), Fr::from(9), Fr::from(4)]
    );
  }

  fn coo_strategy() -> BoxedStrategy<Vec<(usize, usize, FWrap<Fr>)>> {
    let coo_strategy = any::<FWrap<Fr>>().prop_flat_map(|f| (0usize..100, 0usize..100, Just(f)));
    proptest::collection::vec(coo_strategy, 10).boxed()
  }

  proptest! {
      #[test]
      fn test_matrix_iter(mut coo_matrix in coo_strategy()) {
        // process the randomly generated coo matrix
        coo_matrix.sort_by_key(|(row, col, _val)| (*row, *col));
        coo_matrix.dedup_by_key(|(row, col, _val)| (*row, *col));
        let coo_matrix = coo_matrix.into_iter().map(|(row, col, val)| { (row, col, val.0) }).collect::<Vec<_>>();

        let matrix = SparseMatrix::new(&coo_matrix, 100, 100);

        prop_assert_eq!(coo_matrix, matrix.iter().collect::<Vec<_>>());
    }
  }

  #[test]
  fn test_multiply_vec_witness_basic() {
    // Build a 3×3 i32 matrix manually in CSR format
    // z is bit-valued (0/1), so this tests conditional addition
    let matrix = SparseMatrix::<i32> {
      data: vec![2, 7, 3, 4],
      indices: vec![1, 2, 2, 0],
      indptr: vec![0, 2, 3, 4],
      cols: 3,
    };
    let z: Vec<i8> = vec![1, 1, 1]; // all bits set
    let result = matrix.multiply_vec_witness(&z).unwrap();
    // Row 0: 2*1 + 7*1 = 9
    // Row 1: 3*1 = 3
    // Row 2: 4*1 = 4
    assert_eq!(result, vec![9, 3, 4]);

    // Test with zeros
    let z2: Vec<i8> = vec![0, 1, 0]; // only bit 1 set
    let result2 = matrix.multiply_vec_witness(&z2).unwrap();
    // Row 0: 2*1 = 2
    // Row 1: 0
    // Row 2: 0
    assert_eq!(result2, vec![2, 0, 0]);
  }
}
