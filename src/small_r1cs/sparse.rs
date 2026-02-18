//! Sparse matrix with generic coefficient type.
//!
//! `SmallSparseMatrix<C>` stores matrix coefficients of type C (e.g., i32)
//! in CSR (Compressed Sparse Row) format for efficient matrix-vector multiplication.

use super::traits::{Accumulator, Coefficient, WideningMul, Witness};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// CSR sparse matrix with coefficient type C.
///
/// For SHA-256 with C=i32, coefficients are bounded by 2^16 (from 2-limb addition).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SmallSparseMatrix<C: Coefficient> {
    /// Non-zero coefficient values.
    pub data: Vec<C>,
    /// Column indices corresponding to each value in `data`.
    pub indices: Vec<usize>,
    /// Row pointers: row i spans indices[indptr[i]..indptr[i+1]].
    pub indptr: Vec<usize>,
    /// Number of columns in the matrix.
    pub cols: usize,
}

impl<C: Coefficient> SmallSparseMatrix<C> {
    /// Create an empty matrix.
    pub fn empty() -> Self {
        Self {
            data: vec![],
            indices: vec![],
            indptr: vec![0],
            cols: 0,
        }
    }

    /// Number of rows in the matrix.
    pub fn rows(&self) -> usize {
        self.indptr.len().saturating_sub(1)
    }

    /// Number of non-zero elements.
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Matrix-vector multiply: result[i] = Σ_j M[i,j] × z[j].
    ///
    /// Input: z is Vec<W> (e.g., i32 witnesses)
    /// Output: Vec<Acc> (e.g., i64 accumulators)
    ///
    /// Uses widening multiplication to avoid overflow.
    pub fn multiply_vec<W, Acc>(&self, z: &[W]) -> Vec<Acc>
    where
        W: Witness + Send + Sync,
        C: WideningMul<W, Acc> + Send + Sync,
        Acc: Accumulator + Send,
    {
        self.indptr
            .par_windows(2)
            .map(|ptrs| {
                let start = ptrs[0];
                let end = ptrs[1];
                let mut acc = Acc::zero();
                for i in start..end {
                    let coeff = self.data[i];
                    let col = self.indices[i];
                    acc = acc + coeff.wide_mul(z[col]);
                }
                acc
            })
            .collect()
    }

    /// Sequential (non-parallel) matrix-vector multiply.
    ///
    /// Useful for small matrices where parallel overhead isn't worth it.
    pub fn multiply_vec_seq<W, Acc>(&self, z: &[W]) -> Vec<Acc>
    where
        W: Witness,
        C: WideningMul<W, Acc>,
        Acc: Accumulator,
    {
        self.indptr
            .windows(2)
            .map(|ptrs| {
                let start = ptrs[0];
                let end = ptrs[1];
                let mut acc = Acc::zero();
                for i in start..end {
                    let coeff = self.data[i];
                    let col = self.indices[i];
                    acc = acc + coeff.wide_mul(z[col]);
                }
                acc
            })
            .collect()
    }

    /// Get a single row's coefficients and column indices.
    pub fn get_row(&self, row: usize) -> Option<(&[C], &[usize])> {
        if row >= self.rows() {
            return None;
        }
        let start = self.indptr[row];
        let end = self.indptr[row + 1];
        Some((&self.data[start..end], &self.indices[start..end]))
    }

    /// Iterate over rows, yielding (row_index, coefficients, column_indices).
    pub fn iter_rows(&self) -> impl Iterator<Item = (usize, &[C], &[usize])> {
        self.indptr.windows(2).enumerate().map(|(i, ptrs)| {
            let start = ptrs[0];
            let end = ptrs[1];
            (i, &self.data[start..end], &self.indices[start..end])
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multiply_vec() {
        // Matrix:
        // [1, 2, 0]
        // [0, 3, 4]
        let matrix = SmallSparseMatrix {
            data: vec![1i32, 2, 3, 4],
            indices: vec![0, 1, 1, 2],
            indptr: vec![0, 2, 4],
            cols: 3,
        };

        let z: Vec<i32> = vec![1, 2, 3];
        let result: Vec<i64> = matrix.multiply_vec(&z);

        // Row 0: 1*1 + 2*2 = 5
        // Row 1: 3*2 + 4*3 = 18
        assert_eq!(result, vec![5, 18]);
    }

    #[test]
    fn test_empty_matrix() {
        let matrix: SmallSparseMatrix<i32> = SmallSparseMatrix::empty();
        assert_eq!(matrix.rows(), 0);
        assert_eq!(matrix.nnz(), 0);
    }

    #[test]
    fn test_get_row() {
        let matrix = SmallSparseMatrix {
            data: vec![1i32, 2, 3],
            indices: vec![0, 2, 1],
            indptr: vec![0, 2, 3],
            cols: 3,
        };

        let (coeffs, cols) = matrix.get_row(0).unwrap();
        assert_eq!(coeffs, &[1, 2]);
        assert_eq!(cols, &[0, 2]);

        let (coeffs, cols) = matrix.get_row(1).unwrap();
        assert_eq!(coeffs, &[3]);
        assert_eq!(cols, &[1]);

        assert!(matrix.get_row(2).is_none());
    }

    #[test]
    fn test_negative_coefficients() {
        // Matrix with negative coefficients
        let matrix = SmallSparseMatrix {
            data: vec![1i32, -2, 3],
            indices: vec![0, 1, 0],
            indptr: vec![0, 2, 3],
            cols: 2,
        };

        let z: Vec<i32> = vec![10, 5];
        let result: Vec<i64> = matrix.multiply_vec(&z);

        // Row 0: 1*10 + (-2)*5 = 0
        // Row 1: 3*10 = 30
        assert_eq!(result, vec![0, 30]);
    }
}
