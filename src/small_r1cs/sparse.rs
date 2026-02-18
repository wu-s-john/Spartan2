//! Sparse matrix with generic coefficient type.
//!
//! This module re-exports `SparseMatrix` from `r1cs::sparse` as `SmallSparseMatrix`
//! for backwards compatibility. The unified `SparseMatrix<C>` now works with both
//! field elements and native integers.

// Re-export the unified SparseMatrix
pub use crate::r1cs::sparse::SparseMatrix as SmallSparseMatrix;

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_multiply_vec_widening() {
    // Matrix:
    // [1, 2, 0]
    // [0, 3, 4]
    let matrix: SmallSparseMatrix<i32> = SmallSparseMatrix {
      data: vec![1i32, 2, 3, 4],
      indices: vec![0, 1, 1, 2],
      indptr: vec![0, 2, 4],
      cols: 3,
    };

    let z: Vec<i32> = vec![1, 2, 3];
    let result: Vec<i64> = matrix.multiply_vec_widening(&z).unwrap();

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
    let matrix: SmallSparseMatrix<i32> = SmallSparseMatrix {
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
    let matrix: SmallSparseMatrix<i32> = SmallSparseMatrix {
      data: vec![1i32, -2, 3],
      indices: vec![0, 1, 0],
      indptr: vec![0, 2, 3],
      cols: 2,
    };

    let z: Vec<i32> = vec![10, 5];
    let result: Vec<i64> = matrix.multiply_vec_widening(&z).unwrap();

    // Row 0: 1*10 + (-2)*5 = 0
    // Row 1: 3*10 = 30
    assert_eq!(result, vec![0, 30]);
  }
}
