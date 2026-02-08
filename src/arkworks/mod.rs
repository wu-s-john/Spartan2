//! Arkworks Bridge for Spartan2
//!
//! This module provides conversion functions between arkworks R1CS types
//! and Spartan's R1CS types.
//!
//! ## Key Conversions
//!
//! - **Field elements**: `ark_ff::PrimeField` ↔ `ff::PrimeField` via bytes
//! - **Matrices**: `ark_relations::r1cs::Matrix<F>` → `SparseMatrix<F>` (CSR format)
//! - **Column layout**: arkworks `[1 | instances | witnesses]` → Spartan `[witnesses | 1 | public_io]`

use crate::r1cs::SparseMatrix;
use ark_ff::PrimeField as ArkPrimeField;
use ark_relations::r1cs::ConstraintMatrices;
use ff::PrimeField as FfPrimeField;

/// Convert an arkworks field element to an ff field element via little-endian bytes.
///
/// Both fields must have the same modulus for this to be correct.
pub fn ark_to_ff<ArkF, F>(ark_val: &ArkF) -> F
where
  ArkF: ArkPrimeField,
  F: FfPrimeField,
{
  // Get little-endian byte representation from arkworks
  let mut bytes = Vec::new();
  ark_val
    .serialize_uncompressed(&mut bytes)
    .expect("serialization should not fail");

  // Convert to ff field element
  // ff's from_repr expects a specific repr type, so we use from_repr_vartime
  // which accepts the internal representation
  let mut repr = F::Repr::default();
  let repr_bytes = repr.as_mut();

  // Copy bytes, padding or truncating as needed
  let copy_len = bytes.len().min(repr_bytes.len());
  repr_bytes[..copy_len].copy_from_slice(&bytes[..copy_len]);

  F::from_repr_vartime(repr).expect("field element should be valid")
}

/// Convert an ff field element to an arkworks field element via little-endian bytes.
///
/// Both fields must have the same modulus for this to be correct.
pub fn ff_to_ark<F, ArkF>(ff_val: &F) -> ArkF
where
  F: FfPrimeField,
  ArkF: ArkPrimeField,
{
  // Get byte representation from ff
  let repr = ff_val.to_repr();
  let bytes = repr.as_ref();

  // Convert to arkworks field element
  ArkF::from_le_bytes_mod_order(bytes)
}

/// Remap a column index from arkworks layout to Spartan layout.
///
/// Arkworks z-vector: `[1 | instance_vars | witness_vars]`
/// - Column 0: constant 1
/// - Columns 1..num_instance: public instance variables
/// - Columns num_instance..: witness variables
///
/// Spartan z-vector: `[witness_vars | 1 | public_io]`
/// - Columns 0..num_witness: witness variables
/// - Column num_witness: constant 1
/// - Columns num_witness+1..: public inputs
#[inline]
fn remap_column(ark_col: usize, num_instance: usize, num_witness: usize) -> usize {
  if ark_col == 0 {
    // constant 1: arkworks col 0 → spartan col num_witness
    num_witness
  } else if ark_col < num_instance {
    // instance vars: arkworks cols 1..num_instance → spartan cols num_witness+1..
    num_witness + ark_col
  } else {
    // witness vars: arkworks cols num_instance.. → spartan cols 0..
    ark_col - num_instance
  }
}

/// Convert an arkworks sparse matrix to Spartan's CSR format.
///
/// Arkworks uses `Vec<Vec<(coeff, col_idx)>>` (row-major, list of rows).
/// Spartan uses CSR format with `data`, `indices`, `indptr`.
///
/// This function also remaps column indices from arkworks layout to Spartan layout.
pub fn ark_matrix_to_spartan<ArkF, F>(
  ark_matrix: &[Vec<(ArkF, usize)>],
  num_instance: usize,
  num_witness: usize,
) -> SparseMatrix<F>
where
  ArkF: ArkPrimeField,
  F: FfPrimeField,
{
  let num_rows = ark_matrix.len();
  let num_cols = num_instance + num_witness;

  let mut data = Vec::new();
  let mut indices = Vec::new();
  let mut indptr = Vec::with_capacity(num_rows + 1);
  indptr.push(0);

  for row in ark_matrix {
    // Collect and remap entries for this row
    let mut row_entries: Vec<(usize, F)> = row
      .iter()
      .map(|(coeff, col)| {
        let spartan_col = remap_column(*col, num_instance, num_witness);
        let spartan_coeff = ark_to_ff(coeff);
        (spartan_col, spartan_coeff)
      })
      .collect();

    // Sort by column index (CSR requires sorted columns within each row)
    row_entries.sort_by_key(|(col, _)| *col);

    // Append to CSR arrays
    for (col, coeff) in row_entries {
      indices.push(col);
      data.push(coeff);
    }

    indptr.push(data.len());
  }

  SparseMatrix {
    data,
    indices,
    indptr,
    cols: num_cols,
  }
}

/// Convert arkworks `ConstraintMatrices` to Spartan's (A, B, C) `SparseMatrix` tuple.
///
/// This is the main entry point for converting a complete R1CS from arkworks to Spartan.
pub fn convert_constraint_matrices<ArkF, F>(
  matrices: &ConstraintMatrices<ArkF>,
) -> (SparseMatrix<F>, SparseMatrix<F>, SparseMatrix<F>)
where
  ArkF: ArkPrimeField,
  F: FfPrimeField,
{
  let num_instance = matrices.num_instance_variables;
  let num_witness = matrices.num_witness_variables;

  let a = ark_matrix_to_spartan(&matrices.a, num_instance, num_witness);
  let b = ark_matrix_to_spartan(&matrices.b, num_instance, num_witness);
  let c = ark_matrix_to_spartan(&matrices.c, num_instance, num_witness);

  (a, b, c)
}

#[cfg(test)]
mod tests {
  use super::*;
  use ark_bn254::Fr as ArkFr;
  use halo2curves::bn256::Fr as Bn256Fr;

  #[test]
  fn test_field_conversion_roundtrip() {
    // Test zero
    let ark_zero = ArkFr::from(0u64);
    let ff_zero: Bn256Fr = ark_to_ff(&ark_zero);
    let ark_back: ArkFr = ff_to_ark(&ff_zero);
    assert_eq!(ark_zero, ark_back);

    // Test one
    let ark_one = ArkFr::from(1u64);
    let ff_one: Bn256Fr = ark_to_ff(&ark_one);
    let ark_back: ArkFr = ff_to_ark(&ff_one);
    assert_eq!(ark_one, ark_back);

    // Test arbitrary value
    let ark_val = ArkFr::from(12345678901234567890u64);
    let ff_val: Bn256Fr = ark_to_ff(&ark_val);
    let ark_back: ArkFr = ff_to_ark(&ff_val);
    assert_eq!(ark_val, ark_back);

    // Test negative (field subtraction)
    let ark_neg = -ArkFr::from(42u64);
    let ff_neg: Bn256Fr = ark_to_ff(&ark_neg);
    let ark_back: ArkFr = ff_to_ark(&ff_neg);
    assert_eq!(ark_neg, ark_back);
  }

  #[test]
  fn test_ff_to_ark_roundtrip() {
    // Test from ff side
    let ff_val = Bn256Fr::from(999999u64);
    let ark_val: ArkFr = ff_to_ark(&ff_val);
    let ff_back: Bn256Fr = ark_to_ff(&ark_val);
    assert_eq!(ff_val, ff_back);
  }

  #[test]
  fn test_remap_column() {
    // Setup: 3 instance vars (including constant 1), 5 witness vars
    let num_instance = 3;
    let num_witness = 5;

    // Arkworks layout: [1, inst1, inst2, wit0, wit1, wit2, wit3, wit4]
    //                   0    1      2      3     4     5     6     7
    // Spartan layout:  [wit0, wit1, wit2, wit3, wit4, 1, inst1, inst2]
    //                   0      1     2     3     4    5    6      7

    // constant 1: ark col 0 → spartan col 5 (num_witness)
    assert_eq!(remap_column(0, num_instance, num_witness), 5);

    // instance var 1: ark col 1 → spartan col 6 (num_witness + 1)
    assert_eq!(remap_column(1, num_instance, num_witness), 6);

    // instance var 2: ark col 2 → spartan col 7 (num_witness + 2)
    assert_eq!(remap_column(2, num_instance, num_witness), 7);

    // witness var 0: ark col 3 → spartan col 0
    assert_eq!(remap_column(3, num_instance, num_witness), 0);

    // witness var 1: ark col 4 → spartan col 1
    assert_eq!(remap_column(4, num_instance, num_witness), 1);

    // witness var 4: ark col 7 → spartan col 4
    assert_eq!(remap_column(7, num_instance, num_witness), 4);
  }

  #[test]
  fn test_ark_matrix_to_spartan_simple() {
    // Simple 2x4 matrix in arkworks format
    // Row 0: [(1, col 0), (2, col 3)] - constant 1 and witness 0
    // Row 1: [(3, col 1), (4, col 2)] - instance 1 and instance 2
    let ark_matrix: Vec<Vec<(ArkFr, usize)>> = vec![
      vec![(ArkFr::from(1u64), 0), (ArkFr::from(2u64), 3)],
      vec![(ArkFr::from(3u64), 1), (ArkFr::from(4u64), 2)],
    ];

    let num_instance = 3; // includes constant 1
    let num_witness = 2;

    let spartan: SparseMatrix<Bn256Fr> =
      ark_matrix_to_spartan(&ark_matrix, num_instance, num_witness);

    // Verify structure
    assert_eq!(spartan.indptr, vec![0, 2, 4]); // 2 entries per row
    assert_eq!(spartan.cols, 5); // 3 instance + 2 witness

    // Row 0: arkworks (1, col0), (2, col3) → spartan (2, col0), (1, col2)
    // col0 (constant) → col 2 (num_witness)
    // col3 (witness0) → col 0
    // After sorting by column: [(2, 0), (1, 2)]
    assert_eq!(spartan.indices[0], 0); // witness 0
    assert_eq!(spartan.indices[1], 2); // constant 1
    assert_eq!(spartan.data[0], Bn256Fr::from(2u64));
    assert_eq!(spartan.data[1], Bn256Fr::from(1u64));

    // Row 1: arkworks (3, col1), (4, col2) → spartan cols 3, 4
    // col1 (instance1) → col 3 (num_witness + 1)
    // col2 (instance2) → col 4 (num_witness + 2)
    assert_eq!(spartan.indices[2], 3);
    assert_eq!(spartan.indices[3], 4);
    assert_eq!(spartan.data[2], Bn256Fr::from(3u64));
    assert_eq!(spartan.data[3], Bn256Fr::from(4u64));
  }
}
