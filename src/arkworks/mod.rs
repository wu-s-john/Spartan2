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

use crate::{r1cs::SparseMatrix, traits::circuit::SpartanCircuit};
use ark_ff::PrimeField as ArkPrimeField;
use ark_relations::r1cs::{ConstraintMatrices, ConstraintSystemRef};
use ff::{Field as FfField, PrimeField as FfPrimeField};

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

/// Extract and convert witness/instance assignments from a finalized arkworks ConstraintSystem.
///
/// This function extracts the witness and public input assignments from an arkworks
/// constraint system and converts them to Spartan's expected format.
///
/// ## Layout Conversion
///
/// - **Arkworks instance**: `[1, pub1, pub2, ...]` (constant 1 at index 0)
/// - **Arkworks witness**: `[wit1, wit2, ...]`
/// - **Spartan W**: `[wit1, wit2, ...]` (same as arkworks witness)
/// - **Spartan X**: `[pub1, pub2, ...]` (arkworks instance without the constant 1)
///
/// ## Arguments
///
/// * `cs` - Reference to a finalized arkworks ConstraintSystem
///
/// ## Returns
///
/// A tuple `(W, X)` where:
/// - `W` is the witness vector (private inputs)
/// - `X` is the public input vector (public inputs, excluding the constant 1)
///
/// ## Panics
///
/// Panics if the constraint system cannot be borrowed (e.g., if it's still being modified).
pub fn extract_assignments<ArkF, F>(cs: &ConstraintSystemRef<ArkF>) -> (Vec<F>, Vec<F>)
where
  ArkF: ArkPrimeField,
  F: FfPrimeField,
{
  let binding = cs.borrow().expect("failed to borrow constraint system");

  // Witness: all private variables (no layout change needed)
  let w: Vec<F> = binding.witness_assignment.iter().map(ark_to_ff).collect();

  // Public inputs: skip the constant 1 at index 0
  let x: Vec<F> = binding
    .instance_assignment
    .iter()
    .skip(1)
    .map(ark_to_ff)
    .collect();

  (w, x)
}

use crate::traits::Engine;
use bellpepper_core::{ConstraintSystem, LinearCombination, SynthesisError, num::AllocatedNum};

/// Adapter that wraps arkworks R1CS data to implement `SpartanCircuit`.
///
/// This allows arkworks circuits to be proven using `SpartanZkSNARK`.
///
/// ## Usage
///
/// ```ignore
/// // 1. Synthesize your arkworks circuit
/// let cs = ConstraintSystem::<ArkFr>::new_ref();
/// // ... allocate variables and constraints ...
/// cs.finalize();
///
/// // 2. Extract matrices and assignments
/// let matrices = cs.to_matrices().unwrap();
/// let (A, B, C) = convert_constraint_matrices::<ArkFr, Fr>(&matrices);
/// let (W, X) = extract_assignments::<ArkFr, Fr>(&cs);
///
/// // 3. Create adapter
/// let adapter = ArkworksCircuitAdapter::new(
///     matrices.num_constraints,
///     matrices.num_witness_variables,
///     matrices.num_instance_variables - 1,
///     A, B, C, W, X,
/// );
///
/// // 4. Use with SpartanZkSNARK
/// let (pk, vk) = SpartanZkSNARK::<E>::setup(adapter.clone())?;
/// let prep = SpartanZkSNARK::<E>::prep_prove(&pk, adapter.clone(), false)?;
/// let snark = SpartanZkSNARK::<E>::prove(&pk, adapter, &prep, false)?;
/// snark.verify(&vk)?;
/// ```
#[derive(Clone, Debug)]
pub struct ArkworksCircuitAdapter<F: FfPrimeField> {
  /// Number of constraints
  pub num_constraints: usize,
  /// Number of witness variables
  pub num_witness: usize,
  /// Number of public input variables (excluding constant 1)
  pub num_public: usize,
  /// Constraint matrix A in Spartan's CSR format and column layout
  pub A: SparseMatrix<F>,
  /// Constraint matrix B in Spartan's CSR format and column layout
  pub B: SparseMatrix<F>,
  /// Constraint matrix C in Spartan's CSR format and column layout
  pub C: SparseMatrix<F>,
  /// Witness assignment (private inputs)
  pub W: Vec<F>,
  /// Public input assignment (public inputs, excluding constant 1)
  pub X: Vec<F>,
}

impl<F: FfPrimeField> ArkworksCircuitAdapter<F> {
  /// Create a new adapter from arkworks R1CS data.
  ///
  /// # Arguments
  /// * `num_constraints` - Number of R1CS constraints
  /// * `num_witness` - Number of witness variables
  /// * `num_public` - Number of public input variables (excluding constant 1)
  /// * `A, B, C` - Constraint matrices in Spartan's format (use `convert_constraint_matrices`)
  /// * `W` - Witness assignment (use `extract_assignments`)
  /// * `X` - Public input assignment (use `extract_assignments`)
  pub fn new(
    num_constraints: usize,
    num_witness: usize,
    num_public: usize,
    A: SparseMatrix<F>,
    B: SparseMatrix<F>,
    C: SparseMatrix<F>,
    W: Vec<F>,
    X: Vec<F>,
  ) -> Self {
    Self {
      num_constraints,
      num_witness,
      num_public,
      A,
      B,
      C,
      W,
      X,
    }
  }

  /// Build a linear combination from a CSR matrix row.
  ///
  /// Spartan's z-vector layout: [W | 1 | X]
  /// - columns 0..num_witness: witness variables
  /// - column num_witness: constant 1
  /// - columns num_witness+1..: public inputs
  fn build_lc<CS: ConstraintSystem<F>>(
    &self,
    matrix: &SparseMatrix<F>,
    row: usize,
    witness_vars: &[AllocatedNum<F>],
    public_vars: &[AllocatedNum<F>],
  ) -> LinearCombination<F> {
    let start = matrix.indptr[row];
    let end = matrix.indptr[row + 1];

    let mut lc = LinearCombination::zero();

    for i in start..end {
      let col = matrix.indices[i];
      let coeff = matrix.data[i];

      if col < self.num_witness {
        // Witness variable
        lc = lc + (coeff, witness_vars[col].get_variable());
      } else if col == self.num_witness {
        // Constant 1
        lc = lc + (coeff, CS::one());
      } else {
        // Public input
        let pub_idx = col - self.num_witness - 1;
        lc = lc + (coeff, public_vars[pub_idx].get_variable());
      }
    }

    lc
  }
}

impl<E: Engine> SpartanCircuit<E> for ArkworksCircuitAdapter<E::Scalar>
where
  E::Scalar: FfPrimeField,
{
  fn public_values(&self) -> Result<Vec<E::Scalar>, SynthesisError> {
    Ok(self.X.clone())
  }

  fn shared<CS: ConstraintSystem<E::Scalar>>(
    &self,
    _cs: &mut CS,
  ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
    // No shared variables for arkworks circuits
    Ok(vec![])
  }

  fn precommitted<CS: ConstraintSystem<E::Scalar>>(
    &self,
    _cs: &mut CS,
    _shared: &[AllocatedNum<E::Scalar>],
  ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
    // No precommitted variables for arkworks circuits
    Ok(vec![])
  }

  fn num_challenges(&self) -> usize {
    // No challenges for simple arkworks circuits
    0
  }

  fn synthesize<CS: ConstraintSystem<E::Scalar>>(
    &self,
    cs: &mut CS,
    _shared: &[AllocatedNum<E::Scalar>],
    _precommitted: &[AllocatedNum<E::Scalar>],
    _challenges: Option<&[E::Scalar]>,
  ) -> Result<(), SynthesisError> {
    // Allocate witness variables
    let witness_vars: Vec<AllocatedNum<E::Scalar>> = (0..self.num_witness)
      .map(|i| {
        let value = if i < self.W.len() {
          self.W[i]
        } else {
          E::Scalar::ZERO
        };
        AllocatedNum::alloc(cs.namespace(|| format!("w_{}", i)), || Ok(value))
      })
      .collect::<Result<Vec<_>, _>>()?;

    // Allocate public input variables
    let public_vars: Vec<AllocatedNum<E::Scalar>> = (0..self.num_public)
      .map(|i| -> Result<AllocatedNum<E::Scalar>, SynthesisError> {
        let value = if i < self.X.len() {
          self.X[i]
        } else {
          E::Scalar::ZERO
        };
        let var = AllocatedNum::alloc(cs.namespace(|| format!("x_{}", i)), || Ok(value))?;
        // Make it public
        var.inputize(cs.namespace(|| format!("pub_x_{}", i)))?;
        Ok(var)
      })
      .collect::<Result<Vec<_>, _>>()?;

    // Add constraints from matrices
    for row in 0..self.num_constraints {
      let a_lc = self.build_lc::<CS>(&self.A, row, &witness_vars, &public_vars);
      let b_lc = self.build_lc::<CS>(&self.B, row, &witness_vars, &public_vars);
      let c_lc = self.build_lc::<CS>(&self.C, row, &witness_vars, &public_vars);

      cs.enforce(
        || format!("constraint_{}", row),
        |_| a_lc,
        |_| b_lc,
        |_| c_lc,
      );
    }

    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use ark_bn254::Fr as ArkFr;
  use ark_r1cs_std::{alloc::AllocVar, eq::EqGadget, fields::fp::FpVar};
  use ark_relations::r1cs::ConstraintSystem;
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

  #[test]
  fn test_extract_assignments() {
    // Create a simple constraint system with known values
    let cs = ConstraintSystem::<ArkFr>::new_ref();

    // Allocate public inputs (these go to instance_assignment)
    let x = FpVar::new_input(cs.clone(), || Ok(ArkFr::from(3u64))).unwrap();
    let y = FpVar::new_input(cs.clone(), || Ok(ArkFr::from(35u64))).unwrap();

    // Allocate witnesses (these go to witness_assignment)
    let x_squared = FpVar::new_witness(cs.clone(), || Ok(ArkFr::from(9u64))).unwrap();
    let x_cubed = FpVar::new_witness(cs.clone(), || Ok(ArkFr::from(27u64))).unwrap();

    // Add constraints: x² = x * x, x³ = x² * x, y = x³ + x + 5
    let computed_x_squared = &x * &x;
    x_squared.enforce_equal(&computed_x_squared).unwrap();

    let computed_x_cubed = &x_squared * &x;
    x_cubed.enforce_equal(&computed_x_cubed).unwrap();

    let five = FpVar::new_constant(cs.clone(), ArkFr::from(5u64)).unwrap();
    let computed_y = &x_cubed + &x + &five;
    y.enforce_equal(&computed_y).unwrap();

    cs.finalize();

    // Extract assignments
    let (w, x_pub) = extract_assignments::<ArkFr, Bn256Fr>(&cs);

    // Verify public inputs (X): should be [3, 35] (x and y values)
    assert_eq!(x_pub.len(), 2);
    assert_eq!(x_pub[0], Bn256Fr::from(3u64)); // x = 3
    assert_eq!(x_pub[1], Bn256Fr::from(35u64)); // y = 35 = 27 + 3 + 5

    // Verify witnesses (W): should contain x², x³, and intermediate values
    // The exact count depends on how arkworks allocates intermediate variables
    assert!(!w.is_empty());

    // Verify the constraint system is satisfied
    assert!(cs.is_satisfied().unwrap());
  }

  #[test]
  fn test_extract_assignments_with_r1cs_shape() {
    // Create a simple constraint system: x³ + x + 5 = y
    let cs = ConstraintSystem::<ArkFr>::new_ref();

    let x_val = ArkFr::from(3u64);
    let x_squared_val = ArkFr::from(9u64);
    let x_cubed_val = ArkFr::from(27u64);
    let y_val = ArkFr::from(35u64); // 27 + 3 + 5 = 35

    // Public: x (input), y (output)
    let x = FpVar::new_input(cs.clone(), || Ok(x_val)).unwrap();
    let y = FpVar::new_input(cs.clone(), || Ok(y_val)).unwrap();

    // Witness: intermediate values
    let x_squared = FpVar::new_witness(cs.clone(), || Ok(x_squared_val)).unwrap();
    let x_cubed = FpVar::new_witness(cs.clone(), || Ok(x_cubed_val)).unwrap();

    // Constraints
    let computed_x_squared = &x * &x;
    x_squared.enforce_equal(&computed_x_squared).unwrap();

    let computed_x_cubed = &x_squared * &x;
    x_cubed.enforce_equal(&computed_x_cubed).unwrap();

    let five = FpVar::new_constant(cs.clone(), ArkFr::from(5u64)).unwrap();
    let computed_y = &x_cubed + &x + &five;
    y.enforce_equal(&computed_y).unwrap();

    cs.finalize();

    // Verify arkworks constraint system is satisfied
    assert!(
      cs.is_satisfied().unwrap(),
      "Arkworks CS should be satisfied"
    );

    // Get matrices
    let matrices = cs.to_matrices().unwrap();
    let (a, b, c) = convert_constraint_matrices::<ArkFr, Bn256Fr>(&matrices);

    // Get assignments
    let (mut w, x_pub) = extract_assignments::<ArkFr, Bn256Fr>(&cs);

    // Build R1CSShape (don't pad - padding is only needed for proving, not is_sat)
    use crate::{
      provider::Bn254Engine,
      r1cs::{R1CSInstance, R1CSShape, R1CSWitness},
    };

    let shape = R1CSShape::<Bn254Engine>::new(
      matrices.num_constraints,
      matrices.num_witness_variables,
      matrices.num_instance_variables - 1, // exclude constant
      a,
      b,
      c,
    )
    .unwrap();

    // Get commitment key
    let (ck, _vk) = shape.commitment_key();

    // Create witness (this pads W to match shape.num_vars)
    let (witness, comm_w) = R1CSWitness::<Bn254Engine>::new(&ck, &shape, &mut w, false).unwrap();

    // Create instance
    let instance = R1CSInstance::<Bn254Engine>::new(&shape, &comm_w, &x_pub).unwrap();

    // This is the ultimate test: verify R1CS satisfaction
    shape
      .is_sat(&ck, &instance, &witness)
      .expect("R1CS should be satisfied");
  }

  /// Test full SpartanZkSNARK prove/verify flow with arkworks circuit via adapter.
  #[test]
  fn test_arkworks_circuit_adapter_zksnark() {
    use crate::{provider::Bn254Engine, spartan_zk::SpartanZkSNARK, traits::snark::R1CSSNARKTrait};

    // 1. Synthesize arkworks circuit: x³ + x + 5 = y
    let cs = ConstraintSystem::<ArkFr>::new_ref();

    let x_val = ArkFr::from(3u64);
    let x_squared_val = ArkFr::from(9u64);
    let x_cubed_val = ArkFr::from(27u64);
    let y_val = ArkFr::from(35u64); // 27 + 3 + 5 = 35

    // Public: x (input), y (output)
    let x = FpVar::new_input(cs.clone(), || Ok(x_val)).unwrap();
    let y = FpVar::new_input(cs.clone(), || Ok(y_val)).unwrap();

    // Witness: intermediate values
    let x_squared = FpVar::new_witness(cs.clone(), || Ok(x_squared_val)).unwrap();
    let x_cubed = FpVar::new_witness(cs.clone(), || Ok(x_cubed_val)).unwrap();

    // Constraints
    let computed_x_squared = &x * &x;
    x_squared.enforce_equal(&computed_x_squared).unwrap();

    let computed_x_cubed = &x_squared * &x;
    x_cubed.enforce_equal(&computed_x_cubed).unwrap();

    let five = FpVar::new_constant(cs.clone(), ArkFr::from(5u64)).unwrap();
    let computed_y = &x_cubed + &x + &five;
    y.enforce_equal(&computed_y).unwrap();

    cs.finalize();

    // Verify arkworks constraint system is satisfied
    assert!(
      cs.is_satisfied().unwrap(),
      "Arkworks CS should be satisfied"
    );

    // 2. Extract matrices and assignments
    let matrices = cs.to_matrices().unwrap();
    let (a, b, c) = convert_constraint_matrices::<ArkFr, Bn256Fr>(&matrices);
    let (w, x_pub) = extract_assignments::<ArkFr, Bn256Fr>(&cs);

    // 3. Create adapter
    let adapter = ArkworksCircuitAdapter::new(
      matrices.num_constraints,
      matrices.num_witness_variables,
      matrices.num_instance_variables - 1, // exclude constant 1
      a,
      b,
      c,
      w,
      x_pub.clone(),
    );

    // 4. Use with SpartanZkSNARK
    let (pk, vk) =
      SpartanZkSNARK::<Bn254Engine>::setup(adapter.clone()).expect("setup should succeed");

    let prep = SpartanZkSNARK::<Bn254Engine>::prep_prove(&pk, adapter.clone(), false)
      .expect("prep should succeed");

    let snark = SpartanZkSNARK::<Bn254Engine>::prove(&pk, adapter, &prep, false)
      .expect("prove should succeed");

    // 5. Verify the proof
    let result = snark.verify(&vk);
    assert!(result.is_ok(), "verification should succeed");

    // 6. Check that public outputs match
    let public_outputs = result.unwrap();
    assert_eq!(public_outputs.len(), 2);
    assert_eq!(public_outputs[0], Bn256Fr::from(3u64)); // x = 3
    assert_eq!(public_outputs[1], Bn256Fr::from(35u64)); // y = 35
  }
}
