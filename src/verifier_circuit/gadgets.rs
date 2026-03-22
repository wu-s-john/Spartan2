//! Gadgets for in-circuit verification of Spartan ZK proof components.
//!
//! These gadgets verify the NIFS scalar fold and relaxed R1CS satisfiability,
//! which are currently checked natively by `SpartanZkSNARK::verify()`.

use crate::{r1cs::R1CSShape, traits::Engine};
use bellpepper_core::{ConstraintSystem, SynthesisError, num::AllocatedNum};

/// Enforces that `(u_f, X_f)` is the correct scalar fold of relaxed instance
/// `(u1, X1)` with regular instance `X2` (where u2 = 1) using challenge `r`.
///
/// Constraints:
///   u_f = u1 + r                         (1 linear)
///   X_f[k] = X1[k] + r · X2[k]   ∀k     (|X| muls)
pub fn enforce_nifs_scalar_fold<E: Engine, CS: ConstraintSystem<E::Scalar>>(
  mut cs: CS,
  u1: &AllocatedNum<E::Scalar>,
  X1: &[AllocatedNum<E::Scalar>],
  X2: &[AllocatedNum<E::Scalar>],
  r: &AllocatedNum<E::Scalar>,
  u_f: &AllocatedNum<E::Scalar>,
  X_f: &[AllocatedNum<E::Scalar>],
) -> Result<(), SynthesisError> {
  assert_eq!(X1.len(), X2.len());
  assert_eq!(X1.len(), X_f.len());

  // u_f = u1 + r  ⟺  (u_f - u1 - r) · 1 = 0
  cs.enforce(
    || "u_fold",
    |lc| lc + u_f.get_variable() - u1.get_variable() - r.get_variable(),
    |lc| lc + CS::one(),
    |lc| lc,
  );

  // For each k: X_f[k] = X1[k] + r · X2[k]
  for (k, ((x1, x2), xf)) in X1.iter().zip(X2.iter()).zip(X_f.iter()).enumerate() {
    // delta_k = r · X2[k]
    let delta = AllocatedNum::alloc(cs.namespace(|| format!("delta_{k}")), || {
      let r_val = r.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let x2_val = x2.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      Ok(r_val * x2_val)
    })?;
    cs.enforce(
      || format!("r_mul_X2_{k}"),
      |lc| lc + r.get_variable(),
      |lc| lc + x2.get_variable(),
      |lc| lc + delta.get_variable(),
    );

    // X_f[k] = X1[k] + delta_k  ⟺  (X_f[k] - X1[k] - delta_k) · 1 = 0
    cs.enforce(
      || format!("X_fold_{k}"),
      |lc| lc + xf.get_variable() - x1.get_variable() - delta.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc,
    );
  }

  Ok(())
}

/// Enforces relaxed R1CS satisfiability for a known, constant shape.
///
/// Checks: A·z_f ∘ B·z_f = u_f · C·z_f + E_f
///
/// The matrices A, B, C from `shape` are embedded as constant coefficients
/// in the linear combinations — they cost zero constraints.
/// Each constraint row costs 2 multiplication constraints.
pub fn enforce_relaxed_r1cs_sat<E: Engine, CS: ConstraintSystem<E::Scalar>>(
  mut cs: CS,
  shape: &R1CSShape<E>,
  z_f: &[AllocatedNum<E::Scalar>],
  u_f: &AllocatedNum<E::Scalar>,
  E_f: &[AllocatedNum<E::Scalar>],
) -> Result<(), SynthesisError> {
  assert_eq!(z_f.len(), shape.num_vars + 1 + shape.num_io);
  assert_eq!(E_f.len(), shape.num_cons);

  for i in 0..shape.num_cons {
    // Compute witness values for L_i, R_i, O_i
    let l_val = compute_lc_value(&shape.A, i, z_f);
    let r_val = compute_lc_value(&shape.B, i, z_f);
    let o_val = compute_lc_value(&shape.C, i, z_f);

    // P_i = L_i · R_i
    let P_i = AllocatedNum::alloc(cs.namespace(|| format!("P_{i}")), || {
      let l = l_val.ok_or(SynthesisError::AssignmentMissing)?;
      let r = r_val.ok_or(SynthesisError::AssignmentMissing)?;
      Ok(l * r)
    })?;
    cs.enforce(
      || format!("LR_{i}"),
      |lc| build_lc::<E>(&shape.A, i, z_f, lc),
      |lc| build_lc::<E>(&shape.B, i, z_f, lc),
      |lc| lc + P_i.get_variable(),
    );

    // Q_i = u_f · O_i
    let Q_i = AllocatedNum::alloc(cs.namespace(|| format!("Q_{i}")), || {
      let u = u_f.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let o = o_val.ok_or(SynthesisError::AssignmentMissing)?;
      Ok(u * o)
    })?;
    cs.enforce(
      || format!("uO_{i}"),
      |lc| lc + u_f.get_variable(),
      |lc| build_lc::<E>(&shape.C, i, z_f, lc),
      |lc| lc + Q_i.get_variable(),
    );

    // P_i - Q_i = E_f[i]  ⟺  (P_i - Q_i - E_f[i]) · 1 = 0
    cs.enforce(
      || format!("PQE_{i}"),
      |lc| lc + P_i.get_variable() - Q_i.get_variable() - E_f[i].get_variable(),
      |lc| lc + CS::one(),
      |lc| lc,
    );
  }

  Ok(())
}

/// Build a linear combination from a sparse matrix row.
/// row i of matrix M: sum_j M[i,j] * z_f[j]
fn build_lc<E: Engine>(
  matrix: &crate::r1cs::SparseMatrix<E::Scalar>,
  row: usize,
  z_f: &[AllocatedNum<E::Scalar>],
  lc: bellpepper_core::LinearCombination<E::Scalar>,
) -> bellpepper_core::LinearCombination<E::Scalar> {
  let start = matrix.indptr[row];
  let end = matrix.indptr[row + 1];
  (start..end).fold(lc, |lc, pos| {
    let col = matrix.indices[pos];
    let val = matrix.data[pos];
    lc + (val, z_f[col].get_variable())
  })
}

/// Compute the value of a linear combination from a sparse matrix row.
fn compute_lc_value<F: ff::PrimeField>(
  matrix: &crate::r1cs::SparseMatrix<F>,
  row: usize,
  z_f: &[AllocatedNum<F>],
) -> Option<F> {
  let start = matrix.indptr[row];
  let end = matrix.indptr[row + 1];
  let mut acc = F::ZERO;
  for pos in start..end {
    let col = matrix.indices[pos];
    let val = matrix.data[pos];
    acc += val * z_f[col].get_value()?;
  }
  Some(acc)
}
