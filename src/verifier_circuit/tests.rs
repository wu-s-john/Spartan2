use super::gadgets::{enforce_nifs_scalar_fold, enforce_relaxed_r1cs_sat};
use crate::{r1cs::R1CSShape, traits::Engine};
use bellpepper_core::{num::AllocatedNum, test_cs::TestConstraintSystem, ConstraintSystem};
use ff::Field;

type E = crate::provider::PallasHyraxEngine;
type Scalar = <E as Engine>::Scalar;

/// Helper to allocate a vector of scalars as witness variables.
fn alloc_vec<CS: ConstraintSystem<Scalar>>(
  cs: &mut CS,
  prefix: &str,
  vals: &[Scalar],
) -> Vec<AllocatedNum<Scalar>> {
  vals
    .iter()
    .enumerate()
    .map(|(i, v)| {
      AllocatedNum::alloc(cs.namespace(|| format!("{prefix}_{i}")), || Ok(*v)).unwrap()
    })
    .collect()
}

// ─── NIFS scalar fold tests ─────────────────────────────────────────────

#[test]
fn test_enforce_nifs_scalar_fold() {
  let mut rng = rand_core::OsRng;
  let n = 10; // number of public IO entries

  // Random values
  let u1 = Scalar::random(&mut rng);
  let X1: Vec<Scalar> = (0..n).map(|_| Scalar::random(&mut rng)).collect();
  let X2: Vec<Scalar> = (0..n).map(|_| Scalar::random(&mut rng)).collect();
  let r = Scalar::random(&mut rng);

  // Compute correct fold
  let u_f = u1 + r;
  let X_f: Vec<Scalar> = X1.iter().zip(X2.iter()).map(|(x1, x2)| *x1 + r * x2).collect();

  let mut cs = TestConstraintSystem::<Scalar>::new();

  let u1_var = AllocatedNum::alloc(cs.namespace(|| "u1"), || Ok(u1)).unwrap();
  let r_var = AllocatedNum::alloc(cs.namespace(|| "r"), || Ok(r)).unwrap();
  let u_f_var = AllocatedNum::alloc(cs.namespace(|| "u_f"), || Ok(u_f)).unwrap();
  let X1_vars = alloc_vec(&mut cs, "X1", &X1);
  let X2_vars = alloc_vec(&mut cs, "X2", &X2);
  let X_f_vars = alloc_vec(&mut cs, "X_f", &X_f);

  enforce_nifs_scalar_fold::<E, _>(
    cs.namespace(|| "nifs_fold"),
    &u1_var,
    &X1_vars,
    &X2_vars,
    &r_var,
    &u_f_var,
    &X_f_vars,
  )
  .unwrap();

  assert!(cs.is_satisfied(), "NIFS scalar fold should be satisfied");
  // |X| mul constraints (for delta_k) + |X| linear (X_fold) + 1 linear (u_fold)
  // TestConstraintSystem counts all enforce() calls
  let num_constraints = cs.num_constraints();
  // We expect: 1 (u_fold) + n (r_mul) + n (X_fold) = 2n + 1
  assert_eq!(num_constraints, 2 * n + 1, "unexpected constraint count");
}

#[test]
fn test_enforce_nifs_scalar_fold_bad_witness() {
  let mut rng = rand_core::OsRng;
  let n = 5;

  let u1 = Scalar::random(&mut rng);
  let X1: Vec<Scalar> = (0..n).map(|_| Scalar::random(&mut rng)).collect();
  let X2: Vec<Scalar> = (0..n).map(|_| Scalar::random(&mut rng)).collect();
  let r = Scalar::random(&mut rng);

  let u_f = u1 + r;
  let mut X_f: Vec<Scalar> = X1.iter().zip(X2.iter()).map(|(x1, x2)| *x1 + r * x2).collect();
  // Corrupt one entry
  X_f[2] += Scalar::ONE;

  let mut cs = TestConstraintSystem::<Scalar>::new();

  let u1_var = AllocatedNum::alloc(cs.namespace(|| "u1"), || Ok(u1)).unwrap();
  let r_var = AllocatedNum::alloc(cs.namespace(|| "r"), || Ok(r)).unwrap();
  let u_f_var = AllocatedNum::alloc(cs.namespace(|| "u_f"), || Ok(u_f)).unwrap();
  let X1_vars = alloc_vec(&mut cs, "X1", &X1);
  let X2_vars = alloc_vec(&mut cs, "X2", &X2);
  let X_f_vars = alloc_vec(&mut cs, "X_f", &X_f);

  enforce_nifs_scalar_fold::<E, _>(
    cs.namespace(|| "nifs_fold"),
    &u1_var,
    &X1_vars,
    &X2_vars,
    &r_var,
    &u_f_var,
    &X_f_vars,
  )
  .unwrap();

  assert!(
    !cs.is_satisfied(),
    "NIFS scalar fold with bad witness should NOT be satisfied"
  );
}

// ─── Relaxed R1CS sat tests ─────────────────────────────────────────────

/// Build a tiny R1CS shape for testing: single constraint x * x = y
/// with num_vars=2 (x, y), num_io=0, num_cons=1
/// A = [[1, 0, 0]], B = [[1, 0, 0]], C = [[0, 1, 0]]
/// z = [x, y, u] where u is the relaxation parameter
fn tiny_test_shape() -> R1CSShape<E> {
  use crate::r1cs::SparseMatrix;

  // 1 constraint, 2 vars, 0 io
  // z = [x, y, u]  (num_vars=2, then u, then io)
  // A*z = x, B*z = x, C*z = y
  // So: x * x = u * y + E[0]
  let A = SparseMatrix {
    data: vec![Scalar::ONE],
    indices: vec![0],
    indptr: vec![0, 1],
    cols: 3,
  };
  let B = SparseMatrix {
    data: vec![Scalar::ONE],
    indices: vec![0],
    indptr: vec![0, 1],
    cols: 3,
  };
  let C = SparseMatrix {
    data: vec![Scalar::ONE],
    indices: vec![1],
    indptr: vec![0, 1],
    cols: 3,
  };

  R1CSShape::<E>::new(1, 2, 0, A, B, C).expect("valid shape")
}

#[test]
fn test_enforce_relaxed_r1cs_sat() {
  let shape = tiny_test_shape();

  // Pick x=3, so y should satisfy: x*x = u*y + E[0]
  // With u=1, E=[0]: 9 = 1*9 + 0 → y=9
  let x = Scalar::from(3u64);
  let y = Scalar::from(9u64);
  let u = Scalar::ONE;
  let e = Scalar::ZERO;

  // z_f = [x, y, u]
  let z_vals = vec![x, y, u];
  let E_vals = vec![e];

  let mut cs = TestConstraintSystem::<Scalar>::new();

  let z_f = alloc_vec(&mut cs, "z", &z_vals);
  let u_f = AllocatedNum::alloc(cs.namespace(|| "u_f"), || Ok(u)).unwrap();
  let E_f = alloc_vec(&mut cs, "E", &E_vals);

  // u_f should equal z_f[2] for consistency — enforce it
  cs.enforce(
    || "u_eq",
    |lc| lc + u_f.get_variable() - z_f[2].get_variable(),
    |lc| lc + TestConstraintSystem::<Scalar>::one(),
    |lc| lc,
  );

  enforce_relaxed_r1cs_sat::<E, _>(
    cs.namespace(|| "relaxed_sat"),
    &shape,
    &z_f,
    &u_f,
    &E_f,
  )
  .unwrap();

  assert!(
    cs.is_satisfied(),
    "Relaxed R1CS sat with valid witness should be satisfied"
  );
  // 2 mul constraints per row (P_i, Q_i) + 1 linear (PQE), + 1 for u_eq above
  // = 3 * num_cons + 1 = 4
  // But we count enforce() calls: 1 (u_eq) + 3 (per constraint: LR, uO, PQE)
  assert_eq!(cs.num_constraints(), 4);
}

#[test]
fn test_enforce_relaxed_r1cs_sat_with_error() {
  let shape = tiny_test_shape();

  // x=3, u=2, E=[1]: x*x = u*y + E[0] → 9 = 2*y + 1 → y = 4
  let x = Scalar::from(3u64);
  let y = Scalar::from(4u64);
  let u = Scalar::from(2u64);
  let e = Scalar::ONE;

  let z_vals = vec![x, y, u];
  let E_vals = vec![e];

  let mut cs = TestConstraintSystem::<Scalar>::new();

  let z_f = alloc_vec(&mut cs, "z", &z_vals);
  let u_f = AllocatedNum::alloc(cs.namespace(|| "u_f"), || Ok(u)).unwrap();
  let E_f = alloc_vec(&mut cs, "E", &E_vals);

  cs.enforce(
    || "u_eq",
    |lc| lc + u_f.get_variable() - z_f[2].get_variable(),
    |lc| lc + TestConstraintSystem::<Scalar>::one(),
    |lc| lc,
  );

  enforce_relaxed_r1cs_sat::<E, _>(
    cs.namespace(|| "relaxed_sat"),
    &shape,
    &z_f,
    &u_f,
    &E_f,
  )
  .unwrap();

  assert!(
    cs.is_satisfied(),
    "Relaxed R1CS sat with non-trivial u and E should be satisfied"
  );
}

#[test]
fn test_enforce_relaxed_r1cs_sat_bad_witness() {
  let shape = tiny_test_shape();

  // Correct: x=3, y=9, u=1, E=0
  // Corrupt E to make it unsatisfied
  let x = Scalar::from(3u64);
  let y = Scalar::from(9u64);
  let u = Scalar::ONE;
  let e = Scalar::ONE; // Wrong! Should be 0.

  let z_vals = vec![x, y, u];
  let E_vals = vec![e];

  let mut cs = TestConstraintSystem::<Scalar>::new();

  let z_f = alloc_vec(&mut cs, "z", &z_vals);
  let u_f = AllocatedNum::alloc(cs.namespace(|| "u_f"), || Ok(u)).unwrap();
  let E_f = alloc_vec(&mut cs, "E", &E_vals);

  cs.enforce(
    || "u_eq",
    |lc| lc + u_f.get_variable() - z_f[2].get_variable(),
    |lc| lc + TestConstraintSystem::<Scalar>::one(),
    |lc| lc,
  );

  enforce_relaxed_r1cs_sat::<E, _>(
    cs.namespace(|| "relaxed_sat"),
    &shape,
    &z_f,
    &u_f,
    &E_f,
  )
  .unwrap();

  assert!(
    !cs.is_satisfied(),
    "Relaxed R1CS sat with bad E should NOT be satisfied"
  );
}
