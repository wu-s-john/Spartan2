// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Preprocessing Spartan for the plain (non-zk) split-witness flow.

use crate::{
  Blind, Commitment, CommitmentKey, MULTIROUND_COMMITMENT_WIDTH, VerifierKey,
  bellpepper::{
    r1cs::{SpartanShape, SpartanWitness},
    shape_cs::ShapeCS,
    solver::SatisfyingAssignment,
  },
  digest::{DigestComputer, SimpleDigestible},
  errors::SpartanError,
  math::Math,
  polys::{
    eq::EqPolynomial,
    multilinear::{MultilinearPolynomial, SparsePolynomial},
    univariate::UniPoly,
  },
  r1cs::{SplitR1CSInstance, SplitR1CSShape},
  spartan::SpartanPrepSNARK,
  start_span,
  sumcheck::SumcheckProof,
  traits::{
    Engine,
    circuit::SpartanCircuit,
    pcs::{FoldingEngineTrait, PCSEngineTrait},
    snark::{DigestHelperTrait, R1CSSNARKTrait, SpartanDigest},
    transcript::TranscriptEngineTrait,
  },
};
use ff::Field;
use once_cell::sync::OnceCell;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::info;

fn powers<E: Engine>(s: &E::Scalar, n: usize) -> Vec<E::Scalar> {
  assert!(n >= 1);
  let mut out = Vec::with_capacity(n);
  out.push(E::Scalar::ONE);
  for i in 1..n {
    out.push(out[i - 1] * s);
  }
  out
}

fn padded<E: Engine>(v: &[E::Scalar], n: usize) -> Vec<E::Scalar> {
  let mut out = vec![E::Scalar::ZERO; n];
  out[..v.len()].copy_from_slice(v);
  out
}

fn eval_dense<E: Engine>(poly: &[E::Scalar], point: &[E::Scalar]) -> E::Scalar {
  EqPolynomial::evals_from_points(point)
    .into_par_iter()
    .zip(poly.par_iter().copied())
    .map(|(chi, val)| chi * val)
    .sum()
}

fn eval_dense_many<E: Engine>(polys: &[&[E::Scalar]], point: &[E::Scalar]) -> Vec<E::Scalar> {
  let chis = EqPolynomial::evals_from_points(point);
  polys
    .par_iter()
    .map(|poly| {
      chis
        .par_iter()
        .copied()
        .zip(poly.par_iter().copied())
        .map(|(chi, val)| chi * val)
        .sum()
    })
    .collect()
}

fn batch_invert<E: Engine>(values: &[E::Scalar]) -> Result<Vec<E::Scalar>, SpartanError> {
  if values.iter().any(|v| *v == E::Scalar::ZERO) {
    return Err(SpartanError::DivisionByZero);
  }

  let mut prefix = Vec::with_capacity(values.len());
  let mut acc = E::Scalar::ONE;
  for value in values {
    prefix.push(acc);
    acc *= value;
  }

  let acc_inv = Option::<E::Scalar>::from(acc.invert()).ok_or(SpartanError::DivisionByZero)?;
  let mut suffix = acc_inv;
  let mut out = vec![E::Scalar::ZERO; values.len()];
  for i in (0..values.len()).rev() {
    out[i] = suffix * prefix[i];
    suffix *= values[i];
  }

  Ok(out)
}

fn dense_masked_eq_evals<E: Engine>(tau: &[E::Scalar], num_masked_vars: usize) -> Vec<E::Scalar> {
  let mut evals = EqPolynomial::evals_from_points(tau);
  let num_live = 1usize << num_masked_vars;
  evals[..num_live].fill(E::Scalar::ZERO);
  evals
}

fn masked_eq_eval<E: Engine>(
  tau: &[E::Scalar],
  num_masked_vars: usize,
  point: &[E::Scalar],
) -> E::Scalar {
  let split_idx = tau.len() - num_masked_vars;
  let (tau_lo, tau_hi) = tau.split_at(split_idx);
  let (point_lo, point_hi) = point.split_at(split_idx);

  let eq_lo = tau_lo
    .iter()
    .zip(point_lo.iter())
    .map(|(r_i, x_i)| *r_i * *x_i + (E::Scalar::ONE - *r_i) * (E::Scalar::ONE - *x_i))
    .product::<E::Scalar>();
  let eq_hi = tau_hi
    .iter()
    .zip(point_hi.iter())
    .map(|(r_i, x_i)| *r_i * *x_i + (E::Scalar::ONE - *r_i) * (E::Scalar::ONE - *x_i))
    .product::<E::Scalar>();
  let mask_lo = tau_lo
    .iter()
    .zip(point_lo.iter())
    .map(|(r_i, x_i)| (E::Scalar::ONE - *r_i) * (E::Scalar::ONE - *x_i))
    .product::<E::Scalar>();

  (eq_lo - mask_lo) * eq_hi
}

fn identity_eval<E: Engine>(num_vars: usize, point: &[E::Scalar]) -> E::Scalar {
  assert_eq!(num_vars, point.len());
  point
    .iter()
    .enumerate()
    .fold(E::Scalar::ZERO, |acc, (i, r_i)| {
      acc + E::Scalar::from((1usize << (num_vars - 1 - i)) as u64) * *r_i
    })
}

fn folded_poly<E: Engine>(polys: &[&[E::Scalar]], weights: &[E::Scalar]) -> Vec<E::Scalar> {
  assert_eq!(polys.len(), weights.len());
  let n = polys[0].len();
  assert!(polys.iter().all(|poly| poly.len() == n));
  (0..n)
    .into_par_iter()
    .map(|i| {
      polys
        .iter()
        .zip(weights.iter())
        .map(|(poly, w)| poly[i] * *w)
        .sum()
    })
    .collect()
}

fn extract_suffix_point<E: Engine>(point: &[E::Scalar], suffix_num_vars: usize) -> Vec<E::Scalar> {
  point[point.len() - suffix_num_vars..].to_vec()
}

fn prefix_zero_factor<E: Engine>(point: &[E::Scalar], suffix_num_vars: usize) -> E::Scalar {
  point[..point.len() - suffix_num_vars]
    .iter()
    .fold(E::Scalar::ONE, |acc, r_i| acc * (E::Scalar::ONE - r_i))
}

fn eval_tail_half<E: Engine>(
  public_values: &[E::Scalar],
  challenges: &[E::Scalar],
  r_tail: &[E::Scalar],
) -> E::Scalar {
  let tail = std::iter::once(E::Scalar::ONE)
    .chain(public_values.iter().copied())
    .chain(challenges.iter().copied())
    .collect::<Vec<_>>();
  SparsePolynomial::new(r_tail.len(), tail).evaluate(r_tail)
}

fn eval_z_full_from_base_witness<E: Engine>(
  eval_W: E::Scalar,
  public_values: &[E::Scalar],
  challenges: &[E::Scalar],
  point: &[E::Scalar],
  num_vars: usize,
) -> E::Scalar {
  let num_vars_z = num_vars.log_2() + 1;
  let factor = prefix_zero_factor::<E>(point, num_vars_z);
  let r_z = &point[point.len() - num_vars_z..];
  let eval_tail = eval_tail_half::<E>(public_values, challenges, &r_z[1..]);
  factor * ((E::Scalar::ONE - r_z[0]) * eval_W + r_z[0] * eval_tail)
}

fn eval_witness_ext_from_base<E: Engine>(
  eval_W: E::Scalar,
  point: &[E::Scalar],
  num_vars: usize,
) -> E::Scalar {
  prefix_zero_factor::<E>(point, num_vars.log_2()) * eval_W
}

fn timed_commit_poly<E: Engine>(
  ck: &CommitmentKey<E>,
  label: &'static str,
  poly: &[E::Scalar],
  blind: &Blind<E>,
) -> Result<Commitment<E>, SpartanError> {
  let (_commit_span, commit_t) =
    start_span!("pp_preprocess_commit", commit = label, len = poly.len());
  let comm = E::PCS::commit(ck, poly, blind, false)?;
  info!(
    commit = label,
    len = poly.len(),
    elapsed_ms = %commit_t.elapsed().as_millis(),
    "pp_spartan_preprocess_commit"
  );
  Ok(comm)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
struct SplitR1CSShapeSparkRepr<E: Engine> {
  N: usize,
  row_addr: Vec<usize>,
  col_addr: Vec<usize>,
  row: Vec<E::Scalar>,
  col: Vec<E::Scalar>,
  val_A: Vec<E::Scalar>,
  val_B: Vec<E::Scalar>,
  val_C: Vec<E::Scalar>,
  ts_row: Vec<E::Scalar>,
  ts_col: Vec<E::Scalar>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
struct SplitR1CSShapeSparkBlinds<E: Engine> {
  row: Blind<E>,
  col: Blind<E>,
  val_A: Blind<E>,
  val_B: Blind<E>,
  val_C: Blind<E>,
  ts_row: Blind<E>,
  ts_col: Blind<E>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
struct SplitR1CSShapeSparkCommitment<E: Engine> {
  N: usize,
  comm_row: Commitment<E>,
  comm_col: Commitment<E>,
  comm_val_A: Commitment<E>,
  comm_val_B: Commitment<E>,
  comm_val_C: Commitment<E>,
  comm_ts_row: Commitment<E>,
  comm_ts_col: Commitment<E>,
}

impl<E: Engine> SplitR1CSShapeSparkRepr<E> {
  fn new(S: &SplitR1CSShape<E>) -> Self {
    let total_nz = S.A.data.len() + S.B.data.len() + S.C.data.len();
    let N = total_nz
      .max((2 * (S.num_shared + S.num_precommitted + S.num_rest)).max(S.num_cons))
      .next_power_of_two();

    let mut row_addr = vec![0usize; N];
    let mut col_addr = vec![N - 1; N];
    let mut val_A = vec![E::Scalar::ZERO; N];
    let mut val_B = vec![E::Scalar::ZERO; N];
    let mut val_C = vec![E::Scalar::ZERO; N];

    let mut offset = 0usize;
    for (row, col, val) in S.A.iter() {
      row_addr[offset] = row;
      col_addr[offset] = col;
      val_A[offset] = val;
      offset += 1;
    }
    for (row, col, val) in S.B.iter() {
      row_addr[offset] = row;
      col_addr[offset] = col;
      val_B[offset] = val;
      offset += 1;
    }
    for (row, col, val) in S.C.iter() {
      row_addr[offset] = row;
      col_addr[offset] = col;
      val_C[offset] = val;
      offset += 1;
    }

    let timestamp_calc = |trace: &[usize]| -> Vec<E::Scalar> {
      let mut counts = vec![0usize; N];
      for addr in trace {
        counts[*addr] += 1;
      }
      counts
        .into_iter()
        .map(|count| E::Scalar::from(count as u64))
        .collect()
    };

    Self {
      N,
      row: row_addr
        .iter()
        .map(|addr| E::Scalar::from(*addr as u64))
        .collect(),
      col: col_addr
        .iter()
        .map(|addr| E::Scalar::from(*addr as u64))
        .collect(),
      row_addr: row_addr.clone(),
      col_addr: col_addr.clone(),
      val_A,
      val_B,
      val_C,
      ts_row: timestamp_calc(&row_addr),
      ts_col: timestamp_calc(&col_addr),
    }
  }

  fn commit(
    &self,
    ck: &CommitmentKey<E>,
  ) -> Result<
    (
      SplitR1CSShapeSparkCommitment<E>,
      SplitR1CSShapeSparkBlinds<E>,
    ),
    SpartanError,
  > {
    let blinds = SplitR1CSShapeSparkBlinds {
      row: E::PCS::blind(ck, self.N),
      col: E::PCS::blind(ck, self.N),
      val_A: E::PCS::blind(ck, self.N),
      val_B: E::PCS::blind(ck, self.N),
      val_C: E::PCS::blind(ck, self.N),
      ts_row: E::PCS::blind(ck, self.N),
      ts_col: E::PCS::blind(ck, self.N),
    };

    let comms = SplitR1CSShapeSparkCommitment {
      N: self.N,
      comm_row: timed_commit_poly::<E>(ck, "pp_commit_row", &self.row, &blinds.row)?,
      comm_col: timed_commit_poly::<E>(ck, "pp_commit_col", &self.col, &blinds.col)?,
      comm_val_A: timed_commit_poly::<E>(ck, "pp_commit_val_A", &self.val_A, &blinds.val_A)?,
      comm_val_B: timed_commit_poly::<E>(ck, "pp_commit_val_B", &self.val_B, &blinds.val_B)?,
      comm_val_C: timed_commit_poly::<E>(ck, "pp_commit_val_C", &self.val_C, &blinds.val_C)?,
      comm_ts_row: timed_commit_poly::<E>(ck, "pp_commit_ts_row", &self.ts_row, &blinds.ts_row)?,
      comm_ts_col: timed_commit_poly::<E>(ck, "pp_commit_ts_col", &self.ts_col, &blinds.ts_col)?,
    };

    Ok((comms, blinds))
  }

  fn evaluation_oracles(
    &self,
    r_outer_full: &[E::Scalar],
    z_full: &[E::Scalar],
  ) -> (
    Vec<E::Scalar>,
    Vec<E::Scalar>,
    Vec<E::Scalar>,
    Vec<E::Scalar>,
  ) {
    let mem_row = EqPolynomial::evals_from_points(r_outer_full);
    let mem_col = padded::<E>(z_full, self.N);

    let mut L_row = vec![mem_row[0]; self.N];
    let mut L_col = vec![mem_col[self.N - 1]; self.N];
    for i in 0..self.N {
      L_row[i] = mem_row[self.row_addr[i]];
      L_col[i] = mem_col[self.col_addr[i]];
    }

    (mem_row, mem_col, L_row, L_col)
  }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct PpSpartanProverKey<E: Engine> {
  ck: CommitmentKey<E>,
  ck_s: CommitmentKey<E>,
  S: SplitR1CSShape<E>,
  S_repr: SplitR1CSShapeSparkRepr<E>,
  S_comm: SplitR1CSShapeSparkCommitment<E>,
  S_blinds: SplitR1CSShapeSparkBlinds<E>,
  vk_digest: SpartanDigest,
}

impl<E: Engine> PpSpartanProverKey<E> {
  pub fn sizes(&self) -> [usize; 10] {
    self.S.sizes()
  }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct PpSpartanVerifierKey<E: Engine> {
  vk_ee: VerifierKey<E>,
  ck_s: CommitmentKey<E>,
  S: SplitR1CSShape<E>,
  S_comm: SplitR1CSShapeSparkCommitment<E>,
  #[serde(skip, default = "OnceCell::new")]
  digest: OnceCell<SpartanDigest>,
}

impl<E: Engine> SimpleDigestible for PpSpartanVerifierKey<E> {}

impl<E: Engine> DigestHelperTrait<E> for PpSpartanVerifierKey<E> {
  fn digest(&self) -> Result<SpartanDigest, SpartanError> {
    self
      .digest
      .get_or_try_init(|| DigestComputer::<_>::new(self).digest())
      .cloned()
      .map_err(|_| SpartanError::DigestError {
        reason: "Unable to compute digest for PpSpartanVerifierKey".to_string(),
      })
  }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct PpSpartanSNARK<E: Engine> {
  U: SplitR1CSInstance<E>,

  comm_L_row: Commitment<E>,
  comm_L_col: Commitment<E>,
  comm_t_plus_r_inv_row: Commitment<E>,
  comm_w_plus_r_inv_row: Commitment<E>,
  comm_t_plus_r_inv_col: Commitment<E>,
  comm_w_plus_r_inv_col: Commitment<E>,

  sc_proof_outer: SumcheckProof<E>,
  claims_outer: (E::Scalar, E::Scalar, E::Scalar),

  sc_proof_inner: SumcheckProof<E>,

  eval_W: E::Scalar,
  blind_eval_W: Blind<E>,
  eval_arg_W: <E::PCS as PCSEngineTrait<E>>::EvaluationArgument,

  eval_L_row: E::Scalar,
  eval_L_col: E::Scalar,
  eval_val_A: E::Scalar,
  eval_val_B: E::Scalar,
  eval_val_C: E::Scalar,
  eval_row: E::Scalar,
  eval_col: E::Scalar,
  eval_ts_row: E::Scalar,
  eval_ts_col: E::Scalar,
  eval_t_plus_r_inv_row: E::Scalar,
  eval_w_plus_r_inv_row: E::Scalar,
  eval_t_plus_r_inv_col: E::Scalar,
  eval_w_plus_r_inv_col: E::Scalar,

  blind_eval_joint: Blind<E>,
  eval_arg_joint: <E::PCS as PCSEngineTrait<E>>::EvaluationArgument,
}

struct MemoryOracles<E: Engine> {
  t_plus_r_row: Vec<E::Scalar>,
  w_plus_r_row: Vec<E::Scalar>,
  t_plus_r_col: Vec<E::Scalar>,
  w_plus_r_col: Vec<E::Scalar>,
  t_plus_r_inv_row: Vec<E::Scalar>,
  w_plus_r_inv_row: Vec<E::Scalar>,
  t_plus_r_inv_col: Vec<E::Scalar>,
  w_plus_r_inv_col: Vec<E::Scalar>,
}

struct InnerSumcheckState<E: Engine> {
  poly_t_plus_r_inv_row: MultilinearPolynomial<E::Scalar>,
  poly_w_plus_r_inv_row: MultilinearPolynomial<E::Scalar>,
  poly_t_plus_r_row: MultilinearPolynomial<E::Scalar>,
  poly_w_plus_r_row: MultilinearPolynomial<E::Scalar>,
  poly_ts_row: MultilinearPolynomial<E::Scalar>,
  poly_t_plus_r_inv_col: MultilinearPolynomial<E::Scalar>,
  poly_w_plus_r_inv_col: MultilinearPolynomial<E::Scalar>,
  poly_t_plus_r_col: MultilinearPolynomial<E::Scalar>,
  poly_w_plus_r_col: MultilinearPolynomial<E::Scalar>,
  poly_ts_col: MultilinearPolynomial<E::Scalar>,
  poly_eq_rho: MultilinearPolynomial<E::Scalar>,
  poly_L_row: MultilinearPolynomial<E::Scalar>,
  poly_L_col: MultilinearPolynomial<E::Scalar>,
  poly_val: MultilinearPolynomial<E::Scalar>,
  poly_masked_eq: MultilinearPolynomial<E::Scalar>,
  poly_W_ext: MultilinearPolynomial<E::Scalar>,
}

impl<E: Engine> InnerSumcheckState<E> {
  fn size(&self) -> usize {
    self.poly_L_row.Z.len()
  }

  fn bind(&mut self, r: &E::Scalar) {
    [
      &mut self.poly_t_plus_r_inv_row,
      &mut self.poly_w_plus_r_inv_row,
      &mut self.poly_t_plus_r_row,
      &mut self.poly_w_plus_r_row,
      &mut self.poly_ts_row,
      &mut self.poly_t_plus_r_inv_col,
      &mut self.poly_w_plus_r_inv_col,
      &mut self.poly_t_plus_r_col,
      &mut self.poly_w_plus_r_col,
      &mut self.poly_ts_col,
      &mut self.poly_eq_rho,
      &mut self.poly_L_row,
      &mut self.poly_L_col,
      &mut self.poly_val,
      &mut self.poly_masked_eq,
      &mut self.poly_W_ext,
    ]
    .into_iter()
    .for_each(|poly| poly.bind_poly_var_top(r));
  }
}

fn linear_eval_points<E: Engine>(
  poly_a: &[E::Scalar],
  poly_b: &[E::Scalar],
) -> (E::Scalar, E::Scalar, E::Scalar) {
  let half = poly_a.len() / 2;
  (0..half)
    .into_par_iter()
    .map(|i| {
      let a0 = poly_a[i];
      let a1 = poly_a[i + half];
      let b0 = poly_b[i];
      let b1 = poly_b[i + half];
      let da = a1 - a0;
      let db = b1 - b0;
      (a0 - b0, E::Scalar::ZERO, (a0 - da) - (b0 - db))
    })
    .reduce(
      || (E::Scalar::ZERO, E::Scalar::ZERO, E::Scalar::ZERO),
      |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
    )
}

fn cubic_eval_points_with_eq_sub<E: Engine>(
  eq_poly: &[E::Scalar],
  poly_a: &[E::Scalar],
  poly_b: &[E::Scalar],
  poly_c: &[E::Scalar],
) -> (E::Scalar, E::Scalar, E::Scalar) {
  let half = eq_poly.len() / 2;
  (0..half)
    .into_par_iter()
    .map(|i| {
      let e0 = eq_poly[i];
      let e1 = eq_poly[i + half];
      let a0 = poly_a[i];
      let a1 = poly_a[i + half];
      let b0 = poly_b[i];
      let b1 = poly_b[i + half];
      let c0 = poly_c[i];
      let c1 = poly_c[i + half];

      let de = e1 - e0;
      let da = a1 - a0;
      let db = b1 - b0;
      let dc = c1 - c0;

      (
        e0 * (a0 * b0 - c0),
        de * da * db,
        (e0 - de) * ((a0 - da) * (b0 - db) - (c0 - dc)),
      )
    })
    .reduce(
      || (E::Scalar::ZERO, E::Scalar::ZERO, E::Scalar::ZERO),
      |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
    )
}

fn cubic_eval_points_with_eq_sub_one<E: Engine>(
  eq_poly: &[E::Scalar],
  poly_a: &[E::Scalar],
  poly_b: &[E::Scalar],
) -> (E::Scalar, E::Scalar, E::Scalar) {
  let half = eq_poly.len() / 2;
  (0..half)
    .into_par_iter()
    .map(|i| {
      let e0 = eq_poly[i];
      let e1 = eq_poly[i + half];
      let a0 = poly_a[i];
      let a1 = poly_a[i + half];
      let b0 = poly_b[i];
      let b1 = poly_b[i + half];

      let de = e1 - e0;
      let da = a1 - a0;
      let db = b1 - b0;

      (
        e0 * (a0 * b0 - E::Scalar::ONE),
        de * da * db,
        (e0 - de) * ((a0 - da) * (b0 - db) - E::Scalar::ONE),
      )
    })
    .reduce(
      || (E::Scalar::ZERO, E::Scalar::ZERO, E::Scalar::ZERO),
      |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
    )
}

fn cubic_eval_points_three_prod<E: Engine>(
  poly_a: &[E::Scalar],
  poly_b: &[E::Scalar],
  poly_c: &[E::Scalar],
) -> (E::Scalar, E::Scalar, E::Scalar) {
  let half = poly_a.len() / 2;
  (0..half)
    .into_par_iter()
    .map(|i| {
      let a0 = poly_a[i];
      let a1 = poly_a[i + half];
      let b0 = poly_b[i];
      let b1 = poly_b[i + half];
      let c0 = poly_c[i];
      let c1 = poly_c[i + half];

      let da = a1 - a0;
      let db = b1 - b0;
      let dc = c1 - c0;

      (
        a0 * b0 * c0,
        da * db * dc,
        (a0 - da) * (b0 - db) * (c0 - dc),
      )
    })
    .reduce(
      || (E::Scalar::ZERO, E::Scalar::ZERO, E::Scalar::ZERO),
      |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
    )
}

fn quadratic_eval_points<E: Engine>(
  poly_a: &[E::Scalar],
  poly_b: &[E::Scalar],
) -> (E::Scalar, E::Scalar, E::Scalar) {
  let half = poly_a.len() / 2;
  (0..half)
    .into_par_iter()
    .map(|i| {
      let a0 = poly_a[i];
      let a1 = poly_a[i + half];
      let b0 = poly_b[i];
      let b1 = poly_b[i + half];

      let da = a1 - a0;
      let db = b1 - b0;

      (a0 * b0, E::Scalar::ZERO, (a0 - da) * (b0 - db))
    })
    .reduce(
      || (E::Scalar::ZERO, E::Scalar::ZERO, E::Scalar::ZERO),
      |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
    )
}

fn compute_memory_oracles<E: Engine>(
  repr: &SplitR1CSShapeSparkRepr<E>,
  mem_row: &[E::Scalar],
  mem_col: &[E::Scalar],
  L_row: &[E::Scalar],
  L_col: &[E::Scalar],
  gamma: E::Scalar,
  r_mem: E::Scalar,
) -> Result<MemoryOracles<E>, SpartanError> {
  let T_row = (0..repr.N)
    .map(|i| E::Scalar::from(i as u64) + gamma * mem_row[i])
    .collect::<Vec<_>>();
  let T_col = (0..repr.N)
    .map(|i| E::Scalar::from(i as u64) + gamma * mem_col[i])
    .collect::<Vec<_>>();
  let W_row = (0..repr.N)
    .map(|i| repr.row[i] + gamma * L_row[i])
    .collect::<Vec<_>>();
  let W_col = (0..repr.N)
    .map(|i| repr.col[i] + gamma * L_col[i])
    .collect::<Vec<_>>();

  let denoms = T_row
    .iter()
    .chain(W_row.iter())
    .chain(T_col.iter())
    .chain(W_col.iter())
    .map(|v| *v + r_mem)
    .collect::<Vec<_>>();
  let invs = batch_invert::<E>(&denoms)?;

  let n = repr.N;
  let t_plus_r_inv_row_base = &invs[..n];
  let w_plus_r_inv_row = invs[n..2 * n].to_vec();
  let t_plus_r_inv_col_base = &invs[2 * n..3 * n];
  let w_plus_r_inv_col = invs[3 * n..].to_vec();

  let t_plus_r_inv_row = t_plus_r_inv_row_base
    .iter()
    .zip(repr.ts_row.iter())
    .map(|(inv, ts)| *inv * *ts)
    .collect::<Vec<_>>();
  let t_plus_r_inv_col = t_plus_r_inv_col_base
    .iter()
    .zip(repr.ts_col.iter())
    .map(|(inv, ts)| *inv * *ts)
    .collect::<Vec<_>>();

  Ok(MemoryOracles {
    t_plus_r_row: T_row.into_iter().map(|v| v + r_mem).collect(),
    w_plus_r_row: W_row.into_iter().map(|v| v + r_mem).collect(),
    t_plus_r_col: T_col.into_iter().map(|v| v + r_mem).collect(),
    w_plus_r_col: W_col.into_iter().map(|v| v + r_mem).collect(),
    t_plus_r_inv_row,
    w_plus_r_inv_row,
    t_plus_r_inv_col,
    w_plus_r_inv_col,
  })
}

fn prove_batched_inner<E: Engine>(
  state: &mut InnerSumcheckState<E>,
  initial_claim: E::Scalar,
  weights: &[E::Scalar],
  transcript: &mut E::TE,
) -> Result<(SumcheckProof<E>, Vec<E::Scalar>, E::Scalar), SpartanError> {
  let num_rounds = state.size().log_2();
  let mut running_claim = initial_claim;
  let mut r_inner = Vec::with_capacity(num_rounds);
  let mut polys = Vec::with_capacity(num_rounds);

  for _round in 0..num_rounds {
    let evals = [
      linear_eval_points::<E>(
        &state.poly_t_plus_r_inv_row.Z,
        &state.poly_w_plus_r_inv_row.Z,
      ),
      linear_eval_points::<E>(
        &state.poly_t_plus_r_inv_col.Z,
        &state.poly_w_plus_r_inv_col.Z,
      ),
      cubic_eval_points_with_eq_sub::<E>(
        &state.poly_eq_rho.Z,
        &state.poly_t_plus_r_inv_row.Z,
        &state.poly_t_plus_r_row.Z,
        &state.poly_ts_row.Z,
      ),
      cubic_eval_points_with_eq_sub_one::<E>(
        &state.poly_eq_rho.Z,
        &state.poly_w_plus_r_inv_row.Z,
        &state.poly_w_plus_r_row.Z,
      ),
      cubic_eval_points_with_eq_sub::<E>(
        &state.poly_eq_rho.Z,
        &state.poly_t_plus_r_inv_col.Z,
        &state.poly_t_plus_r_col.Z,
        &state.poly_ts_col.Z,
      ),
      cubic_eval_points_with_eq_sub_one::<E>(
        &state.poly_eq_rho.Z,
        &state.poly_w_plus_r_inv_col.Z,
        &state.poly_w_plus_r_col.Z,
      ),
      cubic_eval_points_three_prod::<E>(
        &state.poly_L_row.Z,
        &state.poly_L_col.Z,
        &state.poly_val.Z,
      ),
      quadratic_eval_points::<E>(&state.poly_masked_eq.Z, &state.poly_W_ext.Z),
    ];

    let eval_0: E::Scalar = evals
      .iter()
      .zip(weights.iter())
      .map(|(ev, w)| ev.0 * *w)
      .sum();
    let leading: E::Scalar = evals
      .iter()
      .zip(weights.iter())
      .map(|(ev, w)| ev.1 * *w)
      .sum();
    let eval_neg1: E::Scalar = evals
      .iter()
      .zip(weights.iter())
      .map(|(ev, w)| ev.2 * *w)
      .sum();

    let poly = UniPoly::from_evals_deg3(eval_0, leading, eval_neg1, running_claim);
    transcript.absorb(b"p", &poly);
    let r_i = transcript.squeeze(b"c")?;
    running_claim = poly.evaluate(&r_i);
    state.bind(&r_i);
    let recomputed_claim = weights[0]
      * state
        .poly_t_plus_r_inv_row
        .Z
        .iter()
        .zip(state.poly_w_plus_r_inv_row.Z.iter())
        .map(|(a, b)| *a - *b)
        .sum::<E::Scalar>()
      + weights[1]
        * state
          .poly_t_plus_r_inv_col
          .Z
          .iter()
          .zip(state.poly_w_plus_r_inv_col.Z.iter())
          .map(|(a, b)| *a - *b)
          .sum::<E::Scalar>()
      + weights[2]
        * state
          .poly_eq_rho
          .Z
          .iter()
          .zip(state.poly_t_plus_r_inv_row.Z.iter())
          .zip(state.poly_t_plus_r_row.Z.iter())
          .zip(state.poly_ts_row.Z.iter())
          .map(|(((eq_i, inv_i), t_i), ts_i)| *eq_i * (*inv_i * *t_i - *ts_i))
          .sum::<E::Scalar>()
      + weights[3]
        * state
          .poly_eq_rho
          .Z
          .iter()
          .zip(state.poly_w_plus_r_inv_row.Z.iter())
          .zip(state.poly_w_plus_r_row.Z.iter())
          .map(|((eq_i, inv_i), t_i)| *eq_i * (*inv_i * *t_i - E::Scalar::ONE))
          .sum::<E::Scalar>()
      + weights[4]
        * state
          .poly_eq_rho
          .Z
          .iter()
          .zip(state.poly_t_plus_r_inv_col.Z.iter())
          .zip(state.poly_t_plus_r_col.Z.iter())
          .zip(state.poly_ts_col.Z.iter())
          .map(|(((eq_i, inv_i), t_i), ts_i)| *eq_i * (*inv_i * *t_i - *ts_i))
          .sum::<E::Scalar>()
      + weights[5]
        * state
          .poly_eq_rho
          .Z
          .iter()
          .zip(state.poly_w_plus_r_inv_col.Z.iter())
          .zip(state.poly_w_plus_r_col.Z.iter())
          .map(|((eq_i, inv_i), t_i)| *eq_i * (*inv_i * *t_i - E::Scalar::ONE))
          .sum::<E::Scalar>()
      + weights[6]
        * state
          .poly_L_row
          .Z
          .iter()
          .zip(state.poly_L_col.Z.iter())
          .zip(state.poly_val.Z.iter())
          .map(|((a, b), c)| *a * *b * *c)
          .sum::<E::Scalar>()
      + weights[7]
        * state
          .poly_masked_eq
          .Z
          .iter()
          .zip(state.poly_W_ext.Z.iter())
          .map(|(a, b)| *a * *b)
          .sum::<E::Scalar>();
    debug_assert_eq!(running_claim, recomputed_claim);
    r_inner.push(r_i);
    polys.push(poly.compress());
  }

  Ok((SumcheckProof::new(polys), r_inner, running_claim))
}

impl<E: Engine> PpSpartanSNARK<E> {
  fn prove_transcript_setup<C: SpartanCircuit<E>>(
    pk: &PpSpartanProverKey<E>,
    circuit: &C,
  ) -> Result<E::TE, SpartanError> {
    let mut transcript = E::TE::new(b"PpSpartanSNARK");
    transcript.absorb(b"vk", &pk.vk_digest);
    let public_values = circuit
      .public_values()
      .map_err(|e| SpartanError::SynthesisError {
        reason: format!("Circuit does not provide public IO: {e}"),
      })?;
    transcript.absorb(b"public_values", &public_values.as_slice());
    Ok(transcript)
  }

  fn batch_open_same_point(
    ck: &CommitmentKey<E>,
    ck_s: &CommitmentKey<E>,
    transcript: &mut E::TE,
    point: &[E::Scalar],
    comms: &[Commitment<E>],
    blinds: &[Blind<E>],
    polys: &[&[E::Scalar]],
    evals: &[E::Scalar],
  ) -> Result<(Blind<E>, <E::PCS as PCSEngineTrait<E>>::EvaluationArgument), SpartanError>
  where
    E::PCS: FoldingEngineTrait<E>,
  {
    transcript.absorb(b"joint_evals", &evals);
    let beta = transcript.squeeze(b"joint_beta")?;
    let weights = powers::<E>(&beta, evals.len());
    let folded_comm = <E::PCS as FoldingEngineTrait<E>>::fold_commitments(comms, &weights)?;
    let folded_blind = <E::PCS as FoldingEngineTrait<E>>::fold_blinds(blinds, &weights)?;
    let folded_poly = folded_poly::<E>(polys, &weights);
    let folded_eval: E::Scalar = evals
      .iter()
      .zip(weights.iter())
      .map(|(eval, weight)| *eval * *weight)
      .sum();
    let blind_eval = E::PCS::blind(ck_s, 1);
    let comm_eval = E::PCS::commit(ck_s, &[folded_eval], &blind_eval, false)?;
    let eval_arg = E::PCS::prove(
      ck,
      ck_s,
      transcript,
      &folded_comm,
      &folded_poly,
      &folded_blind,
      point,
      &comm_eval,
      &blind_eval,
    )?;

    Ok((blind_eval, eval_arg))
  }

  fn verify_same_point_batch(
    vk: &PpSpartanVerifierKey<E>,
    transcript: &mut E::TE,
    point: &[E::Scalar],
    comms: &[Commitment<E>],
    evals: &[E::Scalar],
    blind_eval: &Blind<E>,
    eval_arg: &<E::PCS as PCSEngineTrait<E>>::EvaluationArgument,
  ) -> Result<(), SpartanError>
  where
    E::PCS: FoldingEngineTrait<E>,
  {
    transcript.absorb(b"joint_evals", &evals);
    let beta = transcript.squeeze(b"joint_beta")?;
    let weights = powers::<E>(&beta, evals.len());
    let folded_comm = <E::PCS as FoldingEngineTrait<E>>::fold_commitments(comms, &weights)?;
    let folded_eval: E::Scalar = evals
      .iter()
      .zip(weights.iter())
      .map(|(eval, weight)| *eval * *weight)
      .sum();
    let comm_eval = E::PCS::commit(&vk.ck_s, &[folded_eval], blind_eval, false)?;
    E::PCS::verify(
      &vk.vk_ee,
      &vk.ck_s,
      transcript,
      &folded_comm,
      point,
      &comm_eval,
      eval_arg,
    )
  }

  fn prove_regular<C: SpartanCircuit<E>>(
    pk: &PpSpartanProverKey<E>,
    circuit: C,
    prep_snark: &SpartanPrepSNARK<E>,
    is_small: bool,
  ) -> Result<Self, SpartanError>
  where
    E::PCS: FoldingEngineTrait<E>,
  {
    let (_prove_span, prove_t) = start_span!("pp_spartan_snark_prove");
    let mut prep_snark = prep_snark.clone();
    let mut transcript = Self::prove_transcript_setup(pk, &circuit)?;

    let (_sat_span, sat_t) = start_span!("r1cs_instance_and_witness");
    let (U, W) = SatisfyingAssignment::r1cs_instance_and_witness(
      &mut prep_snark,
      &pk.S,
      &pk.ck,
      &circuit,
      is_small,
      &mut transcript,
    )?;
    info!(elapsed_ms = %sat_t.elapsed().as_millis(), "r1cs_instance_and_witness");

    let z = [
      W.W.clone(),
      vec![E::Scalar::ONE],
      U.public_values.clone(),
      U.challenges.clone(),
    ]
    .concat();
    let num_vars = pk.S.num_shared + pk.S.num_precommitted + pk.S.num_rest;
    let num_rounds_x = usize::try_from(pk.S.num_cons.ilog2()).expect("constraint count log2 fits");
    let num_rounds_w = num_vars.log_2();
    let tau = (0..num_rounds_x)
      .map(|_| transcript.squeeze(b"t"))
      .collect::<Result<Vec<_>, SpartanError>>()?;

    let (_mv_span, mv_t) = start_span!("matrix_vector_multiply");
    let (Az, Bz, Cz) = pk.S.multiply_vec(&z)?;
    info!(elapsed_ms = %mv_t.elapsed().as_millis(), "matrix_vector_multiply");

    let (_outer_span, outer_t) = start_span!("outer_sumcheck");
    let (sc_proof_outer, r_x, claims_outer_vec) = SumcheckProof::prove_cubic_with_three_inputs(
      &E::Scalar::ZERO,
      tau,
      &mut MultilinearPolynomial::new(Az),
      &mut MultilinearPolynomial::new(Bz),
      &mut MultilinearPolynomial::new(Cz),
      &mut transcript,
    )?;
    let claims_outer = (
      claims_outer_vec[0],
      claims_outer_vec[1],
      claims_outer_vec[2],
    );
    transcript.absorb(
      b"claims_outer",
      &[claims_outer.0, claims_outer.1, claims_outer.2].as_slice(),
    );
    info!(elapsed_ms = %outer_t.elapsed().as_millis(), "outer_sumcheck");

    let num_rounds_inner = pk.S_repr.N.log_2();
    let num_pad_rounds = num_rounds_inner - num_rounds_x;
    let r_pad = (0..num_pad_rounds)
      .map(|_| transcript.squeeze(b"r_pad"))
      .collect::<Result<Vec<_>, SpartanError>>()?;
    let r_x_full = r_pad
      .iter()
      .chain(r_x.iter())
      .copied()
      .collect::<Vec<E::Scalar>>();
    let factor: E::Scalar = r_pad
      .iter()
      .fold(E::Scalar::ONE, |acc, r_i| acc * (E::Scalar::ONE - r_i));

    let z_full = {
      let mut z_full = Vec::with_capacity(2 * num_vars);
      z_full.extend_from_slice(&W.W);
      z_full.push(E::Scalar::ONE);
      z_full.extend_from_slice(&U.public_values);
      z_full.extend_from_slice(&U.challenges);
      z_full.resize(2 * num_vars, E::Scalar::ZERO);
      z_full
    };
    let W_ext = padded::<E>(&W.W, pk.S_repr.N);

    let (_lookup_span, lookup_t) = start_span!("lookup_construction");
    let (mem_row, mem_col, L_row, L_col) = pk.S_repr.evaluation_oracles(&r_x_full, &z_full);
    let blind_L_row = E::PCS::blind(&pk.ck, pk.S_repr.N);
    let blind_L_col = E::PCS::blind(&pk.ck, pk.S_repr.N);
    let comm_L_row = E::PCS::commit(&pk.ck, &L_row, &blind_L_row, false)?;
    let comm_L_col = E::PCS::commit(&pk.ck, &L_col, &blind_L_col, false)?;
    transcript.absorb(
      b"lookup_comms",
      &[comm_L_row.clone(), comm_L_col.clone()].as_slice(),
    );
    info!(elapsed_ms = %lookup_t.elapsed().as_millis(), "lookup_construction");

    let c = transcript.squeeze(b"inner_c")?;
    let gamma = transcript.squeeze(b"mem_gamma")?;
    let r_mem = transcript.squeeze(b"mem_r")?;

    let (_memory_span, memory_t) = start_span!("memory_oracle_construction");
    let memory =
      compute_memory_oracles::<E>(&pk.S_repr, &mem_row, &mem_col, &L_row, &L_col, gamma, r_mem)?;
    let blind_t_plus_r_inv_row = E::PCS::blind(&pk.ck, pk.S_repr.N);
    let blind_w_plus_r_inv_row = E::PCS::blind(&pk.ck, pk.S_repr.N);
    let blind_t_plus_r_inv_col = E::PCS::blind(&pk.ck, pk.S_repr.N);
    let blind_w_plus_r_inv_col = E::PCS::blind(&pk.ck, pk.S_repr.N);
    let comm_t_plus_r_inv_row = E::PCS::commit(
      &pk.ck,
      &memory.t_plus_r_inv_row,
      &blind_t_plus_r_inv_row,
      false,
    )?;
    let comm_w_plus_r_inv_row = E::PCS::commit(
      &pk.ck,
      &memory.w_plus_r_inv_row,
      &blind_w_plus_r_inv_row,
      false,
    )?;
    let comm_t_plus_r_inv_col = E::PCS::commit(
      &pk.ck,
      &memory.t_plus_r_inv_col,
      &blind_t_plus_r_inv_col,
      false,
    )?;
    let comm_w_plus_r_inv_col = E::PCS::commit(
      &pk.ck,
      &memory.w_plus_r_inv_col,
      &blind_w_plus_r_inv_col,
      false,
    )?;
    transcript.absorb(
      b"memory_comms",
      &[
        comm_t_plus_r_inv_row.clone(),
        comm_w_plus_r_inv_row.clone(),
        comm_t_plus_r_inv_col.clone(),
        comm_w_plus_r_inv_col.clone(),
      ]
      .as_slice(),
    );
    info!(elapsed_ms = %memory_t.elapsed().as_millis(), "memory_oracle_construction");

    let rho = (0..num_rounds_inner)
      .map(|_| transcript.squeeze(b"mem_rho"))
      .collect::<Result<Vec<_>, SpartanError>>()?;
    let inner_batch_challenge = transcript.squeeze(b"inner_batch")?;
    let weights = powers::<E>(&inner_batch_challenge, 8);
    let claim_abc = factor * (claims_outer.0 + c * claims_outer.1 + c * c * claims_outer.2);
    let initial_claim = weights[6] * claim_abc;

    let val = pk
      .S_repr
      .val_A
      .iter()
      .zip(pk.S_repr.val_B.iter())
      .zip(pk.S_repr.val_C.iter())
      .map(|((v_a, v_b), v_c)| *v_a + c * *v_b + c * c * *v_c)
      .collect::<Vec<_>>();
    let mut inner_state = InnerSumcheckState {
      poly_t_plus_r_inv_row: MultilinearPolynomial::new(memory.t_plus_r_inv_row.clone()),
      poly_w_plus_r_inv_row: MultilinearPolynomial::new(memory.w_plus_r_inv_row.clone()),
      poly_t_plus_r_row: MultilinearPolynomial::new(memory.t_plus_r_row.clone()),
      poly_w_plus_r_row: MultilinearPolynomial::new(memory.w_plus_r_row.clone()),
      poly_ts_row: MultilinearPolynomial::new(pk.S_repr.ts_row.clone()),
      poly_t_plus_r_inv_col: MultilinearPolynomial::new(memory.t_plus_r_inv_col.clone()),
      poly_w_plus_r_inv_col: MultilinearPolynomial::new(memory.w_plus_r_inv_col.clone()),
      poly_t_plus_r_col: MultilinearPolynomial::new(memory.t_plus_r_col.clone()),
      poly_w_plus_r_col: MultilinearPolynomial::new(memory.w_plus_r_col.clone()),
      poly_ts_col: MultilinearPolynomial::new(pk.S_repr.ts_col.clone()),
      poly_eq_rho: MultilinearPolynomial::new(EqPolynomial::evals_from_points(&rho)),
      poly_L_row: MultilinearPolynomial::new(L_row.clone()),
      poly_L_col: MultilinearPolynomial::new(L_col.clone()),
      poly_val: MultilinearPolynomial::new(val),
      poly_masked_eq: MultilinearPolynomial::new(dense_masked_eq_evals::<E>(
        &r_x_full,
        num_rounds_w,
      )),
      poly_W_ext: MultilinearPolynomial::new(W_ext),
    };

    let (_inner_span, inner_t) = start_span!("inner_batched_sumcheck");
    let (sc_proof_inner, r_inner, claim_inner_final_prover) =
      prove_batched_inner::<E>(&mut inner_state, initial_claim, &weights, &mut transcript)?;
    info!(elapsed_ms = %inner_t.elapsed().as_millis(), "inner_batched_sumcheck");

    let claim_inner_expected_prover = weights[0]
      * (inner_state.poly_t_plus_r_inv_row[0] - inner_state.poly_w_plus_r_inv_row[0])
      + weights[1] * (inner_state.poly_t_plus_r_inv_col[0] - inner_state.poly_w_plus_r_inv_col[0])
      + weights[2]
        * (inner_state.poly_eq_rho[0]
          * (inner_state.poly_t_plus_r_inv_row[0] * inner_state.poly_t_plus_r_row[0]
            - inner_state.poly_ts_row[0]))
      + weights[3]
        * (inner_state.poly_eq_rho[0]
          * (inner_state.poly_w_plus_r_inv_row[0] * inner_state.poly_w_plus_r_row[0]
            - E::Scalar::ONE))
      + weights[4]
        * (inner_state.poly_eq_rho[0]
          * (inner_state.poly_t_plus_r_inv_col[0] * inner_state.poly_t_plus_r_col[0]
            - inner_state.poly_ts_col[0]))
      + weights[5]
        * (inner_state.poly_eq_rho[0]
          * (inner_state.poly_w_plus_r_inv_col[0] * inner_state.poly_w_plus_r_col[0]
            - E::Scalar::ONE))
      + weights[6]
        * inner_state.poly_L_row[0]
        * inner_state.poly_L_col[0]
        * inner_state.poly_val[0]
      + weights[7] * inner_state.poly_masked_eq[0] * inner_state.poly_W_ext[0];
    debug_assert_eq!(claim_inner_final_prover, claim_inner_expected_prover);

    let witness_point = extract_suffix_point::<E>(&r_inner, num_rounds_w);
    let eval_W = eval_dense::<E>(&W.W, &witness_point);
    debug_assert_eq!(
      eval_witness_ext_from_base::<E>(eval_W, &r_inner, num_vars),
      inner_state.poly_W_ext[0]
    );
    let blind_eval_W = E::PCS::blind(&pk.ck_s, 1);
    let comm_eval_W = E::PCS::commit(&pk.ck_s, &[eval_W], &blind_eval_W, false)?;
    let eval_arg_W = E::PCS::prove(
      &pk.ck,
      &pk.ck_s,
      &mut transcript,
      &U.to_regular_instance()?.comm_W,
      &W.W,
      &W.r_W,
      &witness_point,
      &comm_eval_W,
      &blind_eval_W,
    )?;

    let evals = eval_dense_many::<E>(
      &[
        &L_row,
        &L_col,
        &pk.S_repr.val_A,
        &pk.S_repr.val_B,
        &pk.S_repr.val_C,
        &pk.S_repr.row,
        &pk.S_repr.col,
        &pk.S_repr.ts_row,
        &pk.S_repr.ts_col,
        &memory.t_plus_r_inv_row,
        &memory.w_plus_r_inv_row,
        &memory.t_plus_r_inv_col,
        &memory.w_plus_r_inv_col,
      ],
      &r_inner,
    );
    debug_assert_eq!(evals[0], inner_state.poly_L_row[0]);
    debug_assert_eq!(evals[1], inner_state.poly_L_col[0]);

    let (blind_eval_joint, eval_arg_joint) = Self::batch_open_same_point(
      &pk.ck,
      &pk.ck_s,
      &mut transcript,
      &r_inner,
      &[
        comm_L_row.clone(),
        comm_L_col.clone(),
        pk.S_comm.comm_val_A.clone(),
        pk.S_comm.comm_val_B.clone(),
        pk.S_comm.comm_val_C.clone(),
        pk.S_comm.comm_row.clone(),
        pk.S_comm.comm_col.clone(),
        pk.S_comm.comm_ts_row.clone(),
        pk.S_comm.comm_ts_col.clone(),
        comm_t_plus_r_inv_row.clone(),
        comm_w_plus_r_inv_row.clone(),
        comm_t_plus_r_inv_col.clone(),
        comm_w_plus_r_inv_col.clone(),
      ],
      &[
        blind_L_row.clone(),
        blind_L_col.clone(),
        pk.S_blinds.val_A.clone(),
        pk.S_blinds.val_B.clone(),
        pk.S_blinds.val_C.clone(),
        pk.S_blinds.row.clone(),
        pk.S_blinds.col.clone(),
        pk.S_blinds.ts_row.clone(),
        pk.S_blinds.ts_col.clone(),
        blind_t_plus_r_inv_row.clone(),
        blind_w_plus_r_inv_row.clone(),
        blind_t_plus_r_inv_col.clone(),
        blind_w_plus_r_inv_col.clone(),
      ],
      &[
        &L_row,
        &L_col,
        &pk.S_repr.val_A,
        &pk.S_repr.val_B,
        &pk.S_repr.val_C,
        &pk.S_repr.row,
        &pk.S_repr.col,
        &pk.S_repr.ts_row,
        &pk.S_repr.ts_col,
        &memory.t_plus_r_inv_row,
        &memory.w_plus_r_inv_row,
        &memory.t_plus_r_inv_col,
        &memory.w_plus_r_inv_col,
      ],
      &evals,
    )?;

    info!(elapsed_ms = %prove_t.elapsed().as_millis(), "pp_spartan_snark_prove");
    Ok(Self {
      U,
      comm_L_row,
      comm_L_col,
      comm_t_plus_r_inv_row,
      comm_w_plus_r_inv_row,
      comm_t_plus_r_inv_col,
      comm_w_plus_r_inv_col,
      sc_proof_outer,
      claims_outer,
      sc_proof_inner,
      eval_W,
      blind_eval_W,
      eval_arg_W,
      eval_L_row: evals[0],
      eval_L_col: evals[1],
      eval_val_A: evals[2],
      eval_val_B: evals[3],
      eval_val_C: evals[4],
      eval_row: evals[5],
      eval_col: evals[6],
      eval_ts_row: evals[7],
      eval_ts_col: evals[8],
      eval_t_plus_r_inv_row: evals[9],
      eval_w_plus_r_inv_row: evals[10],
      eval_t_plus_r_inv_col: evals[11],
      eval_w_plus_r_inv_col: evals[12],
      blind_eval_joint,
      eval_arg_joint,
    })
  }
}

impl<E: Engine> R1CSSNARKTrait<E> for PpSpartanSNARK<E>
where
  E::PCS: FoldingEngineTrait<E>,
{
  type ProverKey = PpSpartanProverKey<E>;
  type VerifierKey = PpSpartanVerifierKey<E>;
  type PrepSNARK = SpartanPrepSNARK<E>;

  fn setup<C: SpartanCircuit<E>>(
    circuit: C,
  ) -> Result<(Self::ProverKey, Self::VerifierKey), SpartanError> {
    let (_setup_span, setup_t) = start_span!("pp_spartan_preprocessing_setup");
    let S = ShapeCS::r1cs_shape(&circuit)?;
    let S_repr = SplitR1CSShapeSparkRepr::new(&S);
    let max_commit_len = (S.num_shared + S.num_precommitted + S.num_rest).max(S_repr.N);
    let (ck, vk_ee) = E::PCS::setup(b"ck_pp", max_commit_len, crate::DEFAULT_COMMITMENT_WIDTH);
    let (ck_s, _) = E::PCS::setup(b"ck_s", 1, MULTIROUND_COMMITMENT_WIDTH);
    let (S_comm, S_blinds) = S_repr.commit(&ck)?;

    let vk = PpSpartanVerifierKey {
      vk_ee,
      ck_s: ck_s.clone(),
      S: S.clone(),
      S_comm: S_comm.clone(),
      digest: OnceCell::new(),
    };
    let pk = PpSpartanProverKey {
      ck,
      ck_s,
      S,
      S_repr,
      S_comm,
      S_blinds,
      vk_digest: vk.digest()?,
    };
    info!(elapsed_ms = %setup_t.elapsed().as_millis(), "pp_spartan_preprocessing_setup");
    Ok((pk, vk))
  }

  fn prep_prove<C: SpartanCircuit<E>>(
    pk: &Self::ProverKey,
    circuit: C,
    is_small: bool,
  ) -> Result<Self::PrepSNARK, SpartanError> {
    let mut ps = SatisfyingAssignment::shared_witness(&pk.S, &pk.ck, &circuit, is_small)?;
    SatisfyingAssignment::precommitted_witness(&mut ps, &pk.S, &pk.ck, &circuit, is_small)?;
    Ok(ps)
  }

  fn prove<C: SpartanCircuit<E>>(
    pk: &Self::ProverKey,
    circuit: C,
    prep_snark: &Self::PrepSNARK,
    is_small: bool,
  ) -> Result<Self, SpartanError> {
    Self::prove_regular(pk, circuit, prep_snark, is_small)
  }

  fn verify(&self, vk: &Self::VerifierKey) -> Result<Vec<E::Scalar>, SpartanError> {
    let (_verify_span, verify_t) = start_span!("pp_spartan_snark_verify");
    let mut transcript = E::TE::new(b"PpSpartanSNARK");
    transcript.absorb(b"vk", &vk.digest()?);
    transcript.absorb(b"public_values", &self.U.public_values.as_slice());

    self.U.validate(&vk.S, &mut transcript)?;
    let U_regular = self.U.to_regular_instance()?;
    let num_vars = vk.S.num_shared + vk.S.num_precommitted + vk.S.num_rest;
    let num_rounds_x = usize::try_from(vk.S.num_cons.ilog2()).expect("constraint count log2 fits");
    let num_rounds_w = num_vars.log_2();
    let num_rounds_inner = vk.S_comm.N.log_2();

    let tau = (0..num_rounds_x)
      .map(|_| transcript.squeeze(b"t"))
      .collect::<Result<Vec<_>, SpartanError>>()?;
    let (claim_outer_final, r_x) =
      self
        .sc_proof_outer
        .verify(E::Scalar::ZERO, num_rounds_x, 3, &mut transcript)?;
    let tau_bound = EqPolynomial::new(tau).evaluate(&r_x);
    let claim_outer_expected =
      tau_bound * (self.claims_outer.0 * self.claims_outer.1 - self.claims_outer.2);
    if claim_outer_final != claim_outer_expected {
      return Err(SpartanError::InvalidSumcheckProof);
    }
    transcript.absorb(
      b"claims_outer",
      &[
        self.claims_outer.0,
        self.claims_outer.1,
        self.claims_outer.2,
      ]
      .as_slice(),
    );

    let r_pad = (0..(num_rounds_inner - num_rounds_x))
      .map(|_| transcript.squeeze(b"r_pad"))
      .collect::<Result<Vec<_>, SpartanError>>()?;
    let r_x_full = r_pad
      .iter()
      .chain(r_x.iter())
      .copied()
      .collect::<Vec<E::Scalar>>();
    let factor: E::Scalar = r_pad
      .iter()
      .fold(E::Scalar::ONE, |acc, r_i| acc * (E::Scalar::ONE - r_i));

    transcript.absorb(
      b"lookup_comms",
      &[self.comm_L_row.clone(), self.comm_L_col.clone()].as_slice(),
    );
    let c = transcript.squeeze(b"inner_c")?;
    let gamma = transcript.squeeze(b"mem_gamma")?;
    let r_mem = transcript.squeeze(b"mem_r")?;
    transcript.absorb(
      b"memory_comms",
      &[
        self.comm_t_plus_r_inv_row.clone(),
        self.comm_w_plus_r_inv_row.clone(),
        self.comm_t_plus_r_inv_col.clone(),
        self.comm_w_plus_r_inv_col.clone(),
      ]
      .as_slice(),
    );
    let rho = (0..num_rounds_inner)
      .map(|_| transcript.squeeze(b"mem_rho"))
      .collect::<Result<Vec<_>, SpartanError>>()?;
    let inner_batch_challenge = transcript.squeeze(b"inner_batch")?;
    let weights = powers::<E>(&inner_batch_challenge, 8);
    let claim_abc =
      factor * (self.claims_outer.0 + c * self.claims_outer.1 + c * c * self.claims_outer.2);
    let initial_claim = weights[6] * claim_abc;
    let (claim_inner_final, r_inner) =
      self
        .sc_proof_inner
        .verify(initial_claim, num_rounds_inner, 3, &mut transcript)?;

    let witness_point = extract_suffix_point::<E>(&r_inner, num_rounds_w);
    let eval_W_ext = eval_witness_ext_from_base::<E>(self.eval_W, &r_inner, num_vars);
    let eq_rho = EqPolynomial::new(rho).evaluate(&r_inner);
    let eq_r_x = EqPolynomial::new(r_x_full.clone()).evaluate(&r_inner);
    let eval_t_plus_r_row = identity_eval::<E>(num_rounds_inner, &r_inner) + gamma * eq_r_x + r_mem;
    let eval_w_plus_r_row = self.eval_row + gamma * self.eval_L_row + r_mem;
    let eval_z_full = eval_z_full_from_base_witness::<E>(
      self.eval_W,
      &self.U.public_values,
      &self.U.challenges,
      &r_inner,
      num_vars,
    );
    let eval_t_plus_r_col =
      identity_eval::<E>(num_rounds_inner, &r_inner) + gamma * eval_z_full + r_mem;
    let eval_w_plus_r_col = self.eval_col + gamma * self.eval_L_col + r_mem;
    let masked_eq = masked_eq_eval::<E>(&r_x_full, num_rounds_w, &r_inner);

    let claim_terms = [
      weights[0] * (self.eval_t_plus_r_inv_row - self.eval_w_plus_r_inv_row),
      weights[1] * (self.eval_t_plus_r_inv_col - self.eval_w_plus_r_inv_col),
      weights[2] * (eq_rho * (self.eval_t_plus_r_inv_row * eval_t_plus_r_row - self.eval_ts_row)),
      weights[3] * (eq_rho * (self.eval_w_plus_r_inv_row * eval_w_plus_r_row - E::Scalar::ONE)),
      weights[4] * (eq_rho * (self.eval_t_plus_r_inv_col * eval_t_plus_r_col - self.eval_ts_col)),
      weights[5] * (eq_rho * (self.eval_w_plus_r_inv_col * eval_w_plus_r_col - E::Scalar::ONE)),
      weights[6]
        * self.eval_L_row
        * self.eval_L_col
        * (self.eval_val_A + c * self.eval_val_B + c * c * self.eval_val_C),
      weights[7] * masked_eq * eval_W_ext,
    ];
    let claim_inner_expected: E::Scalar = claim_terms.iter().copied().sum();
    if claim_inner_final != claim_inner_expected {
      return Err(SpartanError::InvalidSumcheckProof);
    }

    let (_pcs_verify_span, pcs_verify_t) = start_span!("pp_spartan_pcs_verify");
    let comm_eval_W = E::PCS::commit(&vk.ck_s, &[self.eval_W], &self.blind_eval_W, false)?;
    E::PCS::verify(
      &vk.vk_ee,
      &vk.ck_s,
      &mut transcript,
      &U_regular.comm_W,
      &witness_point,
      &comm_eval_W,
      &self.eval_arg_W,
    )?;

    Self::verify_same_point_batch(
      vk,
      &mut transcript,
      &r_inner,
      &[
        self.comm_L_row.clone(),
        self.comm_L_col.clone(),
        vk.S_comm.comm_val_A.clone(),
        vk.S_comm.comm_val_B.clone(),
        vk.S_comm.comm_val_C.clone(),
        vk.S_comm.comm_row.clone(),
        vk.S_comm.comm_col.clone(),
        vk.S_comm.comm_ts_row.clone(),
        vk.S_comm.comm_ts_col.clone(),
        self.comm_t_plus_r_inv_row.clone(),
        self.comm_w_plus_r_inv_row.clone(),
        self.comm_t_plus_r_inv_col.clone(),
        self.comm_w_plus_r_inv_col.clone(),
      ],
      &[
        self.eval_L_row,
        self.eval_L_col,
        self.eval_val_A,
        self.eval_val_B,
        self.eval_val_C,
        self.eval_row,
        self.eval_col,
        self.eval_ts_row,
        self.eval_ts_col,
        self.eval_t_plus_r_inv_row,
        self.eval_w_plus_r_inv_row,
        self.eval_t_plus_r_inv_col,
        self.eval_w_plus_r_inv_col,
      ],
      &self.blind_eval_joint,
      &self.eval_arg_joint,
    )?;
    info!(elapsed_ms = %pcs_verify_t.elapsed().as_millis(), "pp_spartan_pcs_verify");

    info!(elapsed_ms = %verify_t.elapsed().as_millis(), "pp_spartan_snark_verify");
    Ok(self.U.public_values.clone())
  }

  fn pk_sizes(pk: &Self::ProverKey) -> [usize; 10] {
    pk.sizes()
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{small_field::DelayedReduction, traits::snark::R1CSSNARKTrait};
  use bellpepper::gadgets::num::AllocatedNum;
  use bellpepper_core::{ConstraintSystem, SynthesisError};

  #[derive(Clone, Default)]
  struct CubicCircuit;

  impl<E: Engine> SpartanCircuit<E> for CubicCircuit {
    fn shared<CS: ConstraintSystem<E::Scalar>>(
      &self,
      _cs: &mut CS,
    ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
      Ok(vec![])
    }

    fn precommitted<CS: ConstraintSystem<E::Scalar>>(
      &self,
      _cs: &mut CS,
      _shared: &[AllocatedNum<E::Scalar>],
    ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
      Ok(vec![])
    }

    fn num_challenges(&self) -> usize {
      0
    }

    fn synthesize<CS: ConstraintSystem<E::Scalar>>(
      &self,
      cs: &mut CS,
      _shared: &[AllocatedNum<E::Scalar>],
      _precommitted: &[AllocatedNum<E::Scalar>],
      _challenges: Option<&[E::Scalar]>,
    ) -> Result<(), SynthesisError> {
      let x = AllocatedNum::alloc(cs.namespace(|| "x"), || Ok(E::Scalar::from(2u64)))?;
      let x_sq = x.square(cs.namespace(|| "x_sq"))?;
      let x_cu = x_sq.mul(cs.namespace(|| "x_cu"), &x)?;
      let y = AllocatedNum::alloc(cs.namespace(|| "y"), || {
        Ok(x_cu.get_value().unwrap() + x.get_value().unwrap() + E::Scalar::from(5u64))
      })?;

      cs.enforce(
        || "x^3 + x + 5 = y",
        |lc| {
          lc + x_cu.get_variable()
            + x.get_variable()
            + CS::one()
            + CS::one()
            + CS::one()
            + CS::one()
            + CS::one()
        },
        |lc| lc + CS::one(),
        |lc| lc + y.get_variable(),
      );
      let _ = y.inputize(cs.namespace(|| "output"));
      Ok(())
    }

    fn public_values(&self) -> Result<Vec<E::Scalar>, SynthesisError> {
      Ok(vec![E::Scalar::from(15u64)])
    }
  }

  fn test_snark_with<E: Engine>()
  where
    E::Scalar: DelayedReduction<E::Scalar>,
    E::PCS: FoldingEngineTrait<E>,
  {
    let circuit = CubicCircuit;
    let (pk, vk) = PpSpartanSNARK::<E>::setup(circuit.clone()).unwrap();
    let prep = PpSpartanSNARK::<E>::prep_prove(&pk, circuit.clone(), false).unwrap();
    let proof = PpSpartanSNARK::<E>::prove(&pk, circuit.clone(), &prep, false).unwrap();
    let out = proof.verify(&vk).unwrap();
    assert_eq!(out, vec![E::Scalar::from(15u64)]);
  }

  #[test]
  fn test_pp_snark() {
    type E = crate::provider::PallasHyraxEngine;
    test_snark_with::<E>();

    type E2 = crate::provider::VestaHyraxEngine;
    test_snark_with::<E2>();
  }
}
