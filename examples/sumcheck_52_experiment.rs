//! 5×52 Column Accumulator Sumcheck Experiment
//!
//! Benchmarks 5×52 column-accumulator MAC kernels against the existing 4×64
//! delayed-reduction sumcheck provers. All variants produce identical proofs.
//!
//! Run with: cargo run --release --example sumcheck_52_experiment
//! Profile single size: cargo run --release --example sumcheck_52_experiment -- --vars 20

use ff::Field;
use num_traits::Zero;
use rayon::prelude::*;
use spartan2::{
  polys::{eq::EqPolynomial, multilinear::MultilinearPolynomial, univariate::UniPoly},
  provider::Bn254Engine,
  small_field::{
    limbs52::{ColumnAcc9, SignedColumnAcc5, mac_ff_52_mixed, mac_fi64_52, to_52},
    montgomery::MontgomeryLimbs,
  },
  small_sumcheck::prove_cubic_small_value,
  sumcheck::{
    PAR_THRESHOLD, SumcheckProof, bind_three_polys_top,
    eq_sumcheck::eval_one_case_cubic_three_inputs,
  },
  traits::{Engine, transcript::TranscriptEngineTrait},
};
use std::time::Instant;

type E = Bn254Engine;
type F = <E as Engine>::Scalar;

// ============================================================================
// 5×52 bind_three_polys_batched_small_value (Step 4)
// ============================================================================

/// Batch-bind l0 top variables using 5×52 column accumulators for field×i64 MAC.
///
/// Equivalent to `bind_three_polys_batched_small_value` but with pre-converted
/// eq table in 5×52 form and `SignedColumnAcc5` accumulators.
fn bind_three_polys_batched_small_value_52(
  poly_a_small: &MultilinearPolynomial<i64>,
  poly_b_small: &MultilinearPolynomial<i64>,
  poly_c_small: &MultilinearPolynomial<i64>,
  challenges: &[F],
) -> (
  MultilinearPolynomial<F>,
  MultilinearPolynomial<F>,
  MultilinearPolynomial<F>,
) {
  let l0 = challenges.len();
  let n = poly_a_small.Z.len();
  debug_assert_eq!(n % (1 << l0), 0);

  let num_prefixes = 1usize << l0;
  let stride = n >> l0;

  // Precompute eq(challenges, p) for all p ∈ {0,1}^l0, converted to 5×52
  let eq_table_field = EqPolynomial::evals_from_points(challenges);
  let eq_table_52: Vec<[u64; 5]> = eq_table_field.iter().map(|e| to_52(e.to_limbs())).collect();

  let compute = |s: usize| -> (F, F, F) {
    let mut acc_a = SignedColumnAcc5::zero();
    let mut acc_b = SignedColumnAcc5::zero();
    let mut acc_c = SignedColumnAcc5::zero();

    for p in 0..num_prefixes {
      let idx = p * stride + s;
      let eq_52 = &eq_table_52[p];

      let val_a = poly_a_small.Z[idx];
      let val_b = poly_b_small.Z[idx];
      let val_c = poly_c_small.Z[idx];

      // field × i64 MAC in 5×52 form
      // Handle sign: accumulate into pos or neg
      if val_a >= 0 {
        mac_fi64_52(&mut acc_a.pos, eq_52, val_a as u64);
      } else {
        mac_fi64_52(&mut acc_a.neg, eq_52, (-val_a) as u64);
      }
      if val_b >= 0 {
        mac_fi64_52(&mut acc_b.pos, eq_52, val_b as u64);
      } else {
        mac_fi64_52(&mut acc_b.neg, eq_52, (-val_b) as u64);
      }
      if val_c >= 0 {
        mac_fi64_52(&mut acc_c.pos, eq_52, val_c as u64);
      } else {
        mac_fi64_52(&mut acc_c.neg, eq_52, (-val_c) as u64);
      }
    }

    (acc_a.reduce::<F>(), acc_b.reduce::<F>(), acc_c.reduce::<F>())
  };

  let results: Vec<(F, F, F)> = if stride >= PAR_THRESHOLD {
    (0..stride).into_par_iter().map(compute).collect()
  } else {
    (0..stride).map(compute).collect()
  };

  let mut out_a = Vec::with_capacity(stride);
  let mut out_b = Vec::with_capacity(stride);
  let mut out_c = Vec::with_capacity(stride);
  for (a, b, c) in results {
    out_a.push(a);
    out_b.push(b);
    out_c.push(c);
  }

  (
    MultilinearPolynomial::new(out_a),
    MultilinearPolynomial::new(out_b),
    MultilinearPolynomial::new(out_c),
  )
}

// ============================================================================
// 5×52 evaluation_points_cubic (Step 5)
// ============================================================================

// ============================================================================
// Eq table precomputation (Step 5)
// ============================================================================

/// Precompute eq polynomial tables in 5×52 form.
///
/// Replicates EqSumCheckInstance::new's eq polynomial construction,
/// then converts all entries to 5×52 representation.
fn preconvert_eq_tables(
  taus: &[F],
) -> (Vec<Vec<[u64; 5]>>, Vec<Vec<[u64; 5]>>, usize, usize) {
  let l = taus.len();
  let first_half = l / 2;
  let second_half = l - first_half;

  let compute_eq_polynomials = |taus: Vec<&F>| -> Vec<Vec<F>> {
    let len = taus.len();
    let mut result = Vec::with_capacity(len + 1);
    result.push(vec![F::ONE]);

    for i in 0..len {
      let tau = taus[i];
      let prev = &result[i];
      let mut v_next = prev.to_vec();
      v_next.par_extend(prev.par_iter().map(|v| *v * tau));
      let (first, last) = v_next.split_at_mut(prev.len());
      first.par_iter_mut().zip(last).for_each(|(a, b)| *a -= *b);
      result.push(v_next);
    }
    result
  };

  let (left_taus, right_taus) = taus.split_at(first_half);
  let left_taus_vec = left_taus.iter().skip(1).rev().collect::<Vec<_>>();
  let right_taus_vec = right_taus.iter().rev().collect::<Vec<_>>();

  let (poly_eq_left, poly_eq_right) = rayon::join(
    || compute_eq_polynomials(left_taus_vec),
    || compute_eq_polynomials(right_taus_vec),
  );

  // Convert to 5×52
  let poly_eq_left_52: Vec<Vec<[u64; 5]>> = poly_eq_left
    .iter()
    .map(|v| v.iter().map(|f| to_52(f.to_limbs())).collect())
    .collect();
  let poly_eq_right_52: Vec<Vec<[u64; 5]>> = poly_eq_right
    .iter()
    .map(|v| v.iter().map(|f| to_52(f.to_limbs())).collect())
    .collect();

  (poly_eq_left_52, poly_eq_right_52, first_half, second_half)
}

/// Eq state tracker — replicates the private fields of EqSumCheckInstance
/// that we need for update_evals.
struct EqState {
  round: usize,
  eval_eq_left: F,
  eq_tau_0_2_3: Vec<(F, F, F)>,
  taus: Vec<F>,
  init_num_vars: usize,
}

impl EqState {
  fn new(taus: &[F]) -> Self {
    let l = taus.len();
    let f2 = F::ONE.double();
    let f1 = F::ONE;
    let eq_tau_0_2_3: Vec<(F, F, F)> = taus
      .iter()
      .map(|tau| {
        let tau2 = tau.double();
        let tau3 = tau2 + tau;
        let tau5 = tau3 + tau2;
        (f1 - tau, tau3 - f1, tau5 - f2)
      })
      .collect();

    Self {
      round: 1,
      eval_eq_left: F::ONE,
      eq_tau_0_2_3,
      taus: taus.to_vec(),
      init_num_vars: l,
    }
  }

  fn update_evals(&self, eval_0: &mut F, eval_2: &mut F, eval_3: &mut F) {
    let p = self.eval_eq_left;
    let eq = self.eq_tau_0_2_3[self.round - 1];
    *eval_0 *= eq.0 * p;
    *eval_2 *= eq.1 * p;
    *eval_3 *= eq.2 * p;
  }

  fn bound(&mut self, r: &F) {
    let tau = self.taus[self.round - 1];
    self.eval_eq_left *= F::ONE - tau - r + (*r * tau).double();
    self.round += 1;
  }
}

/// True 5×52 split-eq prover with custom eq state tracking.
fn prove_split_eq_52_real(
  claim: &F,
  taus: Vec<F>,
  poly_a: &mut MultilinearPolynomial<F>,
  poly_b: &mut MultilinearPolynomial<F>,
  poly_c: &mut MultilinearPolynomial<F>,
  transcript: &mut <E as Engine>::TE,
) -> (SumcheckProof<E>, Vec<F>, Vec<F>) {
  let num_rounds = taus.len();
  let mut r: Vec<F> = Vec::new();
  let mut polys = Vec::new();
  let mut claim_per_round = *claim;

  let mut eq_state = EqState::new(&taus);

  // Pre-convert eq tables to 5×52
  let (poly_eq_left_52, poly_eq_right_52, first_half, second_half) =
    preconvert_eq_tables(&taus);

  for round in 0..num_rounds {
    let half_p = poly_a.Z.len() / 2;
    let in_first_half = eq_state.round < first_half;

    let (mut eval_0, mut eval_2, mut eval_3) = if in_first_half {
      eval_points_first_half_52(
        &poly_eq_left_52,
        &poly_eq_right_52,
        first_half,
        second_half,
        eq_state.round,
        half_p,
        round,
        poly_a,
        poly_b,
        poly_c,
      )
    } else {
      eval_points_second_half_52(
        &poly_eq_right_52,
        eq_state.init_num_vars,
        eq_state.round,
        half_p,
        round,
        poly_a,
        poly_b,
        poly_c,
      )
    };

    eq_state.update_evals(&mut eval_0, &mut eval_2, &mut eval_3);

    let evals = [
      eval_0,
      claim_per_round - eval_0,
      eval_2,
      eval_3,
    ];
    let poly = UniPoly::from_evals(&evals).unwrap();

    transcript.absorb(b"p", &poly);
    let r_i = transcript.squeeze(b"c").unwrap();
    r.push(r_i);
    polys.push(poly.compress());

    claim_per_round = poly.evaluate(&r_i);

    bind_three_polys_top(poly_a, poly_b, poly_c, &r_i);
    eq_state.bound(&r_i);
  }

  (
    SumcheckProof::new(polys),
    r,
    vec![poly_a[0], poly_b[0], poly_c[0]],
  )
}

/// First-half eval points using 5×52 column accumulators (two-phase).
fn eval_points_first_half_52(
  poly_eq_left_52: &[Vec<[u64; 5]>],
  poly_eq_right_52: &[Vec<[u64; 5]>],
  first_half: usize,
  second_half: usize,
  round: usize,
  half_p: usize,
  round_idx: usize,
  poly_a: &MultilinearPolynomial<F>,
  poly_b: &MultilinearPolynomial<F>,
  poly_c: &MultilinearPolynomial<F>,
) -> (F, F, F) {
  let eq_left_52 = &poly_eq_left_52[first_half - round];
  let eq_right_52 = &poly_eq_right_52[second_half];
  let eq_out_len = eq_left_52.len();

  let min_chunk = (eq_out_len / (rayon::current_num_threads() * 4)).max(1);
  let (acc_0, acc_2, acc_3) = (0..eq_out_len)
    .into_par_iter()
    .with_min_len(min_chunk)
    .fold(
      || (ColumnAcc9::zero(), ColumnAcc9::zero(), ColumnAcc9::zero()),
      |mut outer_acc, x_out| {
        let e_out_52 = &eq_left_52[x_out];

        let mut inner_0 = ColumnAcc9::zero();
        let mut inner_2 = ColumnAcc9::zero();
        let mut inner_3 = ColumnAcc9::zero();

        for (x_in, e_in_52) in eq_right_52.iter().enumerate() {
          let id = (x_out << second_half) | x_in;

          let (zero_a, one_a) = (&poly_a.Z[id], &poly_a.Z[id + half_p]);
          let (zero_b, one_b) = (&poly_b.Z[id], &poly_b.Z[id + half_p]);
          let (zero_c, one_c) = (&poly_c.Z[id], &poly_c.Z[id + half_p]);

          let (q0, q2, q3) = eval_one_case_cubic_three_inputs(
            round_idx, zero_a, one_a, zero_b, one_b, zero_c, one_c,
          );

          mac_ff_52_mixed(&mut inner_0.0, e_in_52, q0.to_limbs());
          mac_ff_52_mixed(&mut inner_2.0, e_in_52, q2.to_limbs());
          mac_ff_52_mixed(&mut inner_3.0, e_in_52, q3.to_limbs());
        }

        // Reduce inner, then outer MAC
        let inner_0_red: F = inner_0.reduce();
        let inner_2_red: F = inner_2.reduce();
        let inner_3_red: F = inner_3.reduce();

        mac_ff_52_mixed(&mut outer_acc.0 .0, e_out_52, inner_0_red.to_limbs());
        mac_ff_52_mixed(&mut outer_acc.1 .0, e_out_52, inner_2_red.to_limbs());
        mac_ff_52_mixed(&mut outer_acc.2 .0, e_out_52, inner_3_red.to_limbs());

        outer_acc
      },
    )
    .reduce(
      || (ColumnAcc9::zero(), ColumnAcc9::zero(), ColumnAcc9::zero()),
      |mut a, b| {
        a.0 += b.0;
        a.1 += b.1;
        a.2 += b.2;
        a
      },
    );

  (acc_0.reduce::<F>(), acc_2.reduce::<F>(), acc_3.reduce::<F>())
}

/// Second-half eval points using 5×52 column accumulators (single phase).
fn eval_points_second_half_52(
  poly_eq_right_52: &[Vec<[u64; 5]>],
  init_num_vars: usize,
  round: usize,
  half_p: usize,
  round_idx: usize,
  poly_a: &MultilinearPolynomial<F>,
  poly_b: &MultilinearPolynomial<F>,
  poly_c: &MultilinearPolynomial<F>,
) -> (F, F, F) {
  let eq_right_52 = &poly_eq_right_52[init_num_vars - round];

  let min_chunk = (half_p / (rayon::current_num_threads() * 4)).max(1);
  let (acc_0, acc_2, acc_3) = (0..half_p)
    .into_par_iter()
    .with_min_len(min_chunk)
    .fold(
      || (ColumnAcc9::zero(), ColumnAcc9::zero(), ColumnAcc9::zero()),
      |mut acc, id| {
        let e_52 = &eq_right_52[id];

        let (zero_a, one_a) = (&poly_a.Z[id], &poly_a.Z[id + half_p]);
        let (zero_b, one_b) = (&poly_b.Z[id], &poly_b.Z[id + half_p]);
        let (zero_c, one_c) = (&poly_c.Z[id], &poly_c.Z[id + half_p]);

        let (q0, q2, q3) = eval_one_case_cubic_three_inputs(
          round_idx, zero_a, one_a, zero_b, one_b, zero_c, one_c,
        );

        mac_ff_52_mixed(&mut acc.0 .0, e_52, q0.to_limbs());
        mac_ff_52_mixed(&mut acc.1 .0, e_52, q2.to_limbs());
        mac_ff_52_mixed(&mut acc.2 .0, e_52, q3.to_limbs());

        acc
      },
    )
    .reduce(
      || (ColumnAcc9::zero(), ColumnAcc9::zero(), ColumnAcc9::zero()),
      |mut a, b| {
        a.0 += b.0;
        a.1 += b.1;
        a.2 += b.2;
        a
      },
    );

  (acc_0.reduce::<F>(), acc_2.reduce::<F>(), acc_3.reduce::<F>())
}

/// Small-value prover with 5×52 MAC in both bind and remaining rounds.
fn prove_small_i64_52(
  claim: &F,
  taus: Vec<F>,
  poly_a_small: &MultilinearPolynomial<i64>,
  poly_b_small: &MultilinearPolynomial<i64>,
  poly_c_small: &MultilinearPolynomial<i64>,
  transcript: &mut <E as Engine>::TE,
) -> (SumcheckProof<E>, Vec<F>, Vec<F>) {
  use spartan2::lagrange_accumulator::{SPARTAN_T_DEGREE, build_accumulators_spartan, derive_t1};
  use spartan2::small_sumcheck::{SmallValueSumCheck, build_univariate_round_polynomial};

  let num_rounds = taus.len();
  let mut r: Vec<F> = Vec::with_capacity(num_rounds);
  let mut polys = Vec::new();
  let mut claim_per_round = *claim;

  let l0 = std::cmp::min(3usize, num_rounds.saturating_sub(1));
  assert!(l0 > 0);

  // Small-value rounds (0 to l0-1) — same as standard, uses Lagrange accumulators
  let accumulators = build_accumulators_spartan(poly_a_small, poly_b_small, &taus, l0);
  let mut small_value =
    SmallValueSumCheck::<F, SPARTAN_T_DEGREE>::from_accumulators(accumulators);

  for round in 0..l0 {
    let t_all = small_value.eval_t_all_u(round);
    let t_inf = t_all.at_infinity();
    let t0 = t_all.at_zero();
    let li = small_value.eq_round_values(taus[round]);
    let t1 = derive_t1(li.at_zero(), li.at_one(), claim_per_round, t0).unwrap();

    let poly = build_univariate_round_polynomial(&li, t0, t1, t_inf);

    transcript.absorb(b"p", &poly);
    let r_i = transcript.squeeze(b"c").unwrap();
    r.push(r_i);
    polys.push(poly.compress());
    claim_per_round = poly.evaluate(&r_i);
    small_value.advance(&li, r_i);
  }

  // Transition: bind using 5×52 MAC
  let (mut poly_a, mut poly_b, mut poly_c) =
    bind_three_polys_batched_small_value_52(poly_a_small, poly_b_small, poly_c_small, &r[..l0]);

  // Remaining rounds: use 5×52 eval points
  let mut eq_state = EqState::new(&taus);
  // Advance eq_state by l0 rounds
  for r_i in &r[..l0] {
    eq_state.bound(r_i);
  }

  let (poly_eq_left_52, poly_eq_right_52, first_half, second_half) =
    preconvert_eq_tables(&taus);

  for round in l0..num_rounds {
    let half_p = poly_a.Z.len() / 2;
    let in_first_half = eq_state.round < first_half;

    let (mut eval_0, mut eval_2, mut eval_3) = if in_first_half {
      eval_points_first_half_52(
        &poly_eq_left_52,
        &poly_eq_right_52,
        first_half,
        second_half,
        eq_state.round,
        half_p,
        round,
        &poly_a,
        &poly_b,
        &poly_c,
      )
    } else {
      eval_points_second_half_52(
        &poly_eq_right_52,
        eq_state.init_num_vars,
        eq_state.round,
        half_p,
        round,
        &poly_a,
        &poly_b,
        &poly_c,
      )
    };

    eq_state.update_evals(&mut eval_0, &mut eval_2, &mut eval_3);

    let evals = [
      eval_0,
      claim_per_round - eval_0,
      eval_2,
      eval_3,
    ];
    let poly = UniPoly::from_evals(&evals).unwrap();

    transcript.absorb(b"p", &poly);
    let r_i = transcript.squeeze(b"c").unwrap();
    r.push(r_i);
    polys.push(poly.compress());

    claim_per_round = poly.evaluate(&r_i);

    bind_three_polys_top(&mut poly_a, &mut poly_b, &mut poly_c, &r_i);
    eq_state.bound(&r_i);
  }

  (
    SumcheckProof::new(polys),
    r,
    vec![poly_a[0], poly_b[0], poly_c[0]],
  )
}

// ============================================================================
// Test data and verification helpers
// ============================================================================

fn make_test_data(num_vars: usize) -> (Vec<i32>, Vec<i32>, Vec<F>, F) {
  let n = 1usize << num_vars;
  let az_i32: Vec<i32> = (0..n).map(|i| (i + 1) as i32).collect();
  let bz_i32: Vec<i32> = (0..n).map(|i| (i + 3) as i32).collect();
  let taus: Vec<F> = (0..num_vars).map(|i| F::from((i + 2) as u64)).collect();
  let claim = F::ZERO;
  (az_i32, bz_i32, taus, claim)
}

fn make_field_polys(
  az_i32: &[i32],
  bz_i32: &[i32],
) -> (
  MultilinearPolynomial<F>,
  MultilinearPolynomial<F>,
  MultilinearPolynomial<F>,
) {
  let az: Vec<F> = az_i32.iter().map(|&v| F::from(v as u64)).collect();
  let bz: Vec<F> = bz_i32.iter().map(|&v| F::from(v as u64)).collect();
  let cz: Vec<F> = az.iter().zip(&bz).map(|(a, b)| *a * *b).collect();
  (
    MultilinearPolynomial::new(az),
    MultilinearPolynomial::new(bz),
    MultilinearPolynomial::new(cz),
  )
}

fn make_small_polys(
  az_i32: &[i32],
  bz_i32: &[i32],
) -> (
  MultilinearPolynomial<i64>,
  MultilinearPolynomial<i64>,
  MultilinearPolynomial<i64>,
) {
  let az: Vec<i64> = az_i32.iter().map(|&v| v as i64).collect();
  let bz: Vec<i64> = bz_i32.iter().map(|&v| v as i64).collect();
  let cz: Vec<i64> = az.iter().zip(&bz).map(|(a, b)| a * b).collect();
  (
    MultilinearPolynomial::new(az),
    MultilinearPolynomial::new(bz),
    MultilinearPolynomial::new(cz),
  )
}

fn verify_proof(
  proof: &SumcheckProof<E>,
  claim: &F,
  num_vars: usize,
  taus: &[F],
  expected_r: &[F],
  expected_evals: &[F],
) {
  let mut transcript_v = <E as Engine>::TE::new(b"bench");
  let (final_claim, r_v) = proof.verify(*claim, num_vars, 3, &mut transcript_v).unwrap();
  assert_eq!(r_v, expected_r, "Verify challenges must match prover");
  let tau_eval = EqPolynomial::new(taus.to_vec()).evaluate(&r_v);
  let expected = tau_eval * (expected_evals[0] * expected_evals[1] - expected_evals[2]);
  assert_eq!(final_claim, expected, "Sumcheck final claim mismatch");
}

// ============================================================================
// Main
// ============================================================================

struct BenchResult {
  name: &'static str,
  prove_us: u128,
  proof: SumcheckProof<E>,
  r: Vec<F>,
  evals: Vec<F>,
}

fn main() {
  let args: Vec<String> = std::env::args().collect();
  let num_vars = if args.len() > 2 && args[1] == "--vars" {
    args[2].parse::<usize>().unwrap()
  } else {
    20
  };

  eprintln!("=== Sumcheck 5x52 Experiment (num_vars={}) ===\n", num_vars);

  let (az_i32, bz_i32, taus, claim) = make_test_data(num_vars);

  let mut results: Vec<BenchResult> = Vec::new();

  // 1. Baseline: prove_cubic_with_three_inputs
  {
    let (mut az, mut bz, mut cz) = make_field_polys(&az_i32, &bz_i32);
    let mut transcript = <E as Engine>::TE::new(b"bench");

    let t = Instant::now();
    let (proof, r, evals) = SumcheckProof::<E>::prove_cubic_with_three_inputs(
      &claim,
      taus.clone(),
      &mut az,
      &mut bz,
      &mut cz,
      &mut transcript,
    )
    .unwrap();
    let prove_us = t.elapsed().as_micros();

    verify_proof(&proof, &claim, num_vars, &taus, &r, &evals);
    results.push(BenchResult {
      name: "baseline",
      prove_us,
      proof,
      r,
      evals,
    });
  }

  // 2. Split-eq delayed reduction
  {
    let (mut az, mut bz, mut cz) = make_field_polys(&az_i32, &bz_i32);
    let mut transcript = <E as Engine>::TE::new(b"bench");

    let t = Instant::now();
    let (proof, r, evals) = SumcheckProof::<E>::prove_cubic_with_three_inputs_split_eq_delayed(
      &claim,
      taus.clone(),
      &mut az,
      &mut bz,
      &mut cz,
      &mut transcript,
    )
    .unwrap();
    let prove_us = t.elapsed().as_micros();

    verify_proof(&proof, &claim, num_vars, &taus, &r, &evals);
    results.push(BenchResult {
      name: "split_eq_dmr",
      prove_us,
      proof,
      r,
      evals,
    });
  }

  // 3. Small-value i64
  {
    let (az, bz, cz) = make_small_polys(&az_i32, &bz_i32);
    let mut transcript = <E as Engine>::TE::new(b"bench");

    let t = Instant::now();
    let (proof, r, evals) =
      prove_cubic_small_value::<E, _, 3>(&claim, taus.clone(), &az, &bz, &cz, &mut transcript)
        .unwrap();
    let prove_us = t.elapsed().as_micros();

    verify_proof(&proof, &claim, num_vars, &taus, &r, &evals);
    results.push(BenchResult {
      name: "small_i64",
      prove_us,
      proof,
      r,
      evals,
    });
  }

  // 4. Split-eq with 5×52 MAC
  {
    let (mut az, mut bz, mut cz) = make_field_polys(&az_i32, &bz_i32);
    let mut transcript = <E as Engine>::TE::new(b"bench");

    let t = Instant::now();
    let (proof, r, evals) = prove_split_eq_52_real(
      &claim,
      taus.clone(),
      &mut az,
      &mut bz,
      &mut cz,
      &mut transcript,
    );
    let prove_us = t.elapsed().as_micros();

    verify_proof(&proof, &claim, num_vars, &taus, &r, &evals);
    results.push(BenchResult {
      name: "split_eq_dmr_52",
      prove_us,
      proof,
      r,
      evals,
    });
  }

  // 5. Small-value i64 with 5×52 in bind + remaining rounds
  {
    let (az, bz, cz) = make_small_polys(&az_i32, &bz_i32);
    let mut transcript = <E as Engine>::TE::new(b"bench");

    let t = Instant::now();
    let (proof, r, evals) = prove_small_i64_52(
      &claim,
      taus.clone(),
      &az,
      &bz,
      &cz,
      &mut transcript,
    );
    let prove_us = t.elapsed().as_micros();

    verify_proof(&proof, &claim, num_vars, &taus, &r, &evals);
    results.push(BenchResult {
      name: "small_i64_52",
      prove_us,
      proof,
      r,
      evals,
    });
  }

  // Verify all proofs match
  let baseline = &results[0];
  let mut all_match = true;
  for result in &results[1..] {
    let matches = result.proof == baseline.proof && result.r == baseline.r && result.evals == baseline.evals;
    if !matches {
      eprintln!("MISMATCH: {} does not match baseline!", result.name);
      all_match = false;
    }
  }

  // Print results table
  let baseline_us = results[0].prove_us as f64;
  eprintln!(
    "{:<25} {:>12} {:>10} {:>12}",
    "Method", "prove_us", "speedup", "proof_match"
  );
  eprintln!("{}", "-".repeat(61));
  for result in &results {
    let speedup = baseline_us / result.prove_us as f64;
    let matches = result.proof == baseline.proof && result.r == baseline.r && result.evals == baseline.evals;
    eprintln!(
      "{:<25} {:>12} {:>9.2}x {:>12}",
      result.name,
      result.prove_us,
      speedup,
      if matches { "OK" } else { "FAIL" }
    );
  }

  if all_match {
    eprintln!("\nAll proofs match baseline.");
  } else {
    eprintln!("\nSome proofs FAILED to match!");
    std::process::exit(1);
  }
}
