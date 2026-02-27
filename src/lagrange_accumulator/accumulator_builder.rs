// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Builder functions for constructing Lagrange accumulators (Procedure 9).
//!
//! This module provides:
//! - [`build_accumulators_spartan`]: Optimized builder for Spartan's cubic relation

use super::{
  accumulator::LagrangeAccumulators, csr::Csr, domain::LagrangeIndex,
  extension::extend_to_lagrange_domain, index::AccumulatorPrefixIndex,
  thread_state::SpartanThreadState,
};
use crate::{
  big_num::{DelayedReduction, SmallValue, SmallValueEngine},
  polys::{eq::compute_suffix_eq_pyramid, multilinear::MultilinearPolynomial},
};
use num_traits::Zero;
use rayon::prelude::*;

use super::index::compute_idx4;

/// Polynomial degree D for Spartan's small-value sumcheck.
/// For Spartan's cubic relation (A·B - C), D=2 yields quadratic t_i.
pub const SPARTAN_T_DEGREE: usize = 2;

pub(crate) struct BetaContributions {
  contributions: Csr<AccumulatorPrefixIndex>,
  num_betas: usize,
}

/// Procedure 9: Build accumulators A_i(v, u) for Spartan's first sum-check (Algorithm 6).
///
/// Computes accumulators for: g(X) = eq(τ, X) · (Az(X) · Bz(X) - Cz(X))
///
/// # Two-Pass Approach
///
/// Uses a two-pass approach to avoid allocating an expanded eq table:
/// - Pass 1 (x₀=0): Accumulates into S0 using `e_in_rest` coefficients
/// - Pass 2 (x₀=1): Accumulates into S1 using `e_in_rest` coefficients
/// - Combine: `result = (1-τ₀)×S₀ + τ₀×S₁`
///
/// This eliminates the allocation from `expanded_eq_left()` by using the existing
/// pyramid slice `e_in_rest` and handling τ₀ algebraically.
///
/// # Arguments
///
/// - `az`, `bz`: Witness polynomials (small integer values)
/// - `taus`: Challenge vector of length ℓ
/// - `l0`: Number of small-value rounds (prefix variables)
/// - `tau0`: First inner tau for two-pass combination
/// - `e_in_rest`: eq(τ[1..in_vars], ·), size 2^(in_vars-1) - NO ALLOCATION
/// - `e_xout`: eq(τ[in_vars..], ·), size 2^xout_vars
///
/// # Spartan-specific optimizations (D=2)
///
/// - Skip binary betas: for satisfying witnesses, Az·Bz = Cz on {0,1}^n, so Az·Bz - Cz = 0
/// - Only process betas containing ∞ (non-binary evaluations where contributions are non-zero)
pub fn build_accumulators_spartan<F, SV>(
  az: &MultilinearPolynomial<SV>,
  bz: &MultilinearPolynomial<SV>,
  taus: &[F],
  l0: usize,
  tau0: F,
  e_in_rest: &[F],
  e_xout: &[F],
) -> LagrangeAccumulators<F, 2>
where
  F: SmallValueEngine<SV>,
  SV: SmallValue,
{
  let base: usize = 3; // D + 1 = 2 + 1 = 3
  let l = az.Z.len().trailing_zeros() as usize;
  debug_assert_eq!(az.Z.len(), 1usize << l, "poly size must be power of 2");
  debug_assert_eq!(az.Z.len(), bz.Z.len());
  debug_assert_eq!(taus.len(), l, "taus must have length ℓ");
  debug_assert!(l0 < l, "l0 must be < ℓ");

  let prefix_size = 1usize << l0;

  // Compute e_y for the l0 prefix variables (used in final scatter)
  let e_y = compute_suffix_eq_pyramid(&taus[..l0], l0);

  // Compute variable counts from the known suffix size and e_xout.
  // e_xout.len() = 2^xout_vars, and suffix_vars = in_vars + xout_vars.
  let suffix_vars = l - l0;
  let xout_vars = e_xout.len().trailing_zeros() as usize;
  let in_vars = suffix_vars - xout_vars;

  // Verify e_in_rest has the expected size: 2^(in_vars-1) if in_vars > 0, else 1
  debug_assert_eq!(
    e_in_rest.len(),
    if in_vars == 0 { 1 } else { 1 << (in_vars - 1) },
    "e_in_rest size mismatch: in_vars={}, expected {}, got {}",
    in_vars,
    if in_vars == 0 { 1 } else { 1 << (in_vars - 1) },
    e_in_rest.len()
  );
  let num_x_out = e_xout.len();

  // in_msb is only valid when in_vars > 0 (computed lazily to avoid underflow)
  let in_msb = if in_vars > 0 { 1usize << (in_vars - 1) } else { 0 };

  // Precompute tau0 factors for two-pass combination (only used when in_vars > 0)
  let one_minus_tau0 = F::ONE - tau0;

  let BetaContributions {
    contributions: beta_contributions,
    num_betas,
  } = compute_beta_contributions::<2>(l0);

  // Only betas containing at least one ∞ coordinate contribute non-zero values.
  let betas_with_infty: Vec<usize> = (0..num_betas)
    .filter(|&i| (0..l0).any(|d| (i / base.pow(d as u32)) % base == 0))
    .collect();

  let ext_size = base.pow(l0 as u32); // (D+1)^l0

  // Parallel over x_out with thread-local state
  type State<F2, SV2> = SpartanThreadState<F2, SV2, 2>;

  let fold_results: Vec<State<F, SV>> = (0..num_x_out)
    .into_par_iter()
    .fold(
      || State::<F, SV>::new(num_betas, prefix_size, ext_size),
      |mut state: State<F, SV>, x_out_bits| {
        // Handle two cases based on whether there are inner variables
        if in_vars == 0 {
          // ===== NO INNER VARIABLES =====
          // No two-pass needed; just iterate over x_out with coeff = e_xout
          let coeff = e_xout[x_out_bits];
          let suffix = x_out_bits; // No inner component

          // Load prefix evaluations
          #[allow(clippy::needless_range_loop)]
          for prefix in 0..prefix_size {
            let idx = (prefix << suffix_vars) | suffix;
            state.az.boolean_evals[prefix] = az.Z[idx];
            state.bz.boolean_evals[prefix] = bz.Z[idx];
          }

          // Extend to Lagrange domain
          let az_size = extend_to_lagrange_domain::<SV, 2>(
            &state.az.boolean_evals,
            &mut state.az.extended_evals,
            &mut state.az.extended_scratch,
          );
          let az_ext = &state.az.extended_evals[..az_size];

          let bz_size = extend_to_lagrange_domain::<SV, 2>(
            &state.bz.boolean_evals,
            &mut state.bz.extended_evals,
            &mut state.bz.extended_scratch,
          );
          let bz_ext = &state.bz.extended_evals[..bz_size];

          // Accumulate directly into s_beta
          for &beta_idx in &betas_with_infty {
            let prod = SV::wide_mul(az_ext[beta_idx], bz_ext[beta_idx]);
            // Convert product to field element via accumulator
            let mut temp_acc =
              <F as DelayedReduction<SV::Product>>::Accumulator::default();
            <F as DelayedReduction<SV::Product>>::unreduced_multiply_accumulate(
              &mut temp_acc,
              &F::ONE,
              &prod,
            );
            let prod_field = <F as DelayedReduction<SV::Product>>::reduce(&temp_acc);
            let val = prod_field * coeff;
            if val != F::ZERO {
              <F as DelayedReduction<F>>::unreduced_multiply_accumulate(
                &mut state.s_beta[beta_idx],
                &val,
                &F::ONE,
              );
            }
          }
        } else {
          // ===== TWO-PASS APPROACH (in_vars > 0) =====
          // Precompute combined coefficients for this x_out
          // c0 = e_xout[x_out] × (1 - τ₀)
          // c1 = e_xout[x_out] × τ₀
          let c0 = e_xout[x_out_bits] * one_minus_tau0;
          let c1 = e_xout[x_out_bits] * tau0;

          // ===== PASS 1: x₀ = 0 =====
          state.reset_s0();

          for (x_rest_bits, e_rest) in e_in_rest.iter().enumerate() {
            // x_in has MSB = 0, so x_in = x_rest
            let x_in_bits = x_rest_bits;
            let suffix = (x_in_bits << xout_vars) | x_out_bits;

            // Load prefix evaluations
            #[allow(clippy::needless_range_loop)]
            for prefix in 0..prefix_size {
              let idx = (prefix << suffix_vars) | suffix;
              state.az.boolean_evals[prefix] = az.Z[idx];
              state.bz.boolean_evals[prefix] = bz.Z[idx];
            }

            // Extend to Lagrange domain
            let az_size = extend_to_lagrange_domain::<SV, 2>(
              &state.az.boolean_evals,
              &mut state.az.extended_evals,
              &mut state.az.extended_scratch,
            );
            let az_ext = &state.az.extended_evals[..az_size];

            let bz_size = extend_to_lagrange_domain::<SV, 2>(
              &state.bz.boolean_evals,
              &mut state.bz.extended_evals,
              &mut state.bz.extended_scratch,
            );
            let bz_ext = &state.bz.extended_evals[..bz_size];

            // Accumulate into S0 for each non-binary β
            for &beta_idx in &betas_with_infty {
              let prod = SV::wide_mul(az_ext[beta_idx], bz_ext[beta_idx]);
              <F as DelayedReduction<SV::Product>>::unreduced_multiply_accumulate(
                &mut state.s0[beta_idx],
                e_rest,
                &prod,
              );
            }
          }

          // ===== PASS 2: x₀ = 1 =====
          state.reset_s1();

          for (x_rest_bits, e_rest) in e_in_rest.iter().enumerate() {
            // x_in has MSB = 1
            let x_in_bits = in_msb | x_rest_bits;
            let suffix = (x_in_bits << xout_vars) | x_out_bits;

            // Load prefix evaluations
            #[allow(clippy::needless_range_loop)]
            for prefix in 0..prefix_size {
              let idx = (prefix << suffix_vars) | suffix;
              state.az.boolean_evals[prefix] = az.Z[idx];
              state.bz.boolean_evals[prefix] = bz.Z[idx];
            }

            // Extend to Lagrange domain
            let az_size = extend_to_lagrange_domain::<SV, 2>(
              &state.az.boolean_evals,
              &mut state.az.extended_evals,
              &mut state.az.extended_scratch,
            );
            let az_ext = &state.az.extended_evals[..az_size];

            let bz_size = extend_to_lagrange_domain::<SV, 2>(
              &state.bz.boolean_evals,
              &mut state.bz.extended_evals,
              &mut state.bz.extended_scratch,
            );
            let bz_ext = &state.bz.extended_evals[..bz_size];

            // Accumulate into S1 for each non-binary β
            for &beta_idx in &betas_with_infty {
              let prod = SV::wide_mul(az_ext[beta_idx], bz_ext[beta_idx]);
              <F as DelayedReduction<SV::Product>>::unreduced_multiply_accumulate(
                &mut state.s1[beta_idx],
                e_rest,
                &prod,
              );
            }
          }

          // ===== COMBINE & ACCUMULATE =====
          // For each β: val = s0_val × c0 + s1_val × c1
          // where c0 = e_xout × (1-τ₀), c1 = e_xout × τ₀
          state.clear_beta_values();

          for &beta_idx in &betas_with_infty {
            // Reduce S0 and S1 accumulators to field elements
            let s0_val = <F as DelayedReduction<SV::Product>>::reduce(&state.s0[beta_idx]);
            let s1_val = <F as DelayedReduction<SV::Product>>::reduce(&state.s1[beta_idx]);

            // Combine: (1-τ₀)×S₀ + τ₀×S₁, weighted by e_xout
            let val = s0_val * c0 + s1_val * c1;

            if val == F::ZERO {
              continue;
            }

            // Accumulate into s_beta
            <F as DelayedReduction<F>>::unreduced_multiply_accumulate(
              &mut state.s_beta[beta_idx],
              &val,
              &F::ONE, // Already weighted by e_xout in c0/c1
            );
          }
        }

        state
      },
    )
    .collect();

  // Sequential merge: combines s_beta arrays element-wise.
  let merged = fold_results
    .into_iter()
    .reduce(|mut a, b| {
      for (a_s, b_s) in a.s_beta.iter_mut().zip(&b.s_beta) {
        *a_s += *b_s;
      }
      a
    })
    .expect("num_x_out > 0 guarantees non-empty fold results");

  // Final scatter: A_i(v,u) += E_Y,i[y] × S[β]
  let mut result: LagrangeAccumulators<F, 2> = LagrangeAccumulators::new(l0);
  for beta_idx in 0..num_betas {
    if merged.s_beta[beta_idx].is_zero() {
      continue;
    }
    let s_val = <F as DelayedReduction<F>>::reduce(&merged.s_beta[beta_idx]);
    if s_val == F::ZERO {
      continue;
    }

    for pref in &beta_contributions[beta_idx] {
      let e_y_val = e_y[pref.round_0 as usize][pref.y_idx as usize];
      result.rounds[pref.round_0 as usize].data_mut()[pref.v_idx as usize][pref.u_idx as usize] +=
        s_val * e_y_val;
    }
  }
  result
}

// =============================================================================
// Helper functions
// =============================================================================
pub(crate) fn compute_beta_contributions<const D: usize>(l0: usize) -> BetaContributions {
  let base: usize = D + 1;
  let num_betas = base.pow(l0 as u32);
  let mut contributions: Csr<AccumulatorPrefixIndex> =
    Csr::with_capacity(num_betas, num_betas * l0);
  for b in 0..num_betas {
    let beta = LagrangeIndex::<D>::from_flat_index(b, l0);
    let entries = compute_idx4(&beta);
    contributions.push(&entries);
  }

  BetaContributions {
    contributions,
    num_betas,
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    lagrange_accumulator::domain::LagrangeHatPoint, polys::eq::EqPolynomial,
    provider::pasta::pallas,
  };
  use ff::Field;

  type Scalar = pallas::Scalar;

  // Use the shared constant for polynomial degree in tests
  const D: usize = SPARTAN_T_DEGREE;

  /// Test helper: compute eq tables for two-pass approach from taus.
  /// Uses balanced split: in_vars gets ceiling half, xout_vars gets floor half.
  /// Returns (tau0, e_in_rest, e_xout) for the new signature.
  fn compute_eq_tables_for_test(taus: &[Scalar], l0: usize) -> (Scalar, Vec<Scalar>, Vec<Scalar>) {
    let l = taus.len();
    let suffix_vars = l - l0;
    let in_vars = suffix_vars.div_ceil(2); // ceiling (inner loop, larger)

    // tau0 is the first inner tau
    let tau0 = taus[l0];

    // e_in_rest is eq table for taus[l0+1..l0+in_vars] (excluding tau0)
    let e_in_rest = if in_vars <= 1 {
      vec![Scalar::ONE] // Single element when in_vars is 0 or 1
    } else {
      EqPolynomial::evals_from_points(&taus[l0 + 1..l0 + in_vars])
    };

    // e_xout from remaining taus
    let e_xout = EqPolynomial::evals_from_points(&taus[l0 + in_vars..]);

    (tau0, e_in_rest, e_xout)
  }

  /// Binary-β zero shortcut: Az=Bz=Cz=first variable (x0), so Az·Bz−Cz=0 on binary β.
  /// Non-binary β (∞) should yield non-zero in some bucket.
  #[test]
  fn test_binary_beta_zero_shortcut_behavior() {
    // Use l0=1 so round 0 buckets are fed only by β of length 1 (easy to reason about).
    let l0 = 1;
    let l = 2;

    // Az = Bz = top bit x0 (most significant of 2 bits)
    // For satisfying witness, Cz = Az * Bz = Az (since Az ∈ {0,1} and Az = Bz)
    let az_vals: Vec<i32> = (0..(1 << l)).map(|bits| (bits >> (l - 1)) & 1).collect();
    let bz_vals: Vec<i32> = (0..(1 << l)).map(|bits| (bits >> (l - 1)) & 1).collect();

    let az = MultilinearPolynomial::new(az_vals);
    let bz = MultilinearPolynomial::new(bz_vals);

    let taus: Vec<Scalar> = vec![Scalar::from(3u64), Scalar::from(5u64)];

    let (tau0, e_in_rest, e_xout) = compute_eq_tables_for_test(&taus, l0);
    let acc = build_accumulators_spartan(&az, &bz, &taus, l0, tau0, &e_in_rest, &e_xout);

    // Only round 0 exists (v is empty). β ranges over U_d with binary {0,1} and non-binary {∞}.
    // Buckets for u = 0 should be zero (binary β), bucket for u = ∞ should be non-zero.
    let u_inf = LagrangeHatPoint::<D>::Infinity.to_index(); // 0
    let u_zero = LagrangeHatPoint::<D>::Finite(0).to_index(); // 1

    assert!(
      bool::from(acc.get(0, 0, u_zero).is_zero()),
      "binary β should give zero for u=0"
    );
    assert!(
      !bool::from(acc.get(0, 0, u_inf).is_zero()),
      "non-binary β (∞) should give non-zero"
    );
  }

  /// Test build_accumulators_spartan with i32 witnesses produces consistent results.
  ///
  /// Verifies that running the same computation twice produces the same output.
  #[test]
  fn test_build_accumulators_spartan_small_consistent() {
    let l0 = 2;

    // Define deterministic Az, Bz over {0,1}^4 using small values
    let eval = |bits: usize| -> i32 {
      let x0 = (bits >> 3) & 1;
      let x1 = (bits >> 2) & 1;
      let x2 = (bits >> 1) & 1;
      let x3 = bits & 1;
      (x0 + 2 * x1 + 3 * x2 + 4 * x3 + 5) as i32
    };

    let az_vals: Vec<i32> = (0..16).map(&eval).collect();
    let bz_vals: Vec<i32> = (0..16).map(|b| eval(b) + 7).collect();

    let az = MultilinearPolynomial::new(az_vals);
    let bz = MultilinearPolynomial::new(bz_vals);

    // Taus (length ℓ)
    let taus: Vec<Scalar> = vec![
      Scalar::from(5u64),
      Scalar::from(7u64),
      Scalar::from(11u64),
      Scalar::from(13u64),
    ];

    // Build accumulators twice
    let (tau0, e_in_rest, e_xout) = compute_eq_tables_for_test(&taus, l0);
    let acc1 = build_accumulators_spartan(&az, &bz, &taus, l0, tau0, &e_in_rest, &e_xout);
    let acc2 = build_accumulators_spartan(&az, &bz, &taus, l0, tau0, &e_in_rest, &e_xout);

    // Compare all buckets
    for round in 0..l0 {
      let num_v = (D + 1).pow(round as u32);
      for v_idx in 0..num_v {
        for u_idx in 0..D {
          let got = acc1.get(round, v_idx, u_idx);
          let expect = acc2.get(round, v_idx, u_idx);
          assert_eq!(
            got, expect,
            "Mismatch at round {}, v_idx {}, u_idx {}",
            round, v_idx, u_idx
          );
        }
      }
    }
  }

  /// Test build_accumulators_spartan with i32 witnesses using larger inputs to stress test.
  #[test]
  fn test_build_accumulators_spartan_small_larger() {
    let l0 = 3;
    let l = 10;
    let n = 1 << l;

    // Create polynomials with varying small values
    let az_vals: Vec<i32> = (0..n).map(|i| (i % 1000) + 1).collect();
    let bz_vals: Vec<i32> = (0..n).map(|i| ((i * 7) % 1000) + 1).collect();

    let az = MultilinearPolynomial::new(az_vals);
    let bz = MultilinearPolynomial::new(bz_vals);

    // Random-looking taus
    let taus: Vec<Scalar> = (0..l).map(|i| Scalar::from((i * 7 + 3) as u64)).collect();

    // Build accumulators twice to verify consistency
    let (tau0, e_in_rest, e_xout) = compute_eq_tables_for_test(&taus, l0);
    let acc1 = build_accumulators_spartan(&az, &bz, &taus, l0, tau0, &e_in_rest, &e_xout);
    let acc2 = build_accumulators_spartan(&az, &bz, &taus, l0, tau0, &e_in_rest, &e_xout);

    for round in 0..l0 {
      let num_v = (D + 1).pow(round as u32);
      for v_idx in 0..num_v {
        for u_idx in 0..D {
          let got = acc1.get(round, v_idx, u_idx);
          let expect = acc2.get(round, v_idx, u_idx);
          assert_eq!(
            got, expect,
            "Mismatch at round {}, v_idx {}, u_idx {}",
            round, v_idx, u_idx
          );
        }
      }
    }
  }
}
