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
/// D is the degree bound of t_i(X) (not s_i); for Spartan, D = 2.
///
/// # Type Parameters
///
/// - `F`: Field type with small-value and delayed reduction support
/// - `SV`: Witness value type (i32 or i64)
///
/// # Arguments
///
/// - `az`, `bz`: Witness polynomials (small integer values)
/// - `taus`: Challenge vector of length ℓ
/// - `l0`: Number of small-value rounds (prefix variables)
/// - `e_in`: Precomputed eq evaluations for inner loop variables (from EqSumCheckInstance)
/// - `e_xout`: Precomputed eq evaluations for outer loop variables (from EqSumCheckInstance)
///
/// # Parallelism strategy
///
/// - Outer parallel loop over x_out values (using Rayon fold-reduce)
/// - Each thread maintains thread-local accumulators
/// - Final reduction merges all thread-local results via element-wise addition
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
  e_in: &[F],
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

  // e_in and e_xout are precomputed eq tables for suffix variables, derived from
  // EqSumCheckInstance to enable pyramid reuse. Split follows EqSumCheckInstance convention:
  // - e_in = floor((l-l0)/2) inner vars (FIRST half of suffix taus)
  // - e_xout = ceil((l-l0)/2) outer vars (LAST half of suffix taus)
  let in_vars = e_in.len().trailing_zeros() as usize; // floor (inner loop)
  let xout_vars = e_xout.len().trailing_zeros() as usize; // ceil (outer loop)
  let suffix_vars = xout_vars + in_vars;
  let num_x_out = e_xout.len();

  let BetaContributions {
    contributions: beta_contributions,
    num_betas,
  } = compute_beta_contributions::<2>(l0);

  // Only betas containing at least one ∞ coordinate contribute non-zero values.
  // On binary inputs {0,1}^n, Az·Bz = Cz (R1CS identity), so Az·Bz - Cz = 0.
  // The ∞ coordinate corresponds to the "leading coefficient" of the Lagrange polynomial,
  // which is non-zero only for non-constant polynomials. Non-binary evaluations (those
  // with ∞) are where we accumulate the sumcheck.
  let betas_with_infty: Vec<usize> = (0..num_betas)
    .filter(|&i| (0..l0).any(|d| (i / base.pow(d as u32)).is_multiple_of(base)))
    .collect();

  let ext_size = base.pow(l0 as u32); // (D+1)^l0

  // Parallel over x_out with thread-local state (zero per-iteration allocations)
  type State<F2, SV2> = SpartanThreadState<F2, SV2, 2>;

  let fold_results: Vec<State<F, SV>> = (0..num_x_out)
    .into_par_iter()
    .fold(
      || State::<F, SV>::new(num_betas, prefix_size, ext_size),
      |mut state: State<F, SV>, x_out_bits| {
        // Reset partial sums for this x_out iteration
        state.reset_partial_sums();

        // Inner loop over x_in - accumulate into UNREDUCED form
        // Each beta_partial_sums[beta_idx] accumulates 2^(l/2) terms per x_out.
        // Safety bound for SignedWideLimbs<N> (N limbs, 64 bits per limb):
        //   field_bits + product_bits + (l/2) < 64*N
        // i32 path: N=5, product_bits<=62; i64 path: N=6, product_bits<=126.
        for (x_in_bits, e_in_eval) in e_in.iter().enumerate() {
          // Suffix layout: x_in (inner loop) in HIGH bits, x_out (outer loop) in LOW bits
          // This matches the tau ordering where e_in uses earlier taus and e_xout uses later taus
          let suffix = (x_in_bits << xout_vars) | x_out_bits;

          // Fill prefix buffers by index assignment (no allocation)
          #[allow(clippy::needless_range_loop)]
          for prefix in 0..prefix_size {
            let idx = (prefix << suffix_vars) | suffix;
            state.az.boolean_evals[prefix] = az.Z[idx];
            state.bz.boolean_evals[prefix] = bz.Z[idx];
          }

          // Extend Az and Bz from boolean hypercube {0,1}^l0 to Lagrange domain {∞,0,1}^l0.
          // This transforms 2^l0 boolean evaluations into 3^l0 extended evaluations.
          //
          // For each coordinate, the extension computes:
          //   - ext[∞] = p(1) - p(0)  (the "leading coefficient" or slope)
          //   - ext[0] = p(0)         (original value at 0)
          //   - ext[1] = p(1)         (original value at 1)
          //
          // For multi-variate extension, values at ∞ coordinates are linear combinations:
          //   ext[∞,0] = p(1,0) - p(0,0)
          //   ext[∞,∞] = p(1,1) - p(0,1) - p(1,0) + p(0,0)
          //
          // We use batch extension (O(3^l0)) rather than computing each beta value directly
          // (O(4^l0)) because the iterative algorithm memoizes intermediate differences,
          // avoiding redundant computation of shared sub-expressions.
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

          // Only process betas with ∞ - binary betas contribute 0 for satisfying witnesses
          // Uses delayed modular reduction: accumulates into unreduced wide-limb form.
          // wide_mul computes small × small → product, then unreduced_multiply_accumulate adds field × product
          for &beta_idx in &betas_with_infty {
            let prod = SV::wide_mul(az_ext[beta_idx], bz_ext[beta_idx]);
            F::unreduced_multiply_accumulate(&mut state.partial_sums[beta_idx], e_in_eval, &prod);
          }
        }

        // Pre-compute and filter: reduce all non-zero betas upfront
        // This eliminates closure call overhead in the accumulator building loop
        // Reuse pre-allocated buffer to avoid per-iteration allocations
        for &beta_idx in &betas_with_infty {
          if state.partial_sums[beta_idx].is_zero() {
            continue;
          }
          // Reduce partial sum to field element
          let val = <F as DelayedReduction<SV::Product>>::reduce(&state.partial_sums[beta_idx]);
          if val == F::ZERO {
            continue;
          }
          state.beta_values.push((beta_idx, val));
        }

        // Phase 5: Accumulate S[β] += E_X[x_out] × val
        // E_Y factor is applied in final scatter after all x_out iterations
        let e_xout_val = &e_xout[x_out_bits];
        for &(beta_idx, ref val) in &state.beta_values {
          <F as DelayedReduction<F>>::unreduced_multiply_accumulate(
            &mut state.s_beta[beta_idx],
            val,
            e_xout_val,
          );
        }

        state
      },
    )
    .collect();

  // Sequential merge: combines s_beta arrays element-wise.
  // Using std::iter::Iterator::reduce (not rayon's) - no extra state allocations.
  let merged = fold_results
    .into_iter()
    .reduce(|mut a, b| {
      for (a_s, b_s) in a.s_beta.iter_mut().zip(&b.s_beta) {
        *a_s += *b_s;
      }
      a
    })
    .expect("num_x_out > 0 guarantees non-empty fold results");

  // Phase 6: Final scatter - A_i(v,u) += E_Y,i[y] × S[β]
  // E_Y factor is applied ONCE per β after all x_out, not once per (x_out, β)
  let mut result: LagrangeAccumulators<F, 2> = LagrangeAccumulators::new(l0);
  for beta_idx in 0..num_betas {
    if merged.s_beta[beta_idx].is_zero() {
      continue;
    }
    let s_val = <F as DelayedReduction<F>>::reduce(&merged.s_beta[beta_idx]);
    if s_val == F::ZERO {
      continue;
    }

    // u32/u8 → usize casts are free on x86-64 and ARM64 (zero-extension is implicit)
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

  /// Test helper: compute eq tables for e_in and e_xout from taus.
  /// Uses balanced split: e_in gets ceiling half, e_xout gets floor half.
  fn compute_eq_tables_for_test(taus: &[Scalar], l0: usize) -> (Vec<Scalar>, Vec<Scalar>) {
    let l = taus.len();
    let suffix_vars = l - l0;
    let in_vars = suffix_vars.div_ceil(2); // ceiling (inner loop, larger)

    // Balanced split: e_in from FIRST in_vars, e_xout from remaining (floor)
    let e_in = EqPolynomial::evals_from_points(&taus[l0..l0 + in_vars]);
    let e_xout = EqPolynomial::evals_from_points(&taus[l0 + in_vars..]);

    (e_in, e_xout)
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

    let (e_in, e_xout) = compute_eq_tables_for_test(&taus, l0);
    let acc = build_accumulators_spartan(&az, &bz, &taus, l0, &e_in, &e_xout);

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
    let (e_in, e_xout) = compute_eq_tables_for_test(&taus, l0);
    let acc1 = build_accumulators_spartan(&az, &bz, &taus, l0, &e_in, &e_xout);
    let acc2 = build_accumulators_spartan(&az, &bz, &taus, l0, &e_in, &e_xout);

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
    let (e_in, e_xout) = compute_eq_tables_for_test(&taus, l0);
    let acc1 = build_accumulators_spartan(&az, &bz, &taus, l0, &e_in, &e_xout);
    let acc2 = build_accumulators_spartan(&az, &bz, &taus, l0, &e_in, &e_xout);

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
