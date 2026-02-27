// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Thread-local scratch buffers for accumulator building.
//!
//! These structs eliminate per-iteration heap allocations in the parallel fold loops
//! of `build_accumulators_spartan`. By hoisting buffer allocations to the fold identity
//! closure (called once per Rayon thread subdivision), we reduce allocations from
//! O(num_x_out) to O(num_threads).

use crate::big_num::{DelayedReduction, SmallValue, SmallValueEngine};
use num_traits::Zero;

/// Reusable buffers for Lagrange extension of a single polynomial (Az or Bz).
///
/// Groups the three buffers needed for extending a polynomial from the Boolean
/// hypercube {0,1}^l0 to the Lagrange domain {∞,0,1,...,D}^l0.
pub(crate) struct PolyExtensionBuffers<SV> {
  /// Prefix evaluations over Boolean hypercube. Size: 2^l0
  pub boolean_evals: Vec<SV>,
  /// Result buffer after Lagrange extension. Size: (D+1)^l0
  pub extended_evals: Vec<SV>,
  /// Scratch buffer used during iterative extension. Size: (D+1)^l0
  pub extended_scratch: Vec<SV>,
}

impl<SV: Zero + Clone> PolyExtensionBuffers<SV> {
  /// Create new buffers with the given sizes.
  pub fn new(prefix_size: usize, ext_size: usize) -> Self {
    Self {
      boolean_evals: vec![SV::zero(); prefix_size],
      extended_evals: vec![SV::zero(); ext_size],
      extended_scratch: vec![SV::zero(); ext_size],
    }
  }
}

/// Thread-local scratch buffers for `build_accumulators_spartan`.
///
/// # Motivation
///
/// Without this optimization, the fold closure allocates vectors on every x_out iteration.
/// By hoisting these buffers into a struct created once per Rayon thread subdivision
/// (in the fold identity closure), we reduce allocations from O(num_x_out) to O(num_threads).
///
/// # Two-Pass Approach
///
/// The algorithm uses a two-pass approach to avoid allocating an expanded eq table:
/// - Pass 1 (x₀=0): Accumulates into `s0` using `e_in_rest` coefficients
/// - Pass 2 (x₀=1): Accumulates into `s1` using `e_in_rest` coefficients
/// - Combine: `result = (1-τ₀)×S₀ + τ₀×S₁`
///
/// # Buffer Layout
///
/// - `s0`, `s1`: Two-pass accumulators for F × Wide products (SignedWideLimbs)
/// - `s_beta`: Global accumulator for F × F products across x_out iterations (WideLimbs<9>)
/// - `az`, `bz`: Extension buffers for small-value polynomials
///
/// # Type Parameters
///
/// - `F`: Field type with small-value and delayed reduction support
/// - `SV`: Witness value type (i32 or i64 for small-value witnesses)
/// - `D`: Polynomial degree bound
pub(crate) struct SpartanThreadState<F, SV, const D: usize>
where
  F: SmallValueEngine<SV>,
  SV: SmallValue,
{
  /// S0 accumulator for pass 1 (x₀ = 0).
  /// Type: SignedWideLimbs<6> for i32/i64, SignedWideLimbs<7> for i128.
  /// Accumulates: F × SV::Product (field × wide integer).
  /// Reset at start of each x_out iteration.
  pub s0: Vec<<F as DelayedReduction<SV::Product>>::Accumulator>,
  /// S1 accumulator for pass 2 (x₀ = 1).
  /// Same type as s0.
  pub s1: Vec<<F as DelayedReduction<SV::Product>>::Accumulator>,
  /// S[β] = Σ_{x_out} combined_val[β]
  /// Accumulated across all x_out iterations within a fold task, then merged.
  /// Type: WideLimbs<9> for F × F products.
  pub s_beta: Vec<<F as DelayedReduction<F>>::Accumulator>,
  /// Extension buffers for Az polynomial.
  pub az: PolyExtensionBuffers<SV>,
  /// Extension buffers for Bz polynomial.
  pub bz: PolyExtensionBuffers<SV>,
  /// Reusable buffer for filtered (beta_idx, reduced_value) pairs.
  /// Eliminates per-x_out allocation overhead.
  pub beta_values: Vec<(usize, F)>,
}

impl<F, SV, const D: usize> SpartanThreadState<F, SV, D>
where
  F: SmallValueEngine<SV>,
  SV: SmallValue,
{
  pub fn new(num_betas: usize, prefix_size: usize, ext_size: usize) -> Self {
    Self {
      s0: vec![
        <F as DelayedReduction<SV::Product>>::Accumulator::zero();
        num_betas
      ],
      s1: vec![
        <F as DelayedReduction<SV::Product>>::Accumulator::zero();
        num_betas
      ],
      s_beta: vec![<F as DelayedReduction<F>>::Accumulator::zero(); num_betas],
      az: PolyExtensionBuffers::new(prefix_size, ext_size),
      bz: PolyExtensionBuffers::new(prefix_size, ext_size),
      beta_values: Vec::with_capacity(num_betas),
    }
  }

  /// Reset S0 accumulator for pass 1.
  /// Called at the start of each x_out iteration before pass 1.
  #[inline]
  pub fn reset_s0(&mut self) {
    for sum in &mut self.s0 {
      *sum = <F as DelayedReduction<SV::Product>>::Accumulator::zero();
    }
  }

  /// Reset S1 accumulator for pass 2.
  /// Called after pass 1, before pass 2.
  #[inline]
  pub fn reset_s1(&mut self) {
    for sum in &mut self.s1 {
      *sum = <F as DelayedReduction<SV::Product>>::Accumulator::zero();
    }
  }

  /// Clear beta_values buffer for reuse.
  #[inline]
  pub fn clear_beta_values(&mut self) {
    self.beta_values.clear();
  }
}
