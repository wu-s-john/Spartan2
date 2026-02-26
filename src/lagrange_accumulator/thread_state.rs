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
/// Without this optimization, the fold closure allocates 5 vectors on every x_out iteration:
/// ```ignore
/// |mut acc, x_out_bits| {
///     let mut beta_partial_sums = vec![S::ZERO; num_betas];     // ALLOC
///     let mut az_prefix_boolean_evals = vec![...];                               // ALLOC
///     let mut bz_prefix_boolean_evals = vec![...];                               // ALLOC
///     let mut buf_a = vec![...];                                 // ALLOC
///     let mut buf_b = vec![...];                                 // ALLOC
///     ...
/// }
/// ```
///
/// For typical workloads (l=20, l0=4), num_x_out = 2^6 = 64, causing 320 allocations
/// per parallel task. With Rayon's work-stealing, this leads to significant allocator
/// contention and cache pollution.
///
/// # Solution
///
/// By hoisting these buffers into a struct created once per Rayon thread subdivision
/// (in the fold identity closure), we reduce allocations from O(num_x_out) to O(num_threads).
/// The `reset_partial_sums()` method zeros the sums between iterations (cheap memset).
///
/// # Buffer Layout
///
/// - `az`, `bz`: Extension buffers for Az and Bz polynomials. Both must be available
///   simultaneously to compute Az(β) × Bz(β) for each β.
/// - `s_beta`: Global accumulator S[β] = Σ E_X[x_out] × val, accumulated across x_out iterations.
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
  /// Partial sums indexed by β, accumulated over the x_in loop.
  /// Uses unreduced wide-limb form for delayed modular reduction.
  /// Reset each x_out iteration.
  pub partial_sums: Vec<<F as DelayedReduction<SV::Product>>::Accumulator>,
  /// S[β] = Σ_{x_out} E_X[x_out] × reduced_partial_sum[β]
  /// Accumulated across all x_out iterations within a fold task, then merged.
  /// Uses unreduced F×F form for delayed modular reduction.
  pub s_beta: Vec<<F as DelayedReduction<F>>::Accumulator>,
  /// Extension buffers for Az polynomial.
  pub az: PolyExtensionBuffers<SV>,
  /// Extension buffers for Bz polynomial.
  pub bz: PolyExtensionBuffers<SV>,
  /// Reusable buffer for filtered (beta_idx, reduced_value) pairs in accumulator building phase.
  /// Values are field elements from reducing partial sums.
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
      partial_sums: vec![
        <F as DelayedReduction<SV::Product>>::Accumulator::zero();
        num_betas
      ],
      s_beta: vec![<F as DelayedReduction<F>>::Accumulator::zero(); num_betas],
      az: PolyExtensionBuffers::new(prefix_size, ext_size),
      bz: PolyExtensionBuffers::new(prefix_size, ext_size),
      beta_values: Vec::with_capacity(num_betas),
    }
  }

  /// Zero out partial sums for the next x_out iteration.
  /// This is O(num_betas) but much cheaper than reallocating.
  #[inline]
  pub fn reset_partial_sums(&mut self) {
    for sum in &mut self.partial_sums {
      *sum = <F as DelayedReduction<SV::Product>>::Accumulator::zero();
    }
    self.beta_values.clear();
  }
}
