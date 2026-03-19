// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Thread-local scratch buffers for accumulator building.
//!
//! These structs eliminate per-iteration heap allocations in the parallel fold loops
//! of `build_accumulators_spartan` and `build_accumulators_neutronnova`. By hoisting buffer
//! allocations to the fold identity closure (called once per Rayon thread subdivision),
//! we reduce allocations from O(num_x_out) to O(num_threads).

use super::accumulator::LagrangeAccumulators;
use crate::small_field::{DelayedReduction, SmallValueField, WitnessValue, WideMul};
use ff::PrimeField;
use std::ops::{Add, Sub};

/// Thread-local scratch buffers for `build_accumulators_spartan`.
///
/// # Type Parameters
///
/// - `F`: Field type with small-value and delayed reduction support
/// - `SmallValue`: Witness value type (i32 or i64 for small-value witnesses)
/// - `D`: Polynomial degree bound
pub(crate) struct SpartanThreadState<F, SmallValue, const D: usize>
where
  F: PrimeField
    + SmallValueField<SmallValue>
    + DelayedReduction<SmallValue>
    + DelayedReduction<SmallValue::Product>
    + DelayedReduction<F>
    + Send
    + Sync,
  SmallValue: WideMul
    + Copy
    + Default
    + Add<Output = SmallValue>
    + Sub<Output = SmallValue>
    + Send
    + Sync,
{
  /// Partial sums indexed by β, accumulated over the x_in loop.
  pub partial_sums: Vec<<F as DelayedReduction<SmallValue::Product>>::Accumulator>,
  /// Bucket accumulators for accumulator building phase.
  pub acc: LagrangeAccumulators<<F as DelayedReduction<F>>::Accumulator, D>,
  /// Prefix evaluations of Az for current suffix. Size: 2^l0
  pub az_prefix_boolean_evals: Vec<SmallValue>,
  /// Prefix evaluations of Bz for current suffix. Size: 2^l0
  pub bz_prefix_boolean_evals: Vec<SmallValue>,
  /// Result buffer for Az Lagrange extension.
  pub az_extended_evals: Vec<SmallValue>,
  /// Scratch buffer for Az Lagrange extension.
  pub az_extended_scratch: Vec<SmallValue>,
  /// Result buffer for Bz Lagrange extension.
  pub bz_extended_evals: Vec<SmallValue>,
  /// Scratch buffer for Bz Lagrange extension.
  pub bz_extended_scratch: Vec<SmallValue>,
  /// Reusable buffer for filtered (beta_idx, reduced_value) pairs.
  pub beta_values: Vec<(usize, F)>,
}

impl<F, SmallValue, const D: usize> SpartanThreadState<F, SmallValue, D>
where
  F: PrimeField
    + SmallValueField<SmallValue>
    + DelayedReduction<SmallValue>
    + DelayedReduction<SmallValue::Product>
    + DelayedReduction<F>
    + Send
    + Sync,
  SmallValue: WideMul
    + Copy
    + Default
    + Add<Output = SmallValue>
    + Sub<Output = SmallValue>
    + Send
    + Sync,
{
  pub fn new(l0: usize, num_betas: usize, prefix_size: usize, ext_size: usize) -> Self {
    Self {
      partial_sums: vec![Default::default(); num_betas],
      acc: LagrangeAccumulators::new(l0),
      az_prefix_boolean_evals: vec![SmallValue::default(); prefix_size],
      bz_prefix_boolean_evals: vec![SmallValue::default(); prefix_size],
      az_extended_evals: vec![SmallValue::default(); ext_size],
      az_extended_scratch: vec![SmallValue::default(); ext_size],
      bz_extended_evals: vec![SmallValue::default(); ext_size],
      bz_extended_scratch: vec![SmallValue::default(); ext_size],
      beta_values: Vec::with_capacity(num_betas),
    }
  }

  /// Zero out partial sums for the next x_out iteration.
  #[inline]
  pub fn reset_partial_sums(&mut self) {
    for sum in &mut self.partial_sums {
      *sum = Default::default();
    }
    self.beta_values.clear();
  }
}

/// Thread-local scratch buffers for `build_accumulators_neutronnova`.
///
/// # Type Parameters
///
/// - `F`: Field type for partial sums and scatter accumulators
/// - `SmallValue`: Value type for pref/extension buffers (i32, i64, etc.)
/// - `PS`: Partial sum type (Accumulator for delayed reduction)
/// - `D`: Polynomial degree bound
pub(crate) struct NeutronNovaThreadState<F, SmallValue, PS: Copy + Default + PartialEq, const D: usize>
where
  F: PrimeField
    + SmallValueField<SmallValue>
    + DelayedReduction<SmallValue>
    + DelayedReduction<SmallValue::Product>
    + DelayedReduction<F>
    + Send
    + Sync,
  SmallValue: WideMul
    + Copy
    + Default
    + Add<Output = SmallValue>
    + Sub<Output = SmallValue>
    + Send
    + Sync,
{
  /// Partial sums indexed by β, accumulated over the x_L loop. Reset each x_R iteration.
  pub partial_sums: Vec<PS>,
  /// Bucket accumulators for scatter phase (accumulator for field × field products).
  pub scatter_acc: LagrangeAccumulators<<F as DelayedReduction<F>>::Accumulator, D>,
  /// Prefix evaluations of Az for current x_R. Size: 2^l_b
  pub az_prefix_boolean_evals: Vec<SmallValue>,
  /// Prefix evaluations of Bz for current x_R. Size: 2^l_b
  pub bz_prefix_boolean_evals: Vec<SmallValue>,
  /// Result buffer for Az Lagrange extension. Size: 3^l_b
  pub az_extended_evals: Vec<SmallValue>,
  /// Scratch buffer for Az Lagrange extension.
  pub az_extended_scratch: Vec<SmallValue>,
  /// Result buffer for Bz Lagrange extension.
  pub bz_extended_evals: Vec<SmallValue>,
  /// Scratch buffer for Bz Lagrange extension.
  pub bz_extended_scratch: Vec<SmallValue>,
  /// Reusable buffer for filtered (beta_idx, reduced_value) pairs in scatter phase.
  pub beta_values: Vec<(usize, F)>,
}

impl<F, SmallValue, PS: Copy + Default + PartialEq, const D: usize>
  NeutronNovaThreadState<F, SmallValue, PS, D>
where
  F: PrimeField
    + SmallValueField<SmallValue>
    + DelayedReduction<SmallValue>
    + DelayedReduction<SmallValue::Product>
    + DelayedReduction<F>
    + Send
    + Sync,
  SmallValue: WideMul
    + Copy
    + Default
    + Add<Output = SmallValue>
    + Sub<Output = SmallValue>
    + Send
    + Sync,
{
  pub fn new(l0: usize, num_betas: usize, prefix_size: usize, ext_size: usize) -> Self {
    Self {
      partial_sums: vec![PS::default(); num_betas],
      scatter_acc: LagrangeAccumulators::new(l0),
      az_prefix_boolean_evals: vec![SmallValue::default(); prefix_size],
      bz_prefix_boolean_evals: vec![SmallValue::default(); prefix_size],
      az_extended_evals: vec![SmallValue::default(); ext_size],
      az_extended_scratch: vec![SmallValue::default(); ext_size],
      bz_extended_evals: vec![SmallValue::default(); ext_size],
      bz_extended_scratch: vec![SmallValue::default(); ext_size],
      beta_values: Vec::with_capacity(num_betas),
    }
  }

  /// Zero out partial sums for the next x_R iteration.
  #[inline]
  pub fn reset_partial_sums(&mut self) {
    self.partial_sums.fill(PS::default());
    self.beta_values.clear();
  }
}

/// Thread-local scratch buffers for `build_accumulators_inner`.
///
/// # Type Parameters
///
/// - `F`: Field type with delayed reduction support
/// - `W`: Witness value type (bool, i8, etc.)
/// - `D`: Polynomial degree bound
pub(crate) struct InnerThreadState<F, W: WitnessValue, const D: usize>
where
  F: PrimeField
    + DelayedReduction<W::Extended>
    + DelayedReduction<F>
    + Send
    + Sync,
  W::Extended: Copy + Default + Add<Output = W::Extended> + Sub<Output = W::Extended> + Send + Sync,
{
  /// Partial sums indexed by β, accumulated over suffixes.
  pub partial_sums: Vec<<F as DelayedReduction<W::Extended>>::Accumulator>,
  /// Bucket accumulators — plain field addition (no eq weighting).
  pub acc: LagrangeAccumulators<F, D>,
  /// Extension buffer for z. The first `prefix_size` slots are also used as
  /// the gather target before in-place extension. Size: (D+1)^l0
  pub z_extended_evals: Vec<W::Extended>,
  /// Scratch buffer for z Lagrange extension. Size: (D+1)^l0
  pub z_extended_scratch: Vec<W::Extended>,
  /// Extension buffer for M̃. The first `prefix_size` slots are also used as
  /// the gather target before in-place extension. Size: (D+1)^l0
  pub M_extended_evals: Vec<F>,
  /// Scratch buffer for M̃ Lagrange extension. Size: (D+1)^l0
  pub M_extended_scratch: Vec<F>,
}

impl<F, W: WitnessValue, const D: usize> InnerThreadState<F, W, D>
where
  F: PrimeField
    + DelayedReduction<W::Extended>
    + DelayedReduction<F>
    + Send
    + Sync,
  W::Extended: Copy + Default + Add<Output = W::Extended> + Sub<Output = W::Extended> + Send + Sync,
{
  pub fn new(l0: usize, num_betas: usize, ext_size: usize) -> Self {
    Self {
      partial_sums: vec![Default::default(); num_betas],
      acc: LagrangeAccumulators::new(l0),
      z_extended_evals: vec![W::Extended::default(); ext_size],
      z_extended_scratch: vec![W::Extended::default(); ext_size],
      M_extended_evals: vec![F::ZERO; ext_size],
      M_extended_scratch: vec![F::ZERO; ext_size],
    }
  }
}
