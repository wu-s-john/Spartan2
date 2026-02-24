// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT

//! DelayedReduction trait for accumulating unreduced field products.
//!
//! Modular reduction (Montgomery REDC) is expensive. When summing many
//! products `Σ (field_i × field_j)`, the standard approach does N reductions.
//! Delayed reduction accumulates unreduced products in wide integers, reducing
//! only once at the end.

use super::{
  barrett::{barrett_reduce_6, barrett_reduce_7},
  limbs::{SignedWideLimbs, SubMagResult, WideLimbs, mac, mul_4_by_4, sub_mag},
  montgomery::{MontgomeryLimbs, montgomery_reduce_9},
  small_value_field::SupportsSmallI64,
};
use ff::PrimeField;
use num_traits::Zero;
use std::ops::AddAssign;

/// Trait for delayed modular reduction operations.
///
/// Accumulates unreduced products in wide integers, reducing only at the end.
pub trait DelayedReduction<Value>: Sized {
  /// Wide accumulator type for unreduced products.
  type Accumulator: Copy + Clone + Default + AddAssign + Send + Sync + Zero;

  /// Accumulate: `acc += field × value` without modular reduction.
  fn unreduced_multiply_accumulate(acc: &mut Self::Accumulator, field: &Self, value: &Value);

  /// Reduce the accumulator to a field element.
  fn reduce(acc: &Self::Accumulator) -> Self;
}

// ============================================================================
// DelayedReduction<i32> - for field × i32 products (direct small values)
// ============================================================================

impl<F: MontgomeryLimbs + PrimeField> DelayedReduction<i32> for F {
  /// Accumulator for field × i32 products.
  ///
  /// # Overflow Bounds
  /// - Field element: 254 bits (BN254 Fr)
  /// - i32 magnitude: 32 bits
  /// - Product size: 286 bits (5 limbs)
  /// - SignedWideLimbs<6>: 384 bits capacity
  /// - Headroom: 98 bits → supports up to 2^98 accumulations
  type Accumulator = SignedWideLimbs<6>;

  #[inline(always)]
  fn unreduced_multiply_accumulate(acc: &mut Self::Accumulator, field: &Self, value: &i32) {
    // Extend i32 to i64 and use the same accumulation logic
    let value64 = *value as i64;
    let (target, mag) = if value64 >= 0 {
      (&mut acc.pos, value64 as u64)
    } else {
      (&mut acc.neg, value64.wrapping_neg() as u64)
    };
    let a = field.to_limbs();
    let (r0, c) = mac(target.0[0], a[0], mag, 0);
    let (r1, c) = mac(target.0[1], a[1], mag, c);
    let (r2, c) = mac(target.0[2], a[2], mag, c);
    let (r3, c) = mac(target.0[3], a[3], mag, c);
    let (r4, of) = target.0[4].overflowing_add(c);
    target.0[0] = r0;
    target.0[1] = r1;
    target.0[2] = r2;
    target.0[3] = r3;
    target.0[4] = r4;
    target.0[5] = target.0[5].wrapping_add(of as u64);
  }

  #[inline(always)]
  fn reduce(acc: &Self::Accumulator) -> Self {
    match sub_mag::<6>(&acc.pos.0, &acc.neg.0) {
      SubMagResult::Positive(mag) => F::from_limbs(barrett_reduce_6::<F>(&mag)),
      SubMagResult::Negative(mag) => -F::from_limbs(barrett_reduce_6::<F>(&mag)),
    }
  }
}

// ============================================================================
// DelayedReduction<i64> - for field × i64 products (from i32 × i32 or direct)
// ============================================================================

impl<F: MontgomeryLimbs + PrimeField> DelayedReduction<i64> for F {
  /// Accumulator for field × i64 products.
  ///
  /// # Overflow Bounds
  /// - Field element: 254 bits (BN254 Fr)
  /// - i64 magnitude: 64 bits
  /// - Product size: 318 bits (5 limbs)
  /// - SignedWideLimbs<6>: 384 bits capacity
  /// - Headroom: 66 bits → supports up to 2^66 accumulations
  type Accumulator = SignedWideLimbs<6>;

  #[inline(always)]
  fn unreduced_multiply_accumulate(acc: &mut Self::Accumulator, field: &Self, value: &i64) {
    // Handle sign: accumulate into pos or neg based on sign of value
    let (target, mag) = if *value >= 0 {
      (&mut acc.pos, *value as u64)
    } else {
      (&mut acc.neg, (*value).wrapping_neg() as u64)
    };
    // Fused multiply-accumulate: field × value
    let a = field.to_limbs();
    let (r0, c) = mac(target.0[0], a[0], mag, 0);
    let (r1, c) = mac(target.0[1], a[1], mag, c);
    let (r2, c) = mac(target.0[2], a[2], mag, c);
    let (r3, c) = mac(target.0[3], a[3], mag, c);
    // Propagate carry without multiply (just add)
    let (r4, of) = target.0[4].overflowing_add(c);
    target.0[0] = r0;
    target.0[1] = r1;
    target.0[2] = r2;
    target.0[3] = r3;
    target.0[4] = r4;
    target.0[5] = target.0[5].wrapping_add(of as u64);
  }

  #[inline(always)]
  fn reduce(acc: &Self::Accumulator) -> Self {
    // Subtract in limb space first, then reduce once (saves one Barrett reduction)
    match sub_mag::<6>(&acc.pos.0, &acc.neg.0) {
      SubMagResult::Positive(mag) => F::from_limbs(barrett_reduce_6::<F>(&mag)),
      SubMagResult::Negative(mag) => -F::from_limbs(barrett_reduce_6::<F>(&mag)),
    }
  }
}

// ============================================================================
// DelayedReduction<i128> - for field × i128 products (from i64 × i64)
// ============================================================================

impl<F: SupportsSmallI64 + PrimeField> DelayedReduction<i128> for F {
  /// Accumulator for field × i128 products (from i64 × i64).
  ///
  /// # Overflow Bounds
  /// - Field element: 254 bits (BN254 Fr)
  /// - i128 magnitude: 128 bits
  /// - Product size: 382 bits (6 limbs)
  /// - SignedWideLimbs<7>: 448 bits capacity
  /// - Headroom: 66 bits → supports up to 2^66 accumulations
  type Accumulator = SignedWideLimbs<7>;

  #[inline(always)]
  fn unreduced_multiply_accumulate(acc: &mut Self::Accumulator, field: &Self, value: &i128) {
    let (target, mag) = if *value >= 0 {
      (&mut acc.pos, *value as u128)
    } else {
      (&mut acc.neg, (-*value) as u128)
    };
    // Fused 4×2 multiply-accumulate: two passes at different offsets
    let a = field.to_limbs();
    let b_lo = mag as u64;
    let b_hi = (mag >> 64) as u64;

    // Pass 1: multiply by b_lo at offset 0
    let (r0, c) = mac(target.0[0], a[0], b_lo, 0);
    let (r1, c) = mac(target.0[1], a[1], b_lo, c);
    let (r2, c) = mac(target.0[2], a[2], b_lo, c);
    let (r3, c) = mac(target.0[3], a[3], b_lo, c);
    // Propagate carry without multiply (just add)
    let (r4, of1) = target.0[4].overflowing_add(c);
    let c1 = of1 as u64;
    target.0[0] = r0;

    // Pass 2: multiply by b_hi at offset 1 (add to r1..r5)
    let (r1, c) = mac(r1, a[0], b_hi, 0);
    let (r2, c) = mac(r2, a[1], b_hi, c);
    let (r3, c) = mac(r3, a[2], b_hi, c);
    let (r4, c) = mac(r4, a[3], b_hi, c);
    // Add both carries (c from pass 2, c1 from pass 1) into position 5
    let (r5, c) = mac(target.0[5], c1, 1, c);
    target.0[1] = r1;
    target.0[2] = r2;
    target.0[3] = r3;
    target.0[4] = r4;
    target.0[5] = r5;
    // Propagate final carry through remaining limbs (just add)
    target.0[6] = target.0[6].wrapping_add(c);
  }

  #[inline(always)]
  fn reduce(acc: &Self::Accumulator) -> Self {
    // Subtract in limb space first, then reduce once (saves one Barrett reduction)
    match sub_mag::<7>(&acc.pos.0, &acc.neg.0) {
      SubMagResult::Positive(mag) => F::from_limbs(barrett_reduce_7::<F>(&mag)),
      SubMagResult::Negative(mag) => -F::from_limbs(barrett_reduce_7::<F>(&mag)),
    }
  }
}

// ============================================================================
// DelayedReduction<F> - for field × field products
// ============================================================================

/// DelayedReduction<F> for field × field products.
///
/// Uses WideLimbs<9> (576 bits) as accumulator, supporting up to 2^68 products.
///
/// # Capacity Invariant
///
/// The 9th limb (index 8) accumulates carries from the lower 8 limbs. Each
/// field×field product contributes at most 1 to the carry chain into limb 8.
/// With a u64 limb, we can accumulate up to 2^64 products before overflow.
/// In practice, sumcheck rounds are bounded by polynomial size (≤ 2^40),
/// so this limit is never approached. The debug_assert below catches misuse.
impl<F: MontgomeryLimbs + PrimeField + Copy> DelayedReduction<F> for F {
  type Accumulator = WideLimbs<9>;

  #[inline(always)]
  fn unreduced_multiply_accumulate(acc: &mut Self::Accumulator, field_a: &Self, field_b: &F) {
    // Compute field_a × field_b as 8 limbs and add to accumulator
    let product = mul_4_by_4(field_a.to_limbs(), field_b.to_limbs());
    let mut carry = 0u128;
    for (acc_limb, &prod_limb) in acc.0.iter_mut().take(8).zip(product.iter()) {
      let sum = (*acc_limb as u128) + (prod_limb as u128) + carry;
      *acc_limb = sum as u64;
      carry = sum >> 64;
    }

    // Accumulate carry into the 9th limb. Overflow here means we've exceeded
    // the accumulator's capacity (~2^64 products) - this should never happen
    // in valid usage since sumcheck polynomials are bounded by practical sizes.
    let old_limb8 = acc.0[8];
    acc.0[8] = acc.0[8].wrapping_add(carry as u64);
    debug_assert!(
      acc.0[8] >= old_limb8,
      "DelayedReduction accumulator overflow: limb 8 wrapped from {} to {} (carry={}). \
       Too many products accumulated without reduction.",
      old_limb8,
      acc.0[8],
      carry
    );
  }

  #[inline(always)]
  fn reduce(acc: &Self::Accumulator) -> Self {
    F::from_limbs(montgomery_reduce_9::<F>(&acc.0))
  }
}

// =============================================================================
// Test helpers (exported for use by provider test modules)
// =============================================================================

#[cfg(test)]
pub(crate) fn test_delayed_reduction_sum_impl<F: MontgomeryLimbs + PrimeField + Copy>() {
  use rand::{SeedableRng, rngs::StdRng};

  let mut rng = StdRng::seed_from_u64(54321);

  let n = 1000;
  let a_vec: Vec<F> = (0..n).map(|_| F::random(&mut rng)).collect();
  let b_vec: Vec<F> = (0..n).map(|_| F::random(&mut rng)).collect();

  // Compute sum using standard field arithmetic
  let expected: F = a_vec.iter().zip(b_vec.iter()).map(|(a, b)| *a * *b).sum();

  // Compute using delayed reduction
  let mut acc = WideLimbs::<9>::default();
  for (a, b) in a_vec.iter().zip(b_vec.iter()) {
    <F as DelayedReduction<F>>::unreduced_multiply_accumulate(&mut acc, a, b);
  }
  let result = <F as DelayedReduction<F>>::reduce(&acc);

  assert_eq!(
    result, expected,
    "Delayed reduction sum failed: accumulated result != direct sum"
  );
}

#[cfg(test)]
pub(crate) fn test_delayed_reduction_i32_impl<F: MontgomeryLimbs + PrimeField + Copy>() {
  use super::small_value_field::i64_to_field;
  use rand::{Rng, SeedableRng, rngs::StdRng};

  let mut rng = StdRng::seed_from_u64(54321);

  let mut acc = SignedWideLimbs::<6>::default();
  let mut expected = F::ZERO;

  // Sum 100 field × i32 products (mix of positive and negative)
  for i in 0..100 {
    let field = F::random(&mut rng);
    let value_abs: i32 = rng.gen_range(0..=1000);
    let value: i32 = if i % 2 == 0 { value_abs } else { -value_abs };

    <F as DelayedReduction<i32>>::unreduced_multiply_accumulate(&mut acc, &field, &value);
    expected += field * i64_to_field::<F>(value as i64);
  }

  let result = <F as DelayedReduction<i32>>::reduce(&acc);
  assert_eq!(result, expected, "Delayed reduction i32 failed");
}

#[cfg(test)]
pub(crate) fn test_delayed_reduction_i64_impl<F: MontgomeryLimbs + PrimeField + Copy>() {
  use super::small_value_field::i64_to_field;
  use rand::{Rng, SeedableRng, rngs::StdRng};

  let mut rng = StdRng::seed_from_u64(54321);

  let mut acc = SignedWideLimbs::<6>::default();
  let mut expected = F::ZERO;

  // Sum 100 field × i64 products (mix of positive and negative)
  for i in 0..100 {
    let field = F::random(&mut rng);
    let value_abs: i64 = rng.gen_range(0..=100_000i64);
    let value: i64 = if i % 2 == 0 { value_abs } else { -value_abs };

    <F as DelayedReduction<i64>>::unreduced_multiply_accumulate(&mut acc, &field, &value);
    expected += field * i64_to_field::<F>(value);
  }

  let result = <F as DelayedReduction<i64>>::reduce(&acc);
  assert_eq!(result, expected, "Delayed reduction i64 failed");
}

/// Generate tests for `DelayedReduction` implementation.
#[cfg(test)]
#[macro_export]
macro_rules! test_delayed_reduction {
  ($mod_name:ident, $field:ty) => {
    mod $mod_name {
      #[test]
      fn delayed_reduction_sum() {
        $crate::big_num::delayed_reduction::test_delayed_reduction_sum_impl::<$field>();
      }

      #[test]
      fn delayed_reduction_i32() {
        $crate::big_num::delayed_reduction::test_delayed_reduction_i32_impl::<$field>();
      }

      #[test]
      fn delayed_reduction_i64() {
        $crate::big_num::delayed_reduction::test_delayed_reduction_i64_impl::<$field>();
      }
    }
  };
}
