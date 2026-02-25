// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Barrett reduction for wide limb values.
//!
//! This module provides two reduction strategies:
//! - Generic μ-Barrett reduction for BN254, P256, T256 (requires [`BarrettReductionConstants`])
//! - Pasta 2-fold Barrett in the [`pasta`] submodule for Pallas/Vesta (requires [`PastaReductionConstants`])

use super::{
  field_reduction_constants::BarrettReductionConstants,
  limbs::{gte, mul_3x4_lo4, mul_3x4_lo5, mul_3x5_to_8, mul_4_by_1, sub, sub_5_4},
};

// ==========================================================================
// Generic μ-Barrett reduction (for BN254, P256, T256)
// ==========================================================================

/// Barrett reduction for 6-limb input.
///
/// Uses μ-Barrett algorithm with BARRETT_MU reciprocal.
/// For Pasta fields, use [`pasta::barrett_reduce_6`] instead.
#[inline]
pub(crate) fn barrett_reduce_6<F: BarrettReductionConstants>(c: &[u64; 6]) -> [u64; 4] {
  // Step 1: q1 = floor(x / b³) = [c[3], c[4], c[5]]
  let q1 = [c[3], c[4], c[5]];

  // Step 2: q2 = q1 × μ (3×5 → 8 limbs)
  let q2 = mul_3x5_to_8(&q1, &F::BARRETT_MU);

  // Step 3: q3 = floor(q2 / b⁵) = [q2[5], q2[6], q2[7]]
  let q3 = [q2[5], q2[6], q2[7]];

  if F::USE_4_LIMB_BARRETT {
    // Fast path: 4-limb arithmetic for BN254 (where 2p < b⁴)
    // t = q3 × p (low 4 limbs only)
    let t = mul_3x4_lo4(&q3, &F::MODULUS);

    // r = (x mod b⁴) - t (wrapping subtraction in 4 limbs)
    let x_lo4 = [c[0], c[1], c[2], c[3]];
    let mut r = sub::<4>(&x_lo4, &t);

    // One conditional subtract (proven tight)
    if gte::<4>(&r, &F::MODULUS) {
      r = sub::<4>(&r, &F::MODULUS);
    }

    debug_assert!(
      !gte::<4>(&r, &F::MODULUS),
      "Barrett reduction produced non-canonical result"
    );

    r
  } else {
    // 5-limb path for T256 (where 2p can exceed b⁴)
    let r1 = [c[0], c[1], c[2], c[3], c[4]];
    let r2 = mul_3x4_lo5(&q3, &F::MODULUS);
    let mut r = sub::<5>(&r1, &r2);

    // One conditional subtract
    if r[4] != 0 || gte::<4>(&[r[0], r[1], r[2], r[3]], &F::MODULUS) {
      r = sub_5_4(&r, &F::MODULUS);
    }

    debug_assert!(
      r[4] == 0 && !gte::<4>(&[r[0], r[1], r[2], r[3]], &F::MODULUS),
      "Barrett reduction produced non-canonical result"
    );

    [r[0], r[1], r[2], r[3]]
  }
}

/// Barrett reduction for 7-limb input.
///
/// Folds limb 6 using R384_MOD, then delegates to [`barrett_reduce_6`].
/// For Pasta fields, use [`pasta::barrett_reduce_7`] instead.
#[inline]
pub(crate) fn barrett_reduce_7<F: BarrettReductionConstants>(c: &[u64; 7]) -> [u64; 4] {
  // Fold limb 6: c[6] × 2^384 ≡ c[6] × R384_MOD (mod p)
  let c6_contrib = mul_4_by_1(&F::R384_MOD, c[6]);

  // Add c[0..6] + c6_contrib[0..5] using carrying_add
  let (s0, cy) = c[0].carrying_add(c6_contrib[0], false);
  let (s1, cy) = c[1].carrying_add(c6_contrib[1], cy);
  let (s2, cy) = c[2].carrying_add(c6_contrib[2], cy);
  let (s3, cy) = c[3].carrying_add(c6_contrib[3], cy);
  let (s4, cy) = c[4].carrying_add(c6_contrib[4], cy);
  let (s5, _) = c[5].carrying_add(0, cy);

  barrett_reduce_6::<F>(&[s0, s1, s2, s3, s4, s5])
}

// ==========================================================================
// Pasta 2-fold Barrett reduction (for Pallas, Vesta)
// ==========================================================================

/// Pasta-optimized Barrett reduction using the p = 2^254 + c structure.
///
/// For Pasta primes, we exploit: 2^254 ≡ -PRIME_OFFSET (mod p).
/// This enables 2-fold reduction with only ~4-12 multiplies vs ~24 for generic Barrett.
pub mod pasta {
  use super::super::{
    field_reduction_constants::PastaReductionConstants,
    limbs::{add, gte, mul_2_by_1, select, sub_with_borrow},
  };

  /// Barrett reduction for 6-limb input (Pasta 2-fold algorithm).
  ///
  /// For p = 2^254 + PRIME_OFFSET:
  /// - Split x at bit 254: x = x_lo + x_hi × 2^254
  /// - Reduce: x ≡ x_lo - x_hi × PRIME_OFFSET (mod p)
  /// - If negative: add p
  /// - Repeat fold, then final canonicalization
  #[inline]
  pub fn barrett_reduce_6<F: PastaReductionConstants>(c: &[u64; 6]) -> [u64; 4] {
    // For 6 limbs (384 bits), split at bit 254:
    // - x_lo: bits 0-253 (4 limbs, with limb[3] masked to 62 bits)
    // - x_hi: bits 254-383 (up to 130 bits, 3 limbs)

    // Extract x_lo (low 254 bits)
    let x_lo = [c[0], c[1], c[2], c[3] & 0x3FFF_FFFF_FFFF_FFFF]; // mask to 62 bits

    // Extract x_hi (bits 254+): (c[3] >> 62) | (c[4] << 2) | (c[5] << 66)
    // This gives up to 130 bits in 3 limbs
    let x_hi_0 = (c[3] >> 62) | (c[4] << 2);
    let x_hi_1 = (c[4] >> 62) | (c[5] << 2);
    let x_hi_2 = c[5] >> 62;

    // Compute x_hi × PRIME_OFFSET where PRIME_OFFSET is 2 limbs
    // Result is up to 130 + 128 = 258 bits (5 limbs)
    let prod = mul_limbs_by_offset::<F>(x_hi_0, x_hi_1, x_hi_2);

    // Compute x_lo - prod
    // Since prod can be larger than x_lo, we may get a negative result
    // If negative, add p to make positive
    let (result, neg) = sub_wide_4_5(&x_lo, &prod);

    // If negative, add p
    let result = if neg {
      let (sum, _) = add::<4>(&result, &F::MODULUS);
      sum
    } else {
      result
    };

    // Result is now in [0, 2p) or so, need one more fold if still >= 2^254
    // Check if result >= 2^254 (i.e., bit 254 is set)
    if result[3] >= 0x4000_0000_0000_0000 {
      // Need another fold
      let x_lo2 = [
        result[0],
        result[1],
        result[2],
        result[3] & 0x3FFF_FFFF_FFFF_FFFF,
      ];
      let x_hi2 = result[3] >> 62; // at most 2 bits

      // x_hi2 × PRIME_OFFSET (very small multiply)
      let prod2 = mul_1_by_offset::<F>(x_hi2);

      // x_lo2 - prod2
      let (result2, neg2) = sub_4_4_check_neg(&x_lo2, &prod2);
      let result2 = if neg2 {
        let (sum, _) = add::<4>(&result2, &F::MODULUS);
        sum
      } else {
        result2
      };

      // Final canonicalization (branchless)
      let (sub, borrow) = sub_with_borrow::<4>(&result2, &F::MODULUS);
      let out = select::<4>(borrow == 0, &sub, &result2);

      debug_assert!(
        !gte::<4>(&out, &F::MODULUS),
        "Pasta Barrett reduction produced non-canonical result"
      );

      out
    } else {
      // Final canonicalization (branchless)
      let (sub, borrow) = sub_with_borrow::<4>(&result, &F::MODULUS);
      let out = select::<4>(borrow == 0, &sub, &result);

      debug_assert!(
        !gte::<4>(&out, &F::MODULUS),
        "Pasta Barrett reduction produced non-canonical result"
      );

      out
    }
  }

  /// Barrett reduction for 7-limb input (Pasta 2-fold algorithm).
  ///
  /// For inputs up to 448 bits, we need to bound the number of p-additions.
  /// The key insight: fold the product FIRST before subtracting from x_lo.
  ///
  /// Bounds analysis (p = 2^254 + c where c ≈ 2^126):
  /// - prod = x_hi × c: up to 194 + 126 = 320 bits (can exceed 2^256!)
  /// - Fold prod first: prod_hi (66 bits) × c = 192 bits < p
  ///   → prod_folded ∈ [0, 2^254) after at most 1 p-addition
  /// - x_lo ∈ [0, 2^254), prod_folded ∈ [0, 2^254)
  ///   → x_lo - prod_folded ∈ (-2^254, 2^254), needs at most 1 p-addition
  /// - Total: 2 p-additions max (not 2^66!)
  #[inline]
  pub fn barrett_reduce_7<F: PastaReductionConstants>(c: &[u64; 7]) -> [u64; 4] {
    // For 7 limbs (448 bits), split at bit 254:
    // - x_lo: bits 0-253 (4 limbs, with limb[3] masked to 62 bits)
    // - x_hi: bits 254-447 (up to 194 bits, 4 limbs)

    // Extract x_lo (low 254 bits)
    let x_lo = [c[0], c[1], c[2], c[3] & 0x3FFF_FFFF_FFFF_FFFF];

    // Extract x_hi (bits 254+)
    let x_hi_0 = (c[3] >> 62) | (c[4] << 2);
    let x_hi_1 = (c[4] >> 62) | (c[5] << 2);
    let x_hi_2 = (c[5] >> 62) | (c[6] << 2);
    let x_hi_3 = c[6] >> 62;

    // Compute prod = x_hi × PRIME_OFFSET (up to 320 bits / 5 limbs)
    let prod = mul_4limbs_by_offset::<F>(x_hi_0, x_hi_1, x_hi_2, x_hi_3);

    // STEP 1: Fold prod to reduce it below 2p
    // Split prod at bit 254: prod = prod_lo + prod_hi × 2^254
    // prod_hi is at most 66 bits (320 - 254), so prod_hi × PRIME_OFFSET is at most 192 bits
    let prod_lo = [prod[0], prod[1], prod[2], prod[3] & 0x3FFF_FFFF_FFFF_FFFF];
    let prod_hi_0 = (prod[3] >> 62) | (prod[4] << 2);
    let prod_hi_1 = (prod[4] >> 62) | (prod[5] << 2);
    // prod_hi is at most 66 bits, fits in ~2 limbs

    // prod_hi × PRIME_OFFSET: at most 66 + 126 = 192 bits (3 limbs)
    let prod_hi_times_c = mul_2limbs_by_offset::<F>(prod_hi_0, prod_hi_1);

    // prod_folded = prod_lo - prod_hi × c
    // prod_lo ∈ [0, 2^254), prod_hi × c ∈ [0, 2^192)
    // If negative: after adding p, result ∈ (p - 2^192, p) ⊂ [0, 2^254)
    // If non-negative: result ∈ [0, 2^254)
    // Either way: prod_folded ∈ [0, 2^254) after at most 1 p-addition
    let (prod_folded, prod_neg) = sub_4_4_check_neg(&prod_lo, &prod_hi_times_c);
    let prod_folded = if prod_neg {
      let (sum, _) = add::<4>(&prod_folded, &F::MODULUS);
      sum
    } else {
      prod_folded
    };

    // STEP 2: Compute x_lo - prod_folded
    // Both x_lo and prod_folded are in [0, 2^254), so difference is in (-2^254, 2^254).
    // After 1 p-addition, result is in [0, 2^254). The second check is defensive.
    let (result, neg) = sub_4_4_check_neg(&x_lo, &prod_folded);
    let result = if neg {
      let (sum, _) = add::<4>(&result, &F::MODULUS);
      // Defensive: should never trigger since sum ∈ [0, 2^254) < 2^255
      debug_assert!(
        sum[3] < 0x8000_0000_0000_0000,
        "Unexpected: needed second p-addition in barrett_reduce_7"
      );
      sum
    } else {
      result
    };

    // Result is now in [0, ~2p), pass to 6-limb reducer for final canonicalization
    let as_6 = [result[0], result[1], result[2], result[3], 0, 0];
    barrett_reduce_6::<F>(&as_6)
  }

  /// Multiply 2-limb x_hi by PRIME_OFFSET (2 limbs), producing 4-limb result.
  /// Used for folding the product in barrett_reduce_7.
  #[inline(always)]
  fn mul_2limbs_by_offset<F: PastaReductionConstants>(x_hi_0: u64, x_hi_1: u64) -> [u64; 4] {
    let offset = F::PRIME_OFFSET;
    let mut result = [0u64; 4];

    // x_hi_0 × offset
    let prod0 = mul_2_by_1(&offset, x_hi_0);
    result[0] = prod0[0];
    result[1] = prod0[1];
    result[2] = prod0[2];

    // x_hi_1 × offset (add at offset 1)
    let prod1 = mul_2_by_1(&offset, x_hi_1);
    let (r1, c1) = result[1].carrying_add(prod1[0], false);
    let (r2, c2) = result[2].carrying_add(prod1[1], c1);
    let (r3, _) = prod1[2].carrying_add(0, c2);
    result[1] = r1;
    result[2] = r2;
    result[3] = r3;

    result
  }

  /// Multiply 3-limb x_hi by PRIME_OFFSET (2 limbs), producing 5-limb result.
  #[inline(always)]
  fn mul_limbs_by_offset<F: PastaReductionConstants>(
    x_hi_0: u64,
    x_hi_1: u64,
    x_hi_2: u64,
  ) -> [u64; 5] {
    let offset = F::PRIME_OFFSET;
    let mut result = [0u64; 5];

    // x_hi_0 × offset
    let prod0 = mul_2_by_1(&offset, x_hi_0);
    result[0] = prod0[0];
    result[1] = prod0[1];
    result[2] = prod0[2];

    // x_hi_1 × offset (add at offset 1)
    let prod1 = mul_2_by_1(&offset, x_hi_1);
    let (r1, c1) = result[1].carrying_add(prod1[0], false);
    let (r2, c2) = result[2].carrying_add(prod1[1], c1);
    let (r3, c3) = prod1[2].carrying_add(0, c2);
    result[1] = r1;
    result[2] = r2;
    result[3] = r3;
    result[4] = c3 as u64;

    // x_hi_2 × offset (add at offset 2)
    let prod2 = mul_2_by_1(&offset, x_hi_2);
    let (r2, c1) = result[2].carrying_add(prod2[0], false);
    let (r3, c2) = result[3].carrying_add(prod2[1], c1);
    let (r4, _) = result[4].carrying_add(prod2[2], c2);
    result[2] = r2;
    result[3] = r3;
    result[4] = r4;

    result
  }

  /// Multiply 1-limb x_hi by PRIME_OFFSET, producing 4-limb result.
  #[inline(always)]
  fn mul_1_by_offset<F: PastaReductionConstants>(x_hi: u64) -> [u64; 4] {
    let offset = F::PRIME_OFFSET;
    let prod = mul_2_by_1(&offset, x_hi);
    [prod[0], prod[1], prod[2], 0]
  }

  /// Multiply 4-limb x_hi by PRIME_OFFSET (2 limbs), producing 6-limb result.
  #[inline(always)]
  fn mul_4limbs_by_offset<F: PastaReductionConstants>(
    x_hi_0: u64,
    x_hi_1: u64,
    x_hi_2: u64,
    x_hi_3: u64,
  ) -> [u64; 6] {
    let offset = F::PRIME_OFFSET;
    let mut result = [0u64; 6];

    // x_hi_0 × offset
    let prod0 = mul_2_by_1(&offset, x_hi_0);
    result[0] = prod0[0];
    result[1] = prod0[1];
    result[2] = prod0[2];

    // x_hi_1 × offset (add at offset 1)
    let prod1 = mul_2_by_1(&offset, x_hi_1);
    let (r1, c1) = result[1].carrying_add(prod1[0], false);
    let (r2, c2) = result[2].carrying_add(prod1[1], c1);
    let (r3, c3) = prod1[2].carrying_add(0, c2);
    result[1] = r1;
    result[2] = r2;
    result[3] = r3;
    result[4] = c3 as u64;

    // x_hi_2 × offset (add at offset 2)
    let prod2 = mul_2_by_1(&offset, x_hi_2);
    let (r2, c1) = result[2].carrying_add(prod2[0], false);
    let (r3, c2) = result[3].carrying_add(prod2[1], c1);
    let (r4, c3) = result[4].carrying_add(prod2[2], c2);
    result[2] = r2;
    result[3] = r3;
    result[4] = r4;
    result[5] = c3 as u64;

    // x_hi_3 × offset (add at offset 3)
    let prod3 = mul_2_by_1(&offset, x_hi_3);
    let (r3, c1) = result[3].carrying_add(prod3[0], false);
    let (r4, c2) = result[4].carrying_add(prod3[1], c1);
    let (r5, _) = result[5].carrying_add(prod3[2], c2);
    result[3] = r3;
    result[4] = r4;
    result[5] = r5;

    result
  }

  /// Subtract 5-limb from 4-limb: a - b, return (result, is_negative).
  #[inline(always)]
  fn sub_wide_4_5(a: &[u64; 4], b: &[u64; 5]) -> ([u64; 4], bool) {
    let mut result = [0u64; 4];
    let mut borrow = 0u64;

    for i in 0..4 {
      let (diff, b1) = a[i].overflowing_sub(b[i]);
      let (diff2, b2) = diff.overflowing_sub(borrow);
      result[i] = diff2;
      borrow = (b1 as u64) + (b2 as u64);
    }

    // Check if b[4] > 0 or there's remaining borrow
    let is_negative = borrow > 0 || b[4] > 0;
    (result, is_negative)
  }

  /// Subtract two 4-limb values, return (result, is_negative).
  #[inline(always)]
  fn sub_4_4_check_neg(a: &[u64; 4], b: &[u64; 4]) -> ([u64; 4], bool) {
    let (result, borrow) = sub_with_borrow::<4>(a, b);
    (result, borrow > 0)
  }
}

// ==========================================================================
// Test helpers and macro (exported for use by provider test modules)
// ==========================================================================

#[cfg(test)]
use super::montgomery::MontgomeryLimbs;

/// Test that reducing zero gives zero.
#[cfg(test)]
pub(crate) fn test_barrett_zero_impl<R>(reduce_6: R)
where
  R: Fn(&[u64; 6]) -> [u64; 4],
{
  let c = [0u64; 6];
  let result = reduce_6(&c);
  assert_eq!(result, [0, 0, 0, 0]);
}

/// Test single product reduction.
#[cfg(test)]
pub(crate) fn test_barrett_single_product_impl<F, R>(reduce_6: R)
where
  F: ff::Field + ff::PrimeField + MontgomeryLimbs,
  R: Fn(&[u64; 6]) -> [u64; 4],
{
  let field_elem = F::from(12345u64);
  let small = 9999u64;

  let product = mul_4_by_1(field_elem.to_limbs(), small);
  let c = [
    product[0], product[1], product[2], product[3], product[4], 0,
  ];

  let result = F::from_limbs(reduce_6(&c));
  let expected = field_elem * F::from(small);
  assert_eq!(result, expected);
}

/// Test sum of 100 products.
#[cfg(test)]
pub(crate) fn test_barrett_sum_of_products_impl<F, R>(reduce_6: R)
where
  F: ff::Field + ff::PrimeField + MontgomeryLimbs,
  R: Fn(&[u64; 6]) -> [u64; 4],
{
  use rand_core::{OsRng, RngCore};

  let mut rng = OsRng;
  let mut acc = [0u64; 6];
  let mut expected_sum = F::ZERO;

  for _ in 0..100 {
    let field_elem = F::random(&mut rng);
    let small = rng.next_u64() >> 32;

    expected_sum += field_elem * F::from(small);

    let product = mul_4_by_1(field_elem.to_limbs(), small);
    let mut carry = 0u128;
    for i in 0..5 {
      let sum = (acc[i] as u128) + (product[i] as u128) + carry;
      acc[i] = sum as u64;
      carry = sum >> 64;
    }
    acc[5] = acc[5].wrapping_add(carry as u64);
  }

  let result = F::from_limbs(reduce_6(&acc));
  assert_eq!(result, expected_sum);
}

/// Stress test with 2000 products (release builds only).
#[cfg(test)]
#[allow(dead_code)] // Only used in release builds
pub(crate) fn test_barrett_many_products_impl<F, R>(reduce_6: R)
where
  F: ff::Field + ff::PrimeField + MontgomeryLimbs,
  R: Fn(&[u64; 6]) -> [u64; 4],
{
  use rand_core::{OsRng, RngCore};

  let mut rng = OsRng;
  let mut acc = [0u64; 6];
  let mut expected_sum = F::ZERO;

  for _ in 0..2000 {
    let field_elem = F::random(&mut rng);
    let small = rng.next_u64();

    expected_sum += field_elem * F::from(small);

    let product = mul_4_by_1(field_elem.to_limbs(), small);
    let mut carry = 0u128;
    for i in 0..5 {
      let sum = (acc[i] as u128) + (product[i] as u128) + carry;
      acc[i] = sum as u64;
      carry = sum >> 64;
    }
    acc[5] = acc[5].wrapping_add(carry as u64);
  }

  let result = F::from_limbs(reduce_6(&acc));
  assert_eq!(result, expected_sum);
}

/// Generate tests for Barrett reduction functions.
///
/// # Example
/// ```ignore
/// // For generic Barrett (BN254, P256, T256):
/// crate::test_barrett_reduction!(scalar_br, Scalar, crate::big_num::barrett::barrett_reduce_6::<Scalar>);
///
/// // For Pasta Barrett (Pallas, Vesta):
/// crate::test_barrett_reduction!(pallas_br, Fp, crate::big_num::barrett::pasta::barrett_reduce_6::<Fp>);
/// ```
#[cfg(test)]
#[macro_export]
macro_rules! test_barrett_reduction {
  ($mod_name:ident, $field:ty, $reduce_fn:expr) => {
    mod $mod_name {
      #[test]
      fn zero() {
        $crate::big_num::barrett::test_barrett_zero_impl($reduce_fn);
      }
      #[test]
      fn single_product() {
        $crate::big_num::barrett::test_barrett_single_product_impl::<$field, _>($reduce_fn);
      }
      #[test]
      fn sum_of_products() {
        $crate::big_num::barrett::test_barrett_sum_of_products_impl::<$field, _>($reduce_fn);
      }
      #[test]
      #[cfg(not(debug_assertions))]
      fn many_products() {
        $crate::big_num::barrett::test_barrett_many_products_impl::<$field, _>($reduce_fn);
      }
    }
  };
}
