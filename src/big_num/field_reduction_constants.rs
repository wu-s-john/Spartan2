// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT

//! Field-specific constants for modular reduction.
//!
//! This module defines three independent traits for reduction constants:
//! - [`MontgomeryReductionConstants`] - for Montgomery REDC (all fields)
//! - [`BarrettReductionConstants`] - for generic μ-Barrett reduction (BN254, P256, T256)
//! - [`PastaReductionConstants`] - for Pasta 2-fold Barrett reduction (Pallas, Vesta)
//!
//! Implementations are generated via macros in `macros.rs`.

// ==========================================================================
// MontgomeryReductionConstants - Constants for Montgomery REDC
// ==========================================================================

/// Constants for Montgomery reduction (REDC).
///
/// Used by `montgomery_reduce_9` to reduce 9-limb products to 4-limb field elements.
pub trait MontgomeryReductionConstants {
  /// The 4-limb prime modulus p (little-endian, 256 bits)
  const MODULUS: [u64; 4];

  /// 2^512 mod p - reduces the 9th limb (index 8) of a wide integer
  const R512_MOD: [u64; 4];

  /// Montgomery inverse: -p^(-1) mod 2^64
  /// Used in Montgomery REDC to eliminate low limbs
  const MONT_INV: u64;

  /// R mod p = 2^256 mod p (Montgomery representation of 1)
  /// Used for carry correction after folding: if carry c=1, add R_MOD.
  const R_MOD: [u64; 4];

  /// Q = ⌊R/p⌋, the number of conditional subtracts needed to canonicalize
  /// a value in [0, R) to [0, p).
  const MAX_REDC_SUB_CORRECTIONS: usize;
}

// ==========================================================================
// BarrettReductionConstants - Constants for generic μ-Barrett reduction
// ==========================================================================

/// Constants for generic Barrett reduction.
///
/// Used by `barrett_reduce_6` and `barrett_reduce_7` for BN254, P256, T256.
pub trait BarrettReductionConstants {
  /// The 4-limb prime modulus p (little-endian, 256 bits)
  const MODULUS: [u64; 4];

  /// 2^384 mod p - reduces the 7th limb (index 6) of a wide integer
  const R384_MOD: [u64; 4];

  /// Barrett reciprocal μ = ⌊2^512 / p⌋ (5 limbs).
  ///
  /// Used in true Barrett reduction to compute the quotient estimate:
  /// q ≈ x × μ / 2^512. This allows reducing a 6-limb value to 4 limbs
  /// with exactly one conditional subtract.
  const BARRETT_MU: [u64; 5];

  /// Whether 2p < 2^256, enabling the 4-limb Barrett fast path.
  ///
  /// When true, Barrett remainder r ∈ [0, 2p) fits in 4 limbs, so we can:
  /// - Use `mul_3x4_lo4` instead of `mul_3x4_lo5` (saves 3 multiplications)
  /// - Skip the 5th limb check entirely
  ///
  /// True for BN254Fr (p < 2^255). False for T256Fq (p ≈ 2^256).
  const USE_4_LIMB_BARRETT: bool;
}

// ==========================================================================
// PastaReductionConstants - Constants for Pasta 2-fold Barrett reduction
// ==========================================================================

/// Constants for Pasta 2-fold Barrett reduction.
///
/// For Pasta primes p = 2^254 + PRIME_OFFSET where PRIME_OFFSET fits in 2 limbs.
/// Used by `barrett::pasta::barrett_reduce_6` and `barrett::pasta::barrett_reduce_7`.
///
/// # Key Identity
///
/// Since p = 2^254 + c, we have 2^254 ≡ -c (mod p).
/// This enables efficient 2-fold reduction:
/// - Split x at bit 254: x = x_lo + x_hi × 2^254
/// - Reduce: x ≡ x_lo - x_hi × c (mod p)
pub trait PastaReductionConstants {
  /// The 4-limb prime modulus p (little-endian, 256 bits)
  const MODULUS: [u64; 4];

  /// The offset c where p = 2^254 + c.
  ///
  /// For Pasta primes, MODULUS = [c0, c1, 0, 2^62], so PRIME_OFFSET = [c0, c1].
  const PRIME_OFFSET: [u64; 2];
}

// =============================================================================
// Test helpers (exported for use by provider test modules)
// =============================================================================

#[cfg(test)]
use super::montgomery::MontgomeryLimbs;
#[cfg(test)]
use ff::PrimeField;
#[cfg(test)]
use num_bigint::BigUint;

#[cfg(test)]
fn limbs_to_biguint(limbs: &[u64; 4]) -> BigUint {
  let mut bytes = [0u8; 32];
  for (i, limb) in limbs.iter().enumerate() {
    bytes[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
  }
  BigUint::from_bytes_le(&bytes)
}

#[cfg(test)]
pub(crate) fn test_montgomery_modulus_impl<F: MontgomeryReductionConstants + PrimeField>() {
  let neg_one = -F::ONE;
  let neg_one_limbs = neg_one.to_repr();
  let mut modulus_minus_one = <F as MontgomeryReductionConstants>::MODULUS;
  let (new_val, borrow) = modulus_minus_one[0].overflowing_sub(1);
  modulus_minus_one[0] = new_val;
  assert!(!borrow, "MODULUS should be > 1");
  let mut modulus_bytes = [0u8; 32];
  for (i, limb) in modulus_minus_one.iter().enumerate() {
    modulus_bytes[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
  }
  assert_eq!(neg_one_limbs.as_ref(), &modulus_bytes[..]);
}

#[cfg(test)]
pub(crate) fn test_montgomery_modulus_bit_length_impl<F: MontgomeryReductionConstants>() {
  let p = limbs_to_biguint(&F::MODULUS);
  assert!(p.bits() >= 254 && p.bits() <= 256);
}

#[cfg(test)]
pub(crate) fn test_mont_inv_impl<F: MontgomeryReductionConstants>() {
  let product = F::MODULUS[0].wrapping_mul(F::MONT_INV);
  assert_eq!(product, u64::MAX);
}

#[cfg(test)]
pub(crate) fn test_r_mod_impl<F: MontgomeryReductionConstants + MontgomeryLimbs + PrimeField>() {
  let one = F::ONE;
  let one_limbs = one.to_limbs();
  assert_eq!(<F as MontgomeryReductionConstants>::R_MOD, *one_limbs);
}

#[cfg(test)]
pub(crate) fn test_r512_mod_direct_impl<F: MontgomeryReductionConstants>() {
  let p = limbs_to_biguint(&F::MODULUS);
  let two_pow_512 = BigUint::from(1u64) << 512;
  let expected = &two_pow_512 % &p;
  let actual = limbs_to_biguint(&F::R512_MOD);
  assert_eq!(actual, expected);
}

#[cfg(test)]
pub(crate) fn test_max_redc_sub_corrections_impl<F: MontgomeryReductionConstants>() {
  let p = limbs_to_biguint(&F::MODULUS);
  let r = BigUint::from(1u64) << 256;
  let expected = &r / &p;
  assert_eq!(BigUint::from(F::MAX_REDC_SUB_CORRECTIONS as u64), expected);
}

#[cfg(test)]
fn limbs5_to_biguint(limbs: &[u64; 5]) -> BigUint {
  let mut bytes = [0u8; 40];
  for (i, limb) in limbs.iter().enumerate() {
    bytes[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
  }
  BigUint::from_bytes_le(&bytes)
}

#[cfg(test)]
pub(crate) fn test_barrett_mu_impl<F: BarrettReductionConstants>() {
  let p = limbs_to_biguint(&F::MODULUS);
  let two_pow_512 = BigUint::from(1u64) << 512;
  let expected = &two_pow_512 / &p;
  let actual = limbs5_to_biguint(&F::BARRETT_MU);
  assert_eq!(actual, expected, "BARRETT_MU mismatch");
}

#[cfg(test)]
pub(crate) fn test_r384_mod_impl<F: BarrettReductionConstants>() {
  let p = limbs_to_biguint(&F::MODULUS);
  let two_pow_384 = BigUint::from(1u64) << 384;
  let expected = &two_pow_384 % &p;
  let actual = limbs_to_biguint(&F::R384_MOD);
  assert_eq!(actual, expected, "R384_MOD mismatch");
}

#[cfg(test)]
pub(crate) fn test_pasta_prime_offset_impl<F: PastaReductionConstants>() {
  // Verify p = 2^254 + PRIME_OFFSET
  let p = limbs_to_biguint(&F::MODULUS);
  let two_pow_254 = BigUint::from(1u64) << 254;
  let offset_bytes = [
    F::PRIME_OFFSET[0].to_le_bytes(),
    F::PRIME_OFFSET[1].to_le_bytes(),
  ]
  .concat();
  let offset = BigUint::from_bytes_le(&offset_bytes);
  assert_eq!(p, &two_pow_254 + &offset, "PRIME_OFFSET mismatch");
}

/// Generate tests for `MontgomeryReductionConstants` implementation.
#[cfg(test)]
#[macro_export]
macro_rules! test_montgomery_reduction_constants {
  ($mod_name:ident, $field:ty) => {
    mod $mod_name {
      #[test]
      fn modulus() {
        $crate::big_num::field_reduction_constants::test_montgomery_modulus_impl::<$field>();
      }
      #[test]
      fn modulus_bit_length() {
        $crate::big_num::field_reduction_constants::test_montgomery_modulus_bit_length_impl::<
          $field,
        >();
      }
      #[test]
      fn mont_inv() {
        $crate::big_num::field_reduction_constants::test_mont_inv_impl::<$field>();
      }
      #[test]
      fn r_mod() {
        $crate::big_num::field_reduction_constants::test_r_mod_impl::<$field>();
      }
      #[test]
      fn r512_mod_direct() {
        $crate::big_num::field_reduction_constants::test_r512_mod_direct_impl::<$field>();
      }
      #[test]
      fn max_redc_sub_corrections() {
        $crate::big_num::field_reduction_constants::test_max_redc_sub_corrections_impl::<$field>();
      }
    }
  };
}

/// Generate tests for `BarrettReductionConstants` implementation.
#[cfg(test)]
#[macro_export]
macro_rules! test_barrett_reduction_constants {
  ($mod_name:ident, $field:ty) => {
    mod $mod_name {
      #[test]
      fn barrett_mu() {
        $crate::big_num::field_reduction_constants::test_barrett_mu_impl::<$field>();
      }
      #[test]
      fn r384_mod() {
        $crate::big_num::field_reduction_constants::test_r384_mod_impl::<$field>();
      }
    }
  };
}

/// Generate tests for `PastaReductionConstants` implementation.
#[cfg(test)]
#[macro_export]
macro_rules! test_pasta_reduction_constants {
  ($mod_name:ident, $field:ty) => {
    mod $mod_name {
      #[test]
      fn prime_offset() {
        $crate::big_num::field_reduction_constants::test_pasta_prime_offset_impl::<$field>();
      }
    }
  };
}
