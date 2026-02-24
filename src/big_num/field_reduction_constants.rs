// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT

//! Field-specific constants for modular reduction (Barrett and Montgomery).

use halo2curves::{
  bn256::Fr as Bn254Fr,
  pasta::{Fp, Fq},
  secp256r1::Fq as P256Scalar,
  t256::Fq as T256Fq,
};

// ==========================================================================
// FieldReductionConstants - Trait for field-specific reduction constants
// ==========================================================================

/// Trait providing precomputed constants for efficient modular reduction.
///
/// # Overview
///
/// When reducing a wide integer (more than 4 limbs = 256 bits) modulo a prime p,
/// we need to handle "overflow limbs" that represent values >= 2^256. Each overflow
/// limb at position i represents the value `limb[i] * 2^(64*i)`.
///
/// # The R Constants
///
/// For each bit position beyond 256 bits, we precompute `2^k mod p`:
///
/// | Constant | Value | Used When |
/// |----------|-------|-----------|
/// | `R384_MOD` | 2^384 mod p | Folding 7th limb in Barrett reduction |
/// | `R512_MOD` | 2^512 mod p | Folding 9th limb in Montgomery REDC |
/// | `R_MOD` | 2^256 mod p | Montgomery carry correction |
/// | `BARRETT_MU` | ⌊2^512/p⌋ | True Barrett quotient estimate |
///
/// # Barrett Reduction
///
/// For a 6-limb value x, Barrett reduction computes:
/// 1. q ≈ ⌊x × μ / 2^512⌋ (quotient estimate via reciprocal)
/// 2. r = x - q × p (remainder, at most one correction needed)
///
/// This replaces iterative folding with O(1) operations.
pub trait FieldReductionConstants {
  /// The 4-limb prime modulus p (little-endian, 256 bits)
  const MODULUS: [u64; 4];

  /// 2^384 mod p - reduces the 7th limb (index 6) of a wide integer
  const R384_MOD: [u64; 4];

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
  ///
  /// Used in `montgomery_reduce_8` after the 5th-limb check brings the value below R.
  const MAX_REDC_SUB_CORRECTIONS: usize;

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
  /// True for Pasta Fp/Fq and BN254Fr (p < 2^255).
  /// False for T256Fq (p ≈ 2^256).
  const USE_4_LIMB_BARRETT: bool;

  /// Whether this field has Pasta-style modulus structure.
  ///
  /// True when MODULUS[2] = 0 and MODULUS[3] = 2^62.
  /// This enables specialized Montgomery REDC (2 muls/round instead of 4)
  /// and 2-fold Barrett reduction (using p = 2^254 + c identity).
  ///
  /// True for Pasta Fp/Fq. False for BN254Fr and T256Fq.
  const PASTA_STYLE_MODULUS: bool;

  /// The "c" constant for Pasta primes: p = 2^254 + c where c fits in 2 limbs.
  ///
  /// Used in Pasta 2-fold Barrett reduction: 2^254 ≡ -c (mod p).
  /// Only meaningful when `PASTA_STYLE_MODULUS` is true.
  /// For non-Pasta fields, this is set to [0, 0] and unused.
  const PASTA_C: [u64; 2];
}

// ==========================================================================
// FieldReductionConstants implementation for Fp (Pallas base field)
// ==========================================================================

impl FieldReductionConstants for Fp {
  // p = 0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001
  const MODULUS: [u64; 4] = [
    0x992d30ed00000001,
    0x224698fc094cf91b,
    0x0000000000000000,
    0x4000000000000000,
  ];

  // 2^384 mod p
  const R384_MOD: [u64; 4] = [
    0xcb8792c700000003,
    0x66d3caf41be6eb52,
    0x9b4b3c4bfffffffc,
    0x36e59c0fdacc1b91,
  ];

  // 2^512 mod p
  const R512_MOD: [u64; 4] = [
    0x8c78ecb30000000f,
    0xd7d30dbd8b0de0e7,
    0x7797a99bc3c95d18,
    0x096d41af7b9cb714,
  ];

  // -p^(-1) mod 2^64
  const MONT_INV: u64 = 0x992d30ecffffffff;

  // R mod p = 2^256 mod p (extracted from Fp::ONE.0)
  const R_MOD: [u64; 4] = [
    0x34786d38fffffffd,
    0x992c350be41914ad,
    0xffffffffffffffff,
    0x3fffffffffffffff,
  ];

  // Q = ⌊R/p⌋ = 3 (p ≈ 2^254, so R/p ≈ 4, floor is 3)
  const MAX_REDC_SUB_CORRECTIONS: usize = 3;

  // μ = ⌊2^512 / p⌋ for true Barrett reduction
  const BARRETT_MU: [u64; 5] = [
    0x6d2cf12ffffffff1,
    0xdb96703f6b306e46,
    0xfffffffffffffffd,
    0xffffffffffffffff,
    0x0000000000000003,
  ];

  // p < 2^255, so 2p < 2^256 = b⁴, enabling 4-limb Barrett fast path
  const USE_4_LIMB_BARRETT: bool = true;

  // Pasta-style: MODULUS[2] = 0, MODULUS[3] = 2^62
  const PASTA_STYLE_MODULUS: bool = true;

  // c = MODULUS[0..2] for p = 2^254 + c
  const PASTA_C: [u64; 2] = [0x992d30ed00000001, 0x224698fc094cf91b];
}

// ==========================================================================
// FieldReductionConstants implementation for Fq (Pallas scalar field)
// ==========================================================================

impl FieldReductionConstants for Fq {
  // q = 0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001
  const MODULUS: [u64; 4] = [
    0x8c46eb2100000001,
    0x224698fc0994a8dd,
    0x0000000000000000,
    0x4000000000000000,
  ];

  // 2^384 mod q
  const R384_MOD: [u64; 4] = [
    0xa4d4c16300000003,
    0x66d3caf41cbdfa98,
    0xcee4537bfffffffc,
    0x36e59c0fd9ad5c89,
  ];

  // 2^512 mod q
  const R512_MOD: [u64; 4] = [
    0xfc9678ff0000000f,
    0x67bb433d891a16e3,
    0x7fae231004ccf590,
    0x096d41af7ccfdaa9,
  ];

  // -q^(-1) mod 2^64
  const MONT_INV: u64 = 0x8c46eb20ffffffff;

  // R mod p = 2^256 mod p (extracted from Fq::ONE.0)
  const R_MOD: [u64; 4] = [
    0x5b2b3e9cfffffffd,
    0x992c350be3420567,
    0xffffffffffffffff,
    0x3fffffffffffffff,
  ];

  // Q = ⌊R/p⌋ = 3 (p ≈ 2^254, so R/p ≈ 4, floor is 3)
  const MAX_REDC_SUB_CORRECTIONS: usize = 3;

  // μ = ⌊2^512 / p⌋ for true Barrett reduction
  const BARRETT_MU: [u64; 5] = [
    0x3b914deffffffff1,
    0xdb96703f66b57227,
    0xfffffffffffffffd,
    0xffffffffffffffff,
    0x0000000000000003,
  ];

  // p < 2^255, so 2p < 2^256 = b⁴, enabling 4-limb Barrett fast path
  const USE_4_LIMB_BARRETT: bool = true;

  // Pasta-style: MODULUS[2] = 0, MODULUS[3] = 2^62
  const PASTA_STYLE_MODULUS: bool = true;

  // c = MODULUS[0..2] for p = 2^254 + c
  const PASTA_C: [u64; 2] = [0x8c46eb2100000001, 0x224698fc0994a8dd];
}

// ==========================================================================
// FieldReductionConstants implementation for Bn254Fr (BN254 scalar field)
// ==========================================================================

impl FieldReductionConstants for Bn254Fr {
  // r = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001
  const MODULUS: [u64; 4] = [
    0x43e1f593f0000001,
    0x2833e84879b97091,
    0xb85045b68181585d,
    0x30644e72e131a029,
  ];

  // 2^384 mod r
  const R384_MOD: [u64; 4] = [
    0xb075da81ef8cfeb9,
    0xa7f12acca5b6cd8c,
    0x32c475047957bf7b,
    0x03d581d748ffa25e,
  ];

  // 2^512 mod r
  const R512_MOD: [u64; 4] = [
    0x1bb8e645ae216da7,
    0x53fe3ab1e35c59e3,
    0x8c49833d53bb8085,
    0x0216d0b17f4e44a5,
  ];

  // -r^(-1) mod 2^64
  const MONT_INV: u64 = 0xc2e1f593efffffff;

  // R mod p = 2^256 mod p (extracted from Bn254Fr::ONE.0)
  const R_MOD: [u64; 4] = [
    0xac96341c4ffffffb,
    0x36fc76959f60cd29,
    0x666ea36f7879462e,
    0x0e0a77c19a07df2f,
  ];

  // Q = ⌊R/p⌋ = 5 (p ≈ 0.76 * 2^254, so R/p ≈ 5.26)
  const MAX_REDC_SUB_CORRECTIONS: usize = 5;

  // μ = ⌊2^512 / p⌋ for true Barrett reduction
  const BARRETT_MU: [u64; 5] = [
    0x20703a6be1de9259,
    0x144852009e880ae6,
    0xb074a58680730147,
    0x4a47462623a04a7a,
    0x0000000000000005,
  ];

  // p < 2^255, so 2p < 2^256 = b⁴, enabling 4-limb Barrett fast path
  const USE_4_LIMB_BARRETT: bool = true;

  // Not Pasta-style (dense modulus)
  const PASTA_STYLE_MODULUS: bool = false;

  // Unused for non-Pasta fields
  const PASTA_C: [u64; 2] = [0, 0];
}

// ==========================================================================
// FieldReductionConstants implementation for T256Fq (T256 scalar field)
// ==========================================================================

impl FieldReductionConstants for T256Fq {
  // p = 0xffffffff00000001000000000000000000000000ffffffffffffffffffffffff
  const MODULUS: [u64; 4] = [
    0xffffffffffffffff,
    0x00000000ffffffff,
    0x0000000000000000,
    0xffffffff00000001,
  ];

  // 2^384 mod p
  const R384_MOD: [u64; 4] = [
    0xfffffffefffffffe,
    0x00000002ffffffff,
    0x0000000000000002,
    0xfffffffe00000001,
  ];

  // 2^512 mod p
  const R512_MOD: [u64; 4] = [
    0x0000000000000003,
    0xfffffffbffffffff,
    0xfffffffffffffffe,
    0x00000004fffffffd,
  ];

  // -p^(-1) mod 2^64
  const MONT_INV: u64 = 0x0000000000000001;

  // R mod p = 2^256 mod p (extracted from T256Fq::ONE.0)
  const R_MOD: [u64; 4] = [
    0x0000000000000001,
    0xffffffff00000000,
    0xffffffffffffffff,
    0x00000000fffffffe,
  ];

  // Q = ⌊R/p⌋ = 1 (p ≈ 2^256 - 2^224, so R/p ≈ 1.000...)
  const MAX_REDC_SUB_CORRECTIONS: usize = 1;

  // μ = ⌊2^512 / p⌋ for true Barrett reduction
  const BARRETT_MU: [u64; 5] = [
    0x0000000000000003,
    0xfffffffeffffffff,
    0xfffffffefffffffe,
    0x00000000ffffffff,
    0x0000000000000001,
  ];

  // p ≈ 2^256, so 2p > 2^256 = b⁴, need 5-limb Barrett path
  const USE_4_LIMB_BARRETT: bool = false;

  // Not Pasta-style (different modulus structure)
  const PASTA_STYLE_MODULUS: bool = false;

  // Unused for non-Pasta fields
  const PASTA_C: [u64; 2] = [0, 0];
}

// ==========================================================================
// FieldReductionConstants implementation for P256 scalar (secp256r1::Fq)
// ==========================================================================
// P256Scalar is the GROUP ORDER of secp256r1 (different from T256Fq which
// is the base field order). They form a cycle pair but have different primes.

impl FieldReductionConstants for P256Scalar {
  // n = 0xffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551 (group order)
  const MODULUS: [u64; 4] = [
    0xf3b9cac2fc632551,
    0xbce6faada7179e84,
    0xffffffffffffffff,
    0xffffffff00000000,
  ];

  // 2^384 mod n
  const R384_MOD: [u64; 4] = [
    0xf01cf013fc632551,
    0x679b73e19ad16947,
    0x652e96b7ccca0a99,
    0x431905529c0166ce,
  ];

  // 2^512 mod n
  const R512_MOD: [u64; 4] = [
    0x83244c95be79eea2,
    0x4699799c49bd6fa6,
    0x2845b2392b6bec59,
    0x66e12d94f3d95620,
  ];

  // -n^(-1) mod 2^64
  const MONT_INV: u64 = 0xccd1c8aaee00bc4f;

  // R mod n = 2^256 mod n
  const R_MOD: [u64; 4] = [
    0x0c46353d039cdaaf,
    0x4319055258e8617b,
    0x0000000000000000,
    0x00000000ffffffff,
  ];

  // Q = ⌊R/n⌋ = 1 (n ≈ 2^256 - 2^224, so R/n ≈ 1.000...)
  const MAX_REDC_SUB_CORRECTIONS: usize = 1;

  // μ = ⌊2^512 / n⌋ for Barrett reduction
  const BARRETT_MU: [u64; 5] = [
    0x012ffd85eedf9bfe,
    0x43190552df1a6c21,
    0xfffffffeffffffff,
    0x00000000ffffffff,
    0x0000000000000001,
  ];

  // n ≈ 2^256, so 2n > 2^256 = b⁴, need 5-limb Barrett path
  const USE_4_LIMB_BARRETT: bool = false;

  // Not Pasta-style (different modulus structure)
  const PASTA_STYLE_MODULUS: bool = false;

  // Unused for non-Pasta fields
  const PASTA_C: [u64; 2] = [0, 0];
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
pub(crate) fn test_modulus_impl<F: FieldReductionConstants + PrimeField>() {
  let neg_one = -F::ONE;
  let neg_one_limbs = neg_one.to_repr();
  let mut modulus_minus_one = <F as FieldReductionConstants>::MODULUS;
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
pub(crate) fn test_modulus_bit_length_impl<F: FieldReductionConstants>() {
  let p = limbs_to_biguint(&F::MODULUS);
  assert!(p.bits() >= 254 && p.bits() <= 256);
}

#[cfg(test)]
pub(crate) fn test_mont_inv_impl<F: FieldReductionConstants>() {
  let product = F::MODULUS[0].wrapping_mul(F::MONT_INV);
  assert_eq!(product, u64::MAX);
}

#[cfg(test)]
pub(crate) fn test_r_mod_impl<F: FieldReductionConstants + MontgomeryLimbs + PrimeField>() {
  let one = F::ONE;
  let one_limbs = one.to_limbs();
  assert_eq!(<F as FieldReductionConstants>::R_MOD, *one_limbs);
}

#[cfg(test)]
pub(crate) fn test_r512_mod_direct_impl<F: FieldReductionConstants>() {
  let p = limbs_to_biguint(&F::MODULUS);
  let two_pow_512 = BigUint::from(1u64) << 512;
  let expected = &two_pow_512 % &p;
  let actual = limbs_to_biguint(&F::R512_MOD);
  assert_eq!(actual, expected);
}

#[cfg(test)]
pub(crate) fn test_max_redc_sub_corrections_impl<F: FieldReductionConstants>() {
  let p = limbs_to_biguint(&F::MODULUS);
  let r = BigUint::from(1u64) << 256;
  let expected = &r / &p;
  assert_eq!(BigUint::from(F::MAX_REDC_SUB_CORRECTIONS as u64), expected);
}

/// Generate tests for `FieldReductionConstants` implementation.
#[cfg(test)]
#[macro_export]
macro_rules! test_field_reduction_constants {
  ($mod_name:ident, $field:ty) => {
    mod $mod_name {
      #[test]
      fn modulus() {
        $crate::big_num::field_reduction_constants::test_modulus_impl::<$field>();
      }
      #[test]
      fn modulus_bit_length() {
        $crate::big_num::field_reduction_constants::test_modulus_bit_length_impl::<$field>();
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
