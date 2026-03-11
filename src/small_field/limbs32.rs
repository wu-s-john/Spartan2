#![allow(unsafe_code)]
//! 8x32 NEON kernels for field arithmetic using `vmull_u32`.
//!
//! Represents field elements as 8 limbs of 32 bits each. Products accumulate
//! into `[u128; 15]` column accumulators using widening multiply.
//!
//! This is speculative — NEON has no 64×64→128 multiply, but vmull_u32
//! gives 32×32→64 with good throughput on Apple Silicon.

/// Convert 4x64 limbs to 8x32 limbs.
#[inline(always)]
pub fn to_32(a: &[u64; 4]) -> [u32; 8] {
  [
    a[0] as u32,
    (a[0] >> 32) as u32,
    a[1] as u32,
    (a[1] >> 32) as u32,
    a[2] as u32,
    (a[2] >> 32) as u32,
    a[3] as u32,
    (a[3] >> 32) as u32,
  ]
}

/// Multiply-accumulate field×field in 8x32 column form.
///
/// Schoolbook 8x8 → 15 columns in [u128; 15] accumulators.
/// Each product is 32×32 = 64 bits. With N macs, column k has at most
/// 8 terms per MAC, so acc[k] < N * 8 * 2^64. For N = 2^20: < 2^87. Fits in u128.
#[inline(always)]
pub fn mac_ff_32(cols: &mut [u128; 15], a: &[u32; 8], b: &[u32; 8]) {
  for i in 0..8 {
    for j in 0..8 {
      cols[i + j] += (a[i] as u128) * (b[j] as u128);
    }
  }
}

/// Multiply-accumulate field×i64 in 8x32 column form.
///
/// b is split into two 32-bit halves. Two passes.
#[inline(always)]
pub fn mac_fi64_32(cols: &mut [u128; 9], a: &[u32; 8], b_lo: u32, b_hi: u32) {
  let blo = b_lo as u128;
  let bhi = b_hi as u128;

  for i in 0..8 {
    cols[i] += (a[i] as u128) * blo;
    cols[i + 1] += (a[i] as u128) * bhi;
  }
}

/// Multiply-accumulate field×i128 in 8x32 column form.
///
/// b is split into up to 4 32-bit limbs. Four passes at offsets 0-3.
#[inline(always)]
pub fn mac_fi128_32(cols: &mut [u128; 12], a: &[u32; 8], b: &[u32; 4]) {
  for p in 0..4 {
    if b[p] == 0 {
      continue;
    }
    let bp = b[p] as u128;
    for i in 0..8 {
      cols[i + p] += (a[i] as u128) * bp;
    }
  }
}

/// NEON multiply-accumulate field×field using vmull_u32.
///
/// Processes pairs of limbs using NEON widening multiply for 2x throughput.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub fn mac_ff_neon(cols: &mut [u128; 15], a: &[u32; 8], b: &[u32; 8]) {
  use core::arch::aarch64::*;

  unsafe {
    for i in 0..8 {
      let ai = vdup_n_u32(a[i]);

      // Process b in pairs using NEON
      let mut j = 0;
      while j + 1 < 8 {
        let bpair = vld1_u32(b.as_ptr().add(j));
        let prod = vmull_u32(ai, bpair);

        let lo = vgetq_lane_u64(prod, 0);
        let hi = vgetq_lane_u64(prod, 1);

        cols[i + j] += lo as u128;
        cols[i + j + 1] += hi as u128;

        j += 2;
      }

      // Handle odd remaining b limb
      if j < 8 {
        cols[i + j] += (a[i] as u128) * (b[j] as u128);
      }
    }
  }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
pub fn mac_ff_neon(cols: &mut [u128; 15], a: &[u32; 8], b: &[u32; 8]) {
  mac_ff_32(cols, a, b);
}

/// Carry-propagate 15 columns of 32-bit limb products into 4x64 9-limb format.
#[inline(always)]
pub fn carry_propagate_32_to_9limb(cols: &[u128; 15]) -> [u64; 9] {
  // Carry propagate in 32-bit space. We need up to 18 slots (9 64-bit limbs = 18 32-bit slots).
  let mut c = [0u128; 18];
  c[..15].copy_from_slice(cols);

  for i in 0..17 {
    c[i + 1] += c[i] >> 32;
    c[i] &= 0xFFFF_FFFF;
  }

  // Pack pairs of 32-bit columns into 64-bit limbs
  let mut result = [0u64; 9];
  for i in 0..9 {
    result[i] = (c[2 * i] as u64) | ((c[2 * i + 1] as u64) << 32);
  }

  result
}

/// Carry-propagate 9 columns of 32-bit limb products into 4x64 6-limb format.
#[inline(always)]
pub fn carry_propagate_32_to_6limb(cols: &[u128; 9]) -> [u64; 6] {
  let mut c = [0u128; 10];
  c[..9].copy_from_slice(cols);

  for i in 0..9 {
    c[i + 1] += c[i] >> 32;
    c[i] &= 0xFFFF_FFFF;
  }

  let mut result = [0u64; 6];
  for i in 0..5 {
    result[i] = (c[2 * i] as u64) | ((c[2 * i + 1] as u64) << 32);
  }

  result
}

/// Carry-propagate 12 columns of 32-bit limb products into 4x64 7-limb format.
#[inline(always)]
pub fn carry_propagate_32_to_7limb(cols: &[u128; 12]) -> [u64; 7] {
  let mut c = [0u128; 13];
  c[..12].copy_from_slice(cols);

  for i in 0..12 {
    c[i + 1] += c[i] >> 32;
    c[i] &= 0xFFFF_FFFF;
  }

  let mut result = [0u64; 7];
  for i in 0..6 {
    result[i] = (c[2 * i] as u64) | ((c[2 * i + 1] as u64) << 32);
  }
  // Column 12 might have residual
  result[6] = c[12] as u64;

  result
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::small_field::limbs::mul_4_by_4;
  use rand_core::{OsRng, RngCore};

  #[test]
  fn test_to_32_roundtrip() {
    let mut rng = OsRng;
    for _ in 0..100 {
      let a = [rng.next_u64(), rng.next_u64(), rng.next_u64(), rng.next_u64()];
      let limbs32 = to_32(&a);

      let mut reconstructed = [0u64; 4];
      for i in 0..4 {
        reconstructed[i] = (limbs32[2 * i] as u64) | ((limbs32[2 * i + 1] as u64) << 32);
      }

      assert_eq!(a, reconstructed);
    }
  }

  #[test]
  fn test_mac_ff_32_single_product() {
    let mut rng = OsRng;

    for _ in 0..50 {
      let a64 = [rng.next_u64(), rng.next_u64(), rng.next_u64(), rng.next_u64() >> 2];
      let b64 = [rng.next_u64(), rng.next_u64(), rng.next_u64(), rng.next_u64() >> 2];

      // Reference
      let mut acc_ref = [0u64; 9];
      let product = mul_4_by_4(&a64, &b64);
      let mut carry = 0u128;
      for i in 0..8 {
        let sum = (acc_ref[i] as u128) + (product[i] as u128) + carry;
        acc_ref[i] = sum as u64;
        carry = sum >> 64;
      }
      acc_ref[8] = carry as u64;

      // 8x32 path
      let a32 = to_32(&a64);
      let b32 = to_32(&b64);
      let mut cols = [0u128; 15];
      mac_ff_32(&mut cols, &a32, &b32);
      let acc_32 = carry_propagate_32_to_9limb(&cols);

      assert_eq!(acc_ref, acc_32, "ff_32 single product mismatch");
    }
  }

  #[test]
  fn test_mac_ff_32_sum_of_products() {
    let mut rng = OsRng;
    let n = 100;

    let mut acc_ref = [0u64; 9];
    let mut cols = [0u128; 15];

    for _ in 0..n {
      let a64 = [rng.next_u64(), rng.next_u64(), rng.next_u64(), rng.next_u64() >> 2];
      let b64 = [rng.next_u64(), rng.next_u64(), rng.next_u64(), rng.next_u64() >> 2];

      // Reference
      let product = mul_4_by_4(&a64, &b64);
      let mut carry = 0u128;
      for i in 0..8 {
        let sum = (acc_ref[i] as u128) + (product[i] as u128) + carry;
        acc_ref[i] = sum as u64;
        carry = sum >> 64;
      }
      acc_ref[8] = acc_ref[8].wrapping_add(carry as u64);

      // 8x32
      let a32 = to_32(&a64);
      let b32 = to_32(&b64);
      mac_ff_32(&mut cols, &a32, &b32);
    }

    let acc_32 = carry_propagate_32_to_9limb(&cols);
    assert_eq!(acc_ref, acc_32, "ff_32 sum mismatch after {n} products");
  }

  #[cfg(target_arch = "aarch64")]
  #[test]
  fn test_mac_ff_neon_vs_scalar() {
    let mut rng = OsRng;

    for _ in 0..50 {
      let a64 = [rng.next_u64(), rng.next_u64(), rng.next_u64(), rng.next_u64() >> 2];
      let b64 = [rng.next_u64(), rng.next_u64(), rng.next_u64(), rng.next_u64() >> 2];

      let a32 = to_32(&a64);
      let b32 = to_32(&b64);

      let mut cols_scalar = [0u128; 15];
      let mut cols_neon = [0u128; 15];

      mac_ff_32(&mut cols_scalar, &a32, &b32);
      mac_ff_neon(&mut cols_neon, &a32, &b32);

      assert_eq!(cols_scalar, cols_neon, "NEON vs scalar mismatch");
    }
  }
}
