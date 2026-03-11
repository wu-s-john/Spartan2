//! 5x52 column-accumulator kernels for lazy-carry field arithmetic.
//!
//! Represents field elements as 5 limbs of 52 bits each. Products accumulate
//! into `[u128; N]` column accumulators without any carry propagation, deferring
//! all carries to a single propagation pass at reduction time.
//!
//! This eliminates the serial carry chain that limits ILP in the 4x64 representation.

use super::field_reduction_constants::FieldReductionConstants;
use super::montgomery::montgomery_reduce_9;

const MASK52: u64 = (1u64 << 52) - 1;

/// Convert 4x64 limbs to 5x52 limbs.
#[inline(always)]
pub fn to_52(a: &[u64; 4]) -> [u64; 5] {
  [
    a[0] & MASK52,                               // bits 0-51
    ((a[0] >> 52) | (a[1] << 12)) & MASK52,      // bits 52-103
    ((a[1] >> 40) | (a[2] << 24)) & MASK52,      // bits 104-155
    ((a[2] >> 28) | (a[3] << 36)) & MASK52,      // bits 156-207
    a[3] >> 16,                                    // bits 208-255 (at most 48 bits)
  ]
}

/// Multiply-accumulate field×field in 5x52 column form.
///
/// acc[k] += Σ a[i]*b[j] where i+j=k, for k in 0..9.
/// Each a[i], b[j] < 2^52, so each product < 2^104.
/// With N accumulations, acc[k] < N * 5 * 2^104.
/// For N=2^20: acc[k] < 2^20 * 5 * 2^104 < 2^127. Fits in u128.
#[inline(always)]
pub fn mac_ff_52(cols: &mut [u128; 9], a: &[u64; 5], b: &[u64; 5]) {
  // Schoolbook 5x5 → 9 columns
  // Column 0
  cols[0] += (a[0] as u128) * (b[0] as u128);
  // Column 1
  cols[1] += (a[0] as u128) * (b[1] as u128) + (a[1] as u128) * (b[0] as u128);
  // Column 2
  cols[2] += (a[0] as u128) * (b[2] as u128)
    + (a[1] as u128) * (b[1] as u128)
    + (a[2] as u128) * (b[0] as u128);
  // Column 3
  cols[3] += (a[0] as u128) * (b[3] as u128)
    + (a[1] as u128) * (b[2] as u128)
    + (a[2] as u128) * (b[1] as u128)
    + (a[3] as u128) * (b[0] as u128);
  // Column 4
  cols[4] += (a[0] as u128) * (b[4] as u128)
    + (a[1] as u128) * (b[3] as u128)
    + (a[2] as u128) * (b[2] as u128)
    + (a[3] as u128) * (b[1] as u128)
    + (a[4] as u128) * (b[0] as u128);
  // Column 5
  cols[5] += (a[1] as u128) * (b[4] as u128)
    + (a[2] as u128) * (b[3] as u128)
    + (a[3] as u128) * (b[2] as u128)
    + (a[4] as u128) * (b[1] as u128);
  // Column 6
  cols[6] += (a[2] as u128) * (b[4] as u128)
    + (a[3] as u128) * (b[3] as u128)
    + (a[4] as u128) * (b[2] as u128);
  // Column 7
  cols[7] += (a[3] as u128) * (b[4] as u128) + (a[4] as u128) * (b[3] as u128);
  // Column 8
  cols[8] += (a[4] as u128) * (b[4] as u128);
}

/// Multiply-accumulate field×i64 in 5x52 column form.
///
/// b is a single value (up to 64 bits). We treat it as 1 "limb" and get 5 columns.
/// Result accumulates into cols[0..5].
#[inline(always)]
pub fn mac_fi64_52(cols: &mut [u128; 5], a: &[u64; 5], b: u64) {
  let b128 = b as u128;
  cols[0] += (a[0] as u128) * b128;
  cols[1] += (a[1] as u128) * b128;
  cols[2] += (a[2] as u128) * b128;
  cols[3] += (a[3] as u128) * b128;
  cols[4] += (a[4] as u128) * b128;
}

/// Multiply-accumulate field×i128 in 5x52 column form.
///
/// b is split into b_lo (lower 52 bits) and b_hi (remaining bits).
/// When b comes from i64×i64, b < 2^126, so b_hi < 2^74.
/// Two passes: b_lo at offset 0, b_hi at offset 1.
#[inline(always)]
pub fn mac_fi128_52(cols: &mut [u128; 7], a: &[u64; 5], b_lo: u64, b_hi: u64) {
  let blo = b_lo as u128;
  let bhi = b_hi as u128;

  // Pass 1: a[i] * b_lo → cols[i]
  cols[0] += (a[0] as u128) * blo;
  cols[1] += (a[1] as u128) * blo;
  cols[2] += (a[2] as u128) * blo;
  cols[3] += (a[3] as u128) * blo;
  cols[4] += (a[4] as u128) * blo;

  // Pass 2: a[i] * b_hi → cols[i+1]
  cols[1] += (a[0] as u128) * bhi;
  cols[2] += (a[1] as u128) * bhi;
  cols[3] += (a[2] as u128) * bhi;
  cols[4] += (a[3] as u128) * bhi;
  cols[5] += (a[4] as u128) * bhi;
}

/// Multiply-accumulate field×field with explicit local accumulators.
///
/// Same math as `mac_ff_52` but uses local variables instead of array indexing,
/// giving LLVM maximum register allocation freedom.
#[inline(always)]
pub fn mac_ff_52_locals(
  c0: &mut u128,
  c1: &mut u128,
  c2: &mut u128,
  c3: &mut u128,
  c4: &mut u128,
  c5: &mut u128,
  c6: &mut u128,
  c7: &mut u128,
  c8: &mut u128,
  a: &[u64; 5],
  b: &[u64; 5],
) {
  let (a0, a1, a2, a3, a4) = (a[0] as u128, a[1] as u128, a[2] as u128, a[3] as u128, a[4] as u128);
  let (b0, b1, b2, b3, b4) = (b[0] as u128, b[1] as u128, b[2] as u128, b[3] as u128, b[4] as u128);
  *c0 += a0 * b0;
  *c1 += a0 * b1 + a1 * b0;
  *c2 += a0 * b2 + a1 * b1 + a2 * b0;
  *c3 += a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0;
  *c4 += a0 * b4 + a1 * b3 + a2 * b2 + a3 * b1 + a4 * b0;
  *c5 += a1 * b4 + a2 * b3 + a3 * b2 + a4 * b1;
  *c6 += a2 * b4 + a3 * b3 + a4 * b2;
  *c7 += a3 * b4 + a4 * b3;
  *c8 += a4 * b4;
}

/// Standalone field multiplication using the 5×52 column representation.
///
/// Computes a × b mod p for a single field multiply. This is the full pipeline:
/// convert 4×64 → 5×52, schoolbook multiply into columns, carry propagate, Montgomery REDC.
///
/// For a **single** multiply this is slower than halo2curves (~24-26 ns vs ~14 ns)
/// due to conversion and carry propagation overhead. The 5×52 advantage only
/// appears in dot products where N MACs share one reduction pass.
#[inline]
pub fn mul_52<F: FieldReductionConstants>(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
  let a52 = to_52(a);
  let b52 = to_52(b);
  let mut cols = [0u128; 9];
  mac_ff_52(&mut cols, &a52, &b52);
  let wide = carry_propagate_52_to_9limb(&cols);
  montgomery_reduce_9::<F>(&wide)
}

/// Carry-propagate 9 columns of 52-bit limb products into 4x64 9-limb format.
///
/// After accumulation, each column may be up to 128 bits. We propagate carries
/// from the 52-bit boundaries and repack into standard 4x64 format for
/// Montgomery reduction.
#[inline(always)]
pub fn carry_propagate_52_to_9limb(cols: &[u128; 9]) -> [u64; 9] {
  // 576 bits / 52 ≈ 12 slots needed for carry propagation
  let mut c = [0u128; 12];
  c[..9].copy_from_slice(cols);

  // Carry propagate in 52-bit space
  for i in 0..11 {
    c[i + 1] += c[i] >> 52;
    c[i] &= MASK52 as u128;
  }

  // Pack 52-bit limbs into 64-bit limbs
  let mut result = [0u64; 9];
  let mut bit_acc: u128 = 0;
  let mut bit_pos: u32 = 0;
  let mut out_idx = 0;

  for i in 0..12 {
    bit_acc |= c[i] << bit_pos;
    bit_pos += 52;

    while bit_pos >= 64 && out_idx < 9 {
      result[out_idx] = bit_acc as u64;
      bit_acc >>= 64;
      bit_pos -= 64;
      out_idx += 1;
    }
  }

  if out_idx < 9 {
    result[out_idx] = bit_acc as u64;
  }

  result
}

/// Carry-propagate 5 columns of 52-bit limb products into 4x64 6-limb format.
///
/// For field×i64 accumulator → 6-limb Barrett reduction input.
/// After many accumulations, columns can be large (>52 bits), so we need
/// enough carry-propagation slots to handle the full value.
/// Max value: ~2^328 (1M terms × 2^254 × 2^64) → 7 52-bit limbs.
#[inline(always)]
pub fn carry_propagate_52_to_6limb(cols: &[u128; 5]) -> [u64; 6] {
  // Need enough 52-bit slots: 384 bits / 52 ≈ 8 slots
  let mut c = [0u128; 8];
  c[..5].copy_from_slice(cols);

  // Carry propagate in 52-bit space
  for i in 0..7 {
    c[i + 1] += c[i] >> 52;
    c[i] &= MASK52 as u128;
  }

  // Pack 52-bit limbs into 64-bit limbs
  let mut result = [0u64; 6];
  let mut bit_acc: u128 = 0;
  let mut bit_pos: u32 = 0;
  let mut out_idx = 0;

  for i in 0..8 {
    bit_acc |= c[i] << bit_pos;
    bit_pos += 52;

    while bit_pos >= 64 && out_idx < 6 {
      result[out_idx] = bit_acc as u64;
      bit_acc >>= 64;
      bit_pos -= 64;
      out_idx += 1;
    }
  }

  if out_idx < 6 {
    result[out_idx] = bit_acc as u64;
  }

  result
}

/// Carry-propagate 7 columns of 52-bit limb products into 4x64 7-limb format.
///
/// For field×i128 accumulator → 7-limb Barrett reduction input.
#[inline(always)]
pub fn carry_propagate_52_to_7limb(cols: &[u128; 7]) -> [u64; 7] {
  // 448 bits / 52 ≈ 9 slots needed
  let mut c = [0u128; 10];
  c[..7].copy_from_slice(cols);

  // Carry propagate in 52-bit space
  for i in 0..9 {
    c[i + 1] += c[i] >> 52;
    c[i] &= MASK52 as u128;
  }

  // Pack 52-bit limbs into 64-bit limbs
  let mut result = [0u64; 7];
  let mut bit_acc: u128 = 0;
  let mut bit_pos: u32 = 0;
  let mut out_idx = 0;

  for i in 0..10 {
    bit_acc |= c[i] << bit_pos;
    bit_pos += 52;

    while bit_pos >= 64 && out_idx < 7 {
      result[out_idx] = bit_acc as u64;
      bit_acc >>= 64;
      bit_pos -= 64;
      out_idx += 1;
    }
  }

  if out_idx < 7 {
    result[out_idx] = bit_acc as u64;
  }

  result
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::small_field::limbs::mul_4_by_4;
  use rand_core::{OsRng, RngCore};

  #[test]
  fn test_to_52_roundtrip() {
    let mut rng = OsRng;
    for _ in 0..100 {
      // Use values that fit in 254 bits (BN254 Fr)
      let a = [rng.next_u64(), rng.next_u64(), rng.next_u64(), rng.next_u64() >> 2];
      let limbs52 = to_52(&a);

      // Reconstruct from 52-bit limbs
      let mut reconstructed = [0u64; 4];
      let mut bit_acc: u128 = 0;
      let mut bit_pos = 0u32;
      let mut out_idx = 0;

      for &limb in &limbs52 {
        bit_acc |= (limb as u128) << bit_pos;
        bit_pos += 52;

        while bit_pos >= 64 && out_idx < 4 {
          reconstructed[out_idx] = bit_acc as u64;
          bit_acc >>= 64;
          bit_pos -= 64;
          out_idx += 1;
        }
      }

      assert_eq!(a, reconstructed, "52-bit roundtrip failed");
    }
  }

  #[test]
  fn test_mac_ff_52_vs_reference() {
    let mut rng = OsRng;

    for _ in 0..50 {
      let a64 = [rng.next_u64(), rng.next_u64(), rng.next_u64(), rng.next_u64() >> 2];
      let b64 = [rng.next_u64(), rng.next_u64(), rng.next_u64(), rng.next_u64() >> 2];

      // Reference: 4x64 multiply + accumulate
      let mut acc_ref = [0u64; 9];
      let product = mul_4_by_4(&a64, &b64);
      let mut carry = 0u128;
      for i in 0..8 {
        let sum = (acc_ref[i] as u128) + (product[i] as u128) + carry;
        acc_ref[i] = sum as u64;
        carry = sum >> 64;
      }
      acc_ref[8] = carry as u64;

      // 5x52 path
      let a52 = to_52(&a64);
      let b52 = to_52(&b64);
      let mut cols = [0u128; 9];
      mac_ff_52(&mut cols, &a52, &b52);
      let acc_52 = carry_propagate_52_to_9limb(&cols);

      assert_eq!(acc_ref, acc_52, "ff_52 mismatch");
    }
  }

  #[test]
  fn test_mac_ff_52_sum_of_products() {
    let mut rng = OsRng;
    let n = 100;

    let mut acc_ref = [0u64; 9];
    let mut cols = [0u128; 9];

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

      // 5x52
      let a52 = to_52(&a64);
      let b52 = to_52(&b64);
      mac_ff_52(&mut cols, &a52, &b52);
    }

    let acc_52 = carry_propagate_52_to_9limb(&cols);
    assert_eq!(acc_ref, acc_52, "ff_52 sum mismatch after {n} products");
  }

  #[test]
  fn test_mul_52_vs_halo2curves() {
    use crate::small_field::montgomery::MontgomeryLimbs;
    use ff::Field;
    use halo2curves::bn256::Fr as Bn254Fr;

    let mut rng = OsRng;
    for _ in 0..100 {
      let a = Bn254Fr::random(&mut rng);
      let b = Bn254Fr::random(&mut rng);

      let expected = a * b;
      let result = mul_52::<Bn254Fr>(a.to_limbs(), b.to_limbs());

      assert_eq!(result, *expected.to_limbs(), "mul_52 mismatch");
    }
  }

  #[test]
  fn test_mac_fi64_52_vs_reference() {
    let mut rng = OsRng;

    for _ in 0..100 {
      let a64 = [rng.next_u64(), rng.next_u64(), rng.next_u64(), rng.next_u64() >> 2];
      let b = rng.next_u64();

      // Reference: 4x1 multiply + accumulate into 6 limbs
      let mut acc_ref = [0u64; 6];
      let mut carry = 0u128;
      for i in 0..4 {
        let prod = (a64[i] as u128) * (b as u128) + carry;
        acc_ref[i] = prod as u64;
        carry = prod >> 64;
      }
      acc_ref[4] = carry as u64;

      // 5x52 path
      let a52 = to_52(&a64);
      let mut cols = [0u128; 5];
      mac_fi64_52(&mut cols, &a52, b);
      let acc_52 = carry_propagate_52_to_6limb(&cols);

      assert_eq!(acc_ref, acc_52, "fi64_52 mismatch");
    }
  }
}
