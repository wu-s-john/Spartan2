/// Montgomery reduction for Pallas base field.
///
/// Converts a value from Montgomery form (a * R mod p) back to standard form (a mod p).
/// This is equivalent to computing (a * R^{-1}) mod p via the REDC algorithm.
///
/// Pallas base field constants:
///   p = 0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001
///   INV = 0x992d30ec00000000 — this is p' such that p * p' ≡ -1 mod 2^64
///        (halo2curves uses -p^{-1} mod 2^64 internally)

const N: usize = 4;

/// Pallas base field modulus as [u64; 4] (little-endian)
const PALLAS_MODULUS: [u64; N] = [
  0x992d30ed00000001,
  0x224698fc094cf91b,
  0x0000000000000000,
  0x4000000000000000,
];

/// Montgomery inverse: -p^{-1} mod 2^64 for Pallas base field
/// Computed: p_inv = pow(p, -1, 2^64) = 0x66d2cf1300000001
///           -p_inv mod 2^64 = 0x992d30ecffffffff
const PALLAS_INV: u64 = 0x992d30ecffffffff;

/// Multiply-and-add with carry: returns (lo, carry) where a + b*c + carry_in = lo + carry*2^64
#[inline(always)]
fn mac(a: u64, b: u64, c: u64, carry: &mut u64) -> u64 {
  let tmp = (a as u128) + (b as u128) * (c as u128) + (*carry as u128);
  *carry = (tmp >> 64) as u64;
  tmp as u64
}

/// Montgomery REDC: convert from Montgomery form to standard form.
/// Input: a value in Montgomery form (4 × u64 limbs, LE)
/// Output: the standard form value (4 × u64 limbs, LE)
pub fn raw_reduction_pallas(a: [u64; N]) -> [u64; N] {
  let mut r = a;

  for i in 0..N {
    let k = r[i].wrapping_mul(PALLAS_INV);
    let mut carry = 0u64;

    mac(r[i], k, PALLAS_MODULUS[0], &mut carry);
    for j in 1..N {
      r[(j + i) % N] = mac(r[(j + i) % N], k, PALLAS_MODULUS[j], &mut carry);
    }
    r[i % N] = carry;
  }

  // Conditional subtraction
  let mut borrow = 0i64;
  let mut result = [0u64; N];
  for i in 0..N {
    let diff = (r[i] as i128) - (PALLAS_MODULUS[i] as i128) - (borrow as i128);
    result[i] = diff as u64;
    borrow = if diff < 0 { 1 } else { 0 };
  }

  if borrow != 0 {
    r // No subtraction needed, r < p
  } else {
    result // r >= p, return r - p
  }
}
