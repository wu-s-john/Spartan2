//! SHA-256 circuit using small-value R1CS.
//!
//! Implements the SHA-256 hash function as a circuit with i32 coefficients
//! and i32 witnesses, using the 2-limb addition technique to keep coefficients bounded.

use super::{boolean::Boolean, uint32::UInt32};
use crate::small_r1cs::{
    BatchingSmallCS, Coefficient, SmallConstraintSystem, SmallMultiEqCS, SmallCS, SynthesisError,
    Witness,
};

/// SHA-256 round constants (first 32 bits of fractional parts of cube roots of first 64 primes).
const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// SHA-256 initial hash values (first 32 bits of fractional parts of square roots of first 8 primes).
const IV: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// Compute SHA-256 hash of input bits.
///
/// Input must be a multiple of 8 bits (bytes).
/// Returns 256 output bits in big-endian order.
pub fn small_sha256<W, C, CS>(
    cs: &mut CS,
    input: &[Boolean<W, C>],
) -> Result<Vec<Boolean<W, C>>, SynthesisError>
where
    W: Witness,
    C: Coefficient + From<i32>,
    CS: SmallConstraintSystem<W, C>,
{
    // Pad the input
    let padded = sha256_padding(input);
    let num_blocks = padded.len() / 512;

    // Initialize hash state with IV
    let mut h: [UInt32<W, C>; 8] = IV.map(UInt32::constant);

    // Process each 512-bit block
    for block_idx in 0..num_blocks {
        let block_start = block_idx * 512;
        let block_bits = &padded[block_start..block_start + 512];
        sha256_compression(cs, &mut h, block_bits)?;
    }

    // Collect output bits (big-endian)
    let mut output = Vec::with_capacity(256);
    for h_i in h {
        let be_bits = h_i.into_bits_be();
        output.extend(be_bits);
    }

    Ok(output)
}

/// SHA-256 compression function for one 512-bit block.
///
/// Updates the hash state `h` in place.
pub fn sha256_compression<W, C, CS>(
    cs: &mut CS,
    h: &mut [UInt32<W, C>; 8],
    block: &[Boolean<W, C>],
) -> Result<(), SynthesisError>
where
    W: Witness,
    C: Coefficient + From<i32>,
    CS: SmallConstraintSystem<W, C>,
{
    assert_eq!(block.len(), 512, "Block must be 512 bits");

    // Parse block into 16 32-bit words
    let mut w: Vec<UInt32<W, C>> = Vec::with_capacity(64);
    for i in 0..16 {
        let bits: [Boolean<W, C>; 32] = std::array::from_fn(|j| {
            // Big-endian: first bit of word is MSB
            block[i * 32 + (31 - j)].clone()
        });
        w.push(UInt32::from_bits_le(bits));
    }

    // Extend to 64 words using message schedule
    for i in 16..64 {
        // w[i] = σ1(w[i-2]) + w[i-7] + σ0(w[i-15]) + w[i-16]
        let s0 = w[i - 15].sha256_sigma0(cs)?;
        let s1 = w[i - 2].sha256_sigma1(cs)?;
        let wi = UInt32::add_many(cs, &[s1, w[i - 7].clone(), s0, w[i - 16].clone()])?;
        w.push(wi);
    }

    // Initialize working variables
    let mut a = h[0].clone();
    let mut b = h[1].clone();
    let mut c = h[2].clone();
    let mut d = h[3].clone();
    let mut e = h[4].clone();
    let mut f = h[5].clone();
    let mut g = h[6].clone();
    let mut hh = h[7].clone();

    // 64 rounds
    for i in 0..64 {
        // T1 = h + Σ1(e) + Ch(e,f,g) + K[i] + W[i]
        let sum1_e = e.sha256_sum1(cs)?;
        let ch_efg = UInt32::sha256_ch(cs, &e, &f, &g)?;
        let k_i = UInt32::constant(K[i]);
        let t1 = UInt32::add_many(cs, &[hh.clone(), sum1_e, ch_efg, k_i, w[i].clone()])?;

        // T2 = Σ0(a) + Maj(a,b,c)
        let sum0_a = a.sha256_sum0(cs)?;
        let maj_abc = UInt32::sha256_maj(cs, &a, &b, &c)?;
        let t2 = UInt32::add_many(cs, &[sum0_a, maj_abc])?;

        // Update working variables
        hh = g;
        g = f;
        f = e;
        e = UInt32::add_many(cs, &[d.clone(), t1.clone()])?;
        d = c;
        c = b;
        b = a;
        a = UInt32::add_many(cs, &[t1, t2])?;
    }

    // Add working variables to hash state
    h[0] = UInt32::add_many(cs, &[h[0].clone(), a])?;
    h[1] = UInt32::add_many(cs, &[h[1].clone(), b])?;
    h[2] = UInt32::add_many(cs, &[h[2].clone(), c])?;
    h[3] = UInt32::add_many(cs, &[h[3].clone(), d])?;
    h[4] = UInt32::add_many(cs, &[h[4].clone(), e])?;
    h[5] = UInt32::add_many(cs, &[h[5].clone(), f])?;
    h[6] = UInt32::add_many(cs, &[h[6].clone(), g])?;
    h[7] = UInt32::add_many(cs, &[h[7].clone(), hh])?;

    Ok(())
}

// ========================================
// Batched SHA-256 (uses BatchingSmallCS<17>)
// ========================================

/// Compute SHA-256 hash with batching for reduced constraint count.
///
/// Uses `BatchingSmallCS<12>` internally with 2-limb addition for optimal
/// i32 coefficient bounds. This achieves significant constraint reduction
/// compared to the non-batched version.
///
/// # Coefficient Bounds
///
/// For 5 operands: ceil(log2(5)) + 1 = 4 carry bits
/// - 2-limb addition: max coefficient 2^(16+3) = 2^19
/// - With BatchingSmallCS<12>: 2^19 × 2^11 = 2^30 < 2^31 ✓
///
/// # Efficiency
///
/// - Non-batched: 2 equality constraints per addition
/// - Batched (K=12): 2/12 ≈ 0.167 equality constraints per addition
/// - Savings: ~83% reduction in equality constraints
pub fn small_sha256_batched<W, C>(
    cs: &mut SmallCS<W, C>,
    input: &[Boolean<W, C>],
) -> Result<Vec<Boolean<W, C>>, SynthesisError>
where
    W: Witness,
    C: Coefficient + From<i32>,
{
    // Pad the input
    let padded = sha256_padding(input);
    let num_blocks = padded.len() / 512;

    // Initialize hash state with IV
    let mut h: [UInt32<W, C>; 8] = IV.map(UInt32::constant);

    // Process each 512-bit block with batching
    {
        let mut batched = BatchingSmallCS::<W, C, 12>::new(cs);

        for block_idx in 0..num_blocks {
            let block_start = block_idx * 512;
            let block_bits = &padded[block_start..block_start + 512];
            sha256_compression_batched(&mut batched, &mut h, block_bits)?;
        }
    } // Drop flushes pending constraints

    // Collect output bits (big-endian)
    let mut output = Vec::with_capacity(256);
    for h_i in h {
        let be_bits = h_i.into_bits_be();
        output.extend(be_bits);
    }

    Ok(output)
}

/// SHA-256 compression function with batching.
///
/// Uses `add_many_batched` which requires a CS implementing `SmallMultiEqCS`.
/// This allows the equality constraints from addition to be batched together.
pub fn sha256_compression_batched<W, C, CS>(
    cs: &mut CS,
    h: &mut [UInt32<W, C>; 8],
    block: &[Boolean<W, C>],
) -> Result<(), SynthesisError>
where
    W: Witness,
    C: Coefficient + From<i32>,
    CS: SmallConstraintSystem<W, C> + SmallMultiEqCS<W, C>,
{
    assert_eq!(block.len(), 512, "Block must be 512 bits");

    // Parse block into 16 32-bit words
    let mut w: Vec<UInt32<W, C>> = Vec::with_capacity(64);
    for i in 0..16 {
        let bits: [Boolean<W, C>; 32] = std::array::from_fn(|j| {
            // Big-endian: first bit of word is MSB
            block[i * 32 + (31 - j)].clone()
        });
        w.push(UInt32::from_bits_le(bits));
    }

    // Extend to 64 words using message schedule
    for i in 16..64 {
        // w[i] = σ1(w[i-2]) + w[i-7] + σ0(w[i-15]) + w[i-16]
        let s0 = w[i - 15].sha256_sigma0(cs)?;
        let s1 = w[i - 2].sha256_sigma1(cs)?;
        let wi = UInt32::add_many_batched(cs, &[s1, w[i - 7].clone(), s0, w[i - 16].clone()])?;
        w.push(wi);
    }

    // Initialize working variables
    let mut a = h[0].clone();
    let mut b = h[1].clone();
    let mut c = h[2].clone();
    let mut d = h[3].clone();
    let mut e = h[4].clone();
    let mut f = h[5].clone();
    let mut g = h[6].clone();
    let mut hh = h[7].clone();

    // 64 rounds
    for i in 0..64 {
        // T1 = h + Σ1(e) + Ch(e,f,g) + K[i] + W[i]
        let sum1_e = e.sha256_sum1(cs)?;
        let ch_efg = UInt32::sha256_ch(cs, &e, &f, &g)?;
        let k_i = UInt32::constant(K[i]);
        let t1 = UInt32::add_many_batched(cs, &[hh.clone(), sum1_e, ch_efg, k_i, w[i].clone()])?;

        // T2 = Σ0(a) + Maj(a,b,c)
        let sum0_a = a.sha256_sum0(cs)?;
        let maj_abc = UInt32::sha256_maj(cs, &a, &b, &c)?;
        let t2 = UInt32::add_many_batched(cs, &[sum0_a, maj_abc])?;

        // Update working variables
        hh = g;
        g = f;
        f = e;
        e = UInt32::add_many_batched(cs, &[d.clone(), t1.clone()])?;
        d = c;
        c = b;
        b = a;
        a = UInt32::add_many_batched(cs, &[t1, t2])?;
    }

    // Add working variables to hash state
    h[0] = UInt32::add_many_batched(cs, &[h[0].clone(), a])?;
    h[1] = UInt32::add_many_batched(cs, &[h[1].clone(), b])?;
    h[2] = UInt32::add_many_batched(cs, &[h[2].clone(), c])?;
    h[3] = UInt32::add_many_batched(cs, &[h[3].clone(), d])?;
    h[4] = UInt32::add_many_batched(cs, &[h[4].clone(), e])?;
    h[5] = UInt32::add_many_batched(cs, &[h[5].clone(), f])?;
    h[6] = UInt32::add_many_batched(cs, &[h[6].clone(), g])?;
    h[7] = UInt32::add_many_batched(cs, &[h[7].clone(), hh])?;

    Ok(())
}

// ========================================
// Optimized SHA-256 with i64 coefficients (uses BatchingSmallCS<21>)
// ========================================

/// Compute SHA-256 hash with i64 coefficients for optimal constraint count.
///
/// Uses `BatchingSmallCS<21>` internally with full 35-bit addition.
/// This matches bellpepper's constraint efficiency.
///
/// # Coefficient Bounds
///
/// For 5 operands: max sum = 5 × (2^32 - 1) ≈ 2^35
/// - Full addition: max coefficient 2^34
/// - With BatchingSmallCS<21>: 2^34 × 2^20 = 2^54 < 2^63 ✓
///
/// # Efficiency
///
/// - Full addition: 1 equality constraint per addition (vs 2 for limbed)
/// - Batched (K=21): 1/21 ≈ 0.048 equality constraints per addition
/// - Fused T2: saves 1 addition per round (64 total)
pub fn small_sha256_batched_i64(
    cs: &mut SmallCS<i32, i64>,
    input: &[Boolean<i32, i64>],
) -> Result<Vec<Boolean<i32, i64>>, SynthesisError> {
    // Pad the input
    let padded = sha256_padding(input);
    let num_blocks = padded.len() / 512;

    // Initialize hash state with IV
    let mut h: [UInt32<i32, i64>; 8] = IV.map(UInt32::constant);

    // Process each 512-bit block with batching
    {
        let mut batched = BatchingSmallCS::<i32, i64, 21>::new(cs);

        for block_idx in 0..num_blocks {
            let block_start = block_idx * 512;
            let block_bits = &padded[block_start..block_start + 512];
            sha256_compression_i64(&mut batched, &mut h, block_bits)?;
        }
    } // Drop flushes pending constraints

    // Collect output bits (big-endian)
    let mut output = Vec::with_capacity(256);
    for h_i in h {
        let be_bits = h_i.into_bits_be();
        output.extend(be_bits);
    }

    Ok(output)
}

/// SHA-256 compression function with i64 coefficients.
///
/// Uses `add_many_full` for single-constraint addition and fuses T2 computation
/// for optimal constraint count.
pub fn sha256_compression_i64<CS>(
    cs: &mut CS,
    h: &mut [UInt32<i32, i64>; 8],
    block: &[Boolean<i32, i64>],
) -> Result<(), SynthesisError>
where
    CS: SmallConstraintSystem<i32, i64> + SmallMultiEqCS<i32, i64>,
{
    assert_eq!(block.len(), 512, "Block must be 512 bits");

    // Parse block into 16 32-bit words
    let mut w: Vec<UInt32<i32, i64>> = Vec::with_capacity(64);
    for i in 0..16 {
        let bits: [Boolean<i32, i64>; 32] = std::array::from_fn(|j| {
            // Big-endian: first bit of word is MSB
            block[i * 32 + (31 - j)].clone()
        });
        w.push(UInt32::from_bits_le(bits));
    }

    // Extend to 64 words using message schedule
    for i in 16..64 {
        // w[i] = σ1(w[i-2]) + w[i-7] + σ0(w[i-15]) + w[i-16]
        let s0 = w[i - 15].sha256_sigma0(cs)?;
        let s1 = w[i - 2].sha256_sigma1(cs)?;
        let wi = UInt32::add_many_full(cs, &[s1, w[i - 7].clone(), s0, w[i - 16].clone()])?;
        w.push(wi);
    }

    // Initialize working variables
    let mut a = h[0].clone();
    let mut b = h[1].clone();
    let mut c = h[2].clone();
    let mut d = h[3].clone();
    let mut e = h[4].clone();
    let mut f = h[5].clone();
    let mut g = h[6].clone();
    let mut hh = h[7].clone();

    // 64 rounds
    for i in 0..64 {
        // T1 = h + Σ1(e) + Ch(e,f,g) + K[i] + W[i]
        let sum1_e = e.sha256_sum1(cs)?;
        let ch_efg = UInt32::sha256_ch(cs, &e, &f, &g)?;
        let k_i = UInt32::constant(K[i]);
        let t1 = UInt32::add_many_full(cs, &[hh.clone(), sum1_e, ch_efg, k_i, w[i].clone()])?;

        // Σ0(a) and Maj(a,b,c) for fused T2
        let sum0_a = a.sha256_sum0(cs)?;
        let maj_abc = UInt32::sha256_maj(cs, &a, &b, &c)?;

        // Update working variables
        hh = g;
        g = f;
        f = e;
        e = UInt32::add_many_full(cs, &[d.clone(), t1.clone()])?;
        d = c;
        c = b;
        b = a;
        // Fused: a = T1 + Σ0(a) + Maj(a,b,c) - saves one addition per round!
        a = UInt32::add_many_full(cs, &[t1, sum0_a, maj_abc])?;
    }

    // Add working variables to hash state
    h[0] = UInt32::add_many_full(cs, &[h[0].clone(), a])?;
    h[1] = UInt32::add_many_full(cs, &[h[1].clone(), b])?;
    h[2] = UInt32::add_many_full(cs, &[h[2].clone(), c])?;
    h[3] = UInt32::add_many_full(cs, &[h[3].clone(), d])?;
    h[4] = UInt32::add_many_full(cs, &[h[4].clone(), e])?;
    h[5] = UInt32::add_many_full(cs, &[h[5].clone(), f])?;
    h[6] = UInt32::add_many_full(cs, &[h[6].clone(), g])?;
    h[7] = UInt32::add_many_full(cs, &[h[7].clone(), hh])?;

    Ok(())
}

/// Apply SHA-256 padding to input bits.
///
/// Padding: append 1 bit, then 0s, then 64-bit length (big-endian).
/// Result length is a multiple of 512 bits.
pub fn sha256_padding<W: Witness, C: Coefficient>(
    input: &[Boolean<W, C>],
) -> Vec<Boolean<W, C>> {
    let msg_len = input.len();

    // Calculate padded length: msg + 1 + k zeros + 64 bits
    // where (msg + 1 + k + 64) % 512 == 0
    let mut padded_len = msg_len + 1 + 64;
    if padded_len % 512 != 0 {
        padded_len += 512 - (padded_len % 512);
    }

    let mut padded = Vec::with_capacity(padded_len);

    // Copy message bits
    padded.extend(input.iter().cloned());

    // Append 1 bit
    padded.push(Boolean::constant(true));

    // Append zeros (k bits)
    let num_zeros = padded_len - msg_len - 1 - 64;
    for _ in 0..num_zeros {
        padded.push(Boolean::constant(false));
    }

    // Append length as 64-bit big-endian
    let len_bits = msg_len as u64;
    for i in (0..64).rev() {
        padded.push(Boolean::constant((len_bits >> i) & 1 == 1));
    }

    assert_eq!(padded.len(), padded_len);
    assert_eq!(padded_len % 512, 0);

    padded
}

/// Convert bytes to boolean bits (big-endian per byte).
pub fn bytes_to_bits<W: Witness, C: Coefficient>(bytes: &[u8]) -> Vec<Boolean<W, C>> {
    let mut bits = Vec::with_capacity(bytes.len() * 8);
    for byte in bytes {
        for i in (0..8).rev() {
            bits.push(Boolean::constant((byte >> i) & 1 == 1));
        }
    }
    bits
}

/// Convert boolean bits to bytes (big-endian per byte).
pub fn bits_to_bytes<W: Witness, C: Coefficient>(bits: &[Boolean<W, C>]) -> Vec<u8> {
    assert_eq!(bits.len() % 8, 0);
    let mut bytes = Vec::with_capacity(bits.len() / 8);

    for chunk in bits.chunks(8) {
        let mut byte = 0u8;
        for (i, bit) in chunk.iter().enumerate() {
            if bit.get_value() == Some(true) {
                byte |= 1 << (7 - i);
            }
        }
        bytes.push(byte);
    }

    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    #[test]
    fn test_sha256_empty() {
        let mut cs = SmallCS::<i32, i32>::new();

        let input: Vec<Boolean<i32, i32>> = vec![];
        let output = small_sha256(&mut cs, &input).unwrap();

        let output_bytes = bits_to_bytes(&output);

        // Compare with native SHA-256
        let expected = Sha256::digest(b"");
        assert_eq!(&output_bytes[..], &expected[..]);
        assert!(cs.is_satisfied::<i64>());
    }

    #[test]
    fn test_sha256_abc() {
        let mut cs = SmallCS::<i32, i32>::new();

        let input = bytes_to_bits::<i32, i32>(b"abc");
        let output = small_sha256(&mut cs, &input).unwrap();

        let output_bytes = bits_to_bytes(&output);

        // Compare with native SHA-256
        let expected = Sha256::digest(b"abc");
        assert_eq!(&output_bytes[..], &expected[..]);
        assert!(cs.is_satisfied::<i64>());

        println!("SHA-256 constraints: {}", cs.num_constraints());
    }

    #[test]
    fn test_sha256_longer() {
        let mut cs = SmallCS::<i32, i32>::new();

        let msg = b"The quick brown fox jumps over the lazy dog";
        let input = bytes_to_bits::<i32, i32>(msg);
        let output = small_sha256(&mut cs, &input).unwrap();

        let output_bytes = bits_to_bytes(&output);

        // Compare with native SHA-256
        let expected = Sha256::digest(msg);
        assert_eq!(&output_bytes[..], &expected[..]);
        assert!(cs.is_satisfied::<i64>());
    }

    #[test]
    fn test_sha256_two_blocks() {
        let mut cs = SmallCS::<i32, i32>::new();

        // 64 bytes = 512 bits, needs two blocks after padding
        let msg = b"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        assert_eq!(msg.len(), 64);

        let input = bytes_to_bits::<i32, i32>(msg);
        let output = small_sha256(&mut cs, &input).unwrap();

        let output_bytes = bits_to_bytes(&output);

        // Compare with native SHA-256
        let expected = Sha256::digest(msg);
        assert_eq!(&output_bytes[..], &expected[..]);
        assert!(cs.is_satisfied::<i64>());
    }

    #[test]
    fn test_padding() {
        // Empty message: 0 bits + 1 + 447 zeros + 64 = 512
        let empty: Vec<Boolean<i32, i32>> = vec![];
        let padded = sha256_padding(&empty);
        assert_eq!(padded.len(), 512);

        // 24 bits (3 bytes): 24 + 1 + 423 zeros + 64 = 512
        let three_bytes = bytes_to_bits::<i32, i32>(b"abc");
        let padded = sha256_padding(&three_bytes);
        assert_eq!(padded.len(), 512);

        // 448 bits (56 bytes): 448 + 1 + 63 zeros + 64 = 576 (not 512!)
        // Actually: 448 + 1 + (512-449) zeros + 64 = 1024 - wait, let me recalculate
        // 448 + 1 + k + 64 where result % 512 == 0
        // 513 + k where (513+k) % 512 == 0
        // k = 512 - 1 = 511, total = 1024
        let fifty_six_bytes = vec![Boolean::<i32, i32>::constant(false); 448];
        let padded = sha256_padding(&fifty_six_bytes);
        assert_eq!(padded.len(), 1024);
    }

    // ========================================
    // Tests for batched SHA-256
    // ========================================

    #[test]
    fn test_sha256_batched_empty() {
        let mut cs = SmallCS::<i32, i32>::new();

        let input: Vec<Boolean<i32, i32>> = vec![];
        let output = small_sha256_batched(&mut cs, &input).unwrap();

        let output_bytes = bits_to_bytes(&output);

        // Compare with native SHA-256
        let expected = Sha256::digest(b"");
        assert_eq!(&output_bytes[..], &expected[..]);
        assert!(cs.is_satisfied::<i64>());
    }

    #[test]
    fn test_sha256_batched_abc() {
        let mut cs = SmallCS::<i32, i32>::new();

        let input = bytes_to_bits::<i32, i32>(b"abc");
        let output = small_sha256_batched(&mut cs, &input).unwrap();

        let output_bytes = bits_to_bytes(&output);

        // Compare with native SHA-256
        let expected = Sha256::digest(b"abc");
        assert_eq!(&output_bytes[..], &expected[..]);
        assert!(cs.is_satisfied::<i64>());

        println!("Batched SHA-256 constraints: {}", cs.num_constraints());
    }

    #[test]
    fn test_sha256_batched_longer() {
        let mut cs = SmallCS::<i32, i32>::new();

        let msg = b"The quick brown fox jumps over the lazy dog";
        let input = bytes_to_bits::<i32, i32>(msg);
        let output = small_sha256_batched(&mut cs, &input).unwrap();

        let output_bytes = bits_to_bytes(&output);

        // Compare with native SHA-256
        let expected = Sha256::digest(msg);
        assert_eq!(&output_bytes[..], &expected[..]);
        assert!(cs.is_satisfied::<i64>());
    }

    #[test]
    fn test_sha256_constraint_count_comparison() {
        // Compare constraint counts between non-batched and batched
        let mut cs_nobatch = SmallCS::<i32, i32>::new();
        let mut cs_batched = SmallCS::<i32, i32>::new();

        let input_nobatch = bytes_to_bits::<i32, i32>(b"abc");
        let input_batched = bytes_to_bits::<i32, i32>(b"abc");

        let _output_nobatch = small_sha256(&mut cs_nobatch, &input_nobatch).unwrap();
        let _output_batched = small_sha256_batched(&mut cs_batched, &input_batched).unwrap();

        let constraints_nobatch = cs_nobatch.num_constraints();
        let constraints_batched = cs_batched.num_constraints();

        println!("Non-batched SHA-256 constraints: {}", constraints_nobatch);
        println!("Batched SHA-256 constraints:     {}", constraints_batched);
        println!(
            "Reduction: {:.1}%",
            100.0 * (1.0 - constraints_batched as f64 / constraints_nobatch as f64)
        );

        // Batched should have significantly fewer constraints
        assert!(
            constraints_batched < constraints_nobatch,
            "Batched should have fewer constraints"
        );
        assert!(cs_nobatch.is_satisfied::<i64>());
        assert!(cs_batched.is_satisfied::<i64>());
    }
}
