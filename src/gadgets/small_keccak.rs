// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Keccak-256 circuit using the pure-integer small-value constraint system.
//!
//! All witnesses are bits (0/1), all coefficients are in {-1, 0, 1, 2} ⊂ i8.
//! No field elements are created during circuit synthesis.
//!
//! # Operations
//!
//! | Operation | Constraints per bit | Coefficients |
//! |-----------|---------------------|--------------|
//! | XOR       | 1                   | {-1, 1, 2}   |
//! | AND-NOT   | 1                   | {-1, 1}      |
//! | ROT       | 0 (index shuffle)   | —            |
//! | Permute   | 0 (index shuffle)   | —            |

use crate::gadgets::small_boolean::{Double, NegOne, SmallBoolean};
use crate::small_constraint_system::SmallConstraintSystem;
use bellpepper_core::SynthesisError;

/// Keccak-f[1600] round constants (24 rounds).
const ROUND_CONSTANTS: [u64; 24] = [
  0x0000000000000001,
  0x0000000000008082,
  0x800000000000808a,
  0x8000000080008000,
  0x000000000000808b,
  0x0000000080000001,
  0x8000000080008081,
  0x8000000000008009,
  0x000000000000008a,
  0x0000000000000088,
  0x0000000080008009,
  0x000000008000000a,
  0x000000008000808b,
  0x800000000000008b,
  0x8000000000008089,
  0x8000000000008003,
  0x8000000000008002,
  0x8000000000000080,
  0x000000000000800a,
  0x800000008000000a,
  0x8000000080008081,
  0x8000000000008080,
  0x0000000080000001,
  0x8000000080008008,
];

/// Keccak rotation offsets. ROTATION_OFFSETS[x][y] gives the rotation
/// amount for lane (x, y) per FIPS 202 Table 2.
const ROTATION_OFFSETS: [[usize; 5]; 5] = [
  [0, 36, 3, 41, 18],   // x=0
  [1, 44, 10, 45, 2],   // x=1
  [62, 6, 43, 15, 61],  // x=2
  [28, 55, 25, 21, 56], // x=3
  [27, 20, 39, 8, 14],  // x=4
];

/// Index into a flat 1600-bit state array.
/// State is organized as state[x][y][z] where x,y ∈ [0,5), z ∈ [0,64).
#[inline(always)]
fn idx(x: usize, y: usize, z: usize) -> usize {
  64 * (5 * y + x) + z
}

/// Trait bounds for Keccak gadget value types.
pub trait KeccakValue: Copy + From<bool> + NegOne + Double {}
impl<T: Copy + From<bool> + NegOne + Double> KeccakValue for T {}

/// XOR-NOT-AND: compute `a ^ ((!b) & c)` used in chi step.
///
/// Constraint: `(1-b) * c = t` (AND-NOT), then `a ^ t` (XOR).
/// Coefficients: {-1, 0, 1, 2}.
fn xor_not_and<V, CS>(
  cs: &mut CS,
  a: &SmallBoolean,
  b: &SmallBoolean,
  c: &SmallBoolean,
  prefix: &str,
) -> Result<SmallBoolean, SynthesisError>
where
  V: KeccakValue,
  CS: SmallConstraintSystem<V>,
{
  // First compute (!b) & c
  let not_b = b.not();
  let t = SmallBoolean::and(
    cs.namespace(|| format!("{prefix}_andnot")).inner,
    &not_b,
    c,
  )?;
  // Then XOR with a
  SmallBoolean::xor(cs.namespace(|| format!("{prefix}_xor")).inner, a, &t)
}

/// Keccak-f[1600] permutation over SmallBoolean state.
fn keccak_f<V, CS>(
  cs: &mut CS,
  state: &mut Vec<SmallBoolean>,
) -> Result<(), SynthesisError>
where
  V: KeccakValue,
  CS: SmallConstraintSystem<V>,
{
  for round in 0..24 {
    keccak_round::<V, _>(cs, state, round)?;
  }
  Ok(())
}

/// Single Keccak round: θ, ρ, π, χ, ι.
fn keccak_round<V, CS>(
  cs: &mut CS,
  state: &mut Vec<SmallBoolean>,
  round: usize,
) -> Result<(), SynthesisError>
where
  V: KeccakValue,
  CS: SmallConstraintSystem<V>,
{
  // θ step
  theta::<V, _>(cs, state, round)?;
  // ρ and π steps (combined, no constraints)
  rho_pi(state);
  // χ step
  chi::<V, _>(cs, state, round)?;
  // ι step (XOR with round constant, no constraints for constant bits)
  iota(state, round);
  Ok(())
}

/// θ step: column parity XOR.
///
/// C[x][z] = state[x][0][z] ^ state[x][1][z] ^ ... ^ state[x][4][z]
/// D[x][z] = C[x-1][z] ^ C[x+1][z-1]
/// state[x][y][z] ^= D[x][z]
fn theta<V, CS>(
  cs: &mut CS,
  state: &mut Vec<SmallBoolean>,
  round: usize,
) -> Result<(), SynthesisError>
where
  V: KeccakValue,
  CS: SmallConstraintSystem<V>,
{
  // Compute column parity C[x][z]
  let mut c: Vec<Vec<SmallBoolean>> = (0..5)
    .map(|_| (0..64).map(|_| SmallBoolean::Constant(false)).collect())
    .collect();
  for x in 0..5 {
    for z in 0..64 {
      // XOR of 5 values: chain binary XORs
      let mut acc = state[idx(x, 0, z)].clone();
      for y in 1..5 {
        acc = SmallBoolean::xor(
          cs.namespace(|| format!("r{round}_theta_c_{x}_{z}_y{y}")).inner,
          &acc,
          &state[idx(x, y, z)],
        )?;
      }
      c[x][z] = acc;
    }
  }

  // Compute D[x][z] = C[x-1][z] ^ C[x+1][(z-1) mod 64]
  let mut d: Vec<Vec<SmallBoolean>> = (0..5)
    .map(|_| (0..64).map(|_| SmallBoolean::Constant(false)).collect())
    .collect();
  for x in 0..5 {
    let x_minus_1 = (x + 4) % 5;
    let x_plus_1 = (x + 1) % 5;
    for z in 0..64 {
      let z_minus_1 = (z + 63) % 64;
      d[x][z] = SmallBoolean::xor(
        cs.namespace(|| format!("r{round}_theta_d_{x}_{z}")).inner,
        &c[x_minus_1][z],
        &c[x_plus_1][z_minus_1],
      )?;
    }
  }

  // Apply: state[x][y][z] ^= D[x][z]
  for x in 0..5 {
    for y in 0..5 {
      for z in 0..64 {
        state[idx(x, y, z)] = SmallBoolean::xor(
          cs.namespace(|| format!("r{round}_theta_apply_{x}_{y}_{z}")).inner,
          &state[idx(x, y, z)],
          &d[x][z],
        )?;
      }
    }
  }

  Ok(())
}

/// ρ (rotation) and π (permutation) steps combined. No constraints.
fn rho_pi(state: &mut Vec<SmallBoolean>) {
  let mut new_state: Vec<SmallBoolean> = (0..1600)
    .map(|_| SmallBoolean::Constant(false))
    .collect();
  for x in 0..5 {
    for y in 0..5 {
      let new_x = y;
      let new_y = (2 * x + 3 * y) % 5;
      let rot = ROTATION_OFFSETS[x][y];
      for z in 0..64 {
        let src_z = (z + 64 - rot) % 64;
        new_state[idx(new_x, new_y, z)] = state[idx(x, y, src_z)].clone();
      }
    }
  }
  *state = new_state;
}

/// χ step: non-linear operation.
///
/// state[x][y][z] = state[x][y][z] ^ ((!state[x+1][y][z]) & state[x+2][y][z])
fn chi<V, CS>(
  cs: &mut CS,
  state: &mut Vec<SmallBoolean>,
  round: usize,
) -> Result<(), SynthesisError>
where
  V: KeccakValue,
  CS: SmallConstraintSystem<V>,
{
  // Process one row (y-plane) at a time
  for y in 0..5 {
    // Snapshot the row before modification
    let mut row: Vec<SmallBoolean> = (0..320)
      .map(|_| SmallBoolean::Constant(false))
      .collect();
    for x in 0..5 {
      for z in 0..64 {
        row[x * 64 + z] = state[idx(x, y, z)].clone();
      }
    }

    for x in 0..5 {
      let x1 = (x + 1) % 5;
      let x2 = (x + 2) % 5;
      for z in 0..64 {
        state[idx(x, y, z)] = xor_not_and::<V, _>(
          cs,
          &row[x * 64 + z],
          &row[x1 * 64 + z],
          &row[x2 * 64 + z],
          &format!("r{round}_chi_{x}_{y}_{z}"),
        )?;
      }
    }
  }
  Ok(())
}

/// ι step: XOR round constant into lane (0,0). No constraints (constants).
fn iota(state: &mut Vec<SmallBoolean>, round: usize) {
  let rc = ROUND_CONSTANTS[round];
  for z in 0..64 {
    if (rc >> z) & 1 == 1 {
      // XOR with constant true = NOT
      state[idx(0, 0, z)] = state[idx(0, 0, z)].not();
    }
  }
}

/// Compute Keccak-256 hash of arbitrary byte-aligned input using the pure-integer constraint system.
///
/// Input length must be a multiple of 8 bits. Returns 256 output bits.
/// All constraints use coefficients in {-1, 0, 1, 2} ⊂ i8.
pub fn small_keccak256<V, CS>(
  cs: &mut CS,
  input: &[SmallBoolean],
) -> Result<Vec<SmallBoolean>, SynthesisError>
where
  V: KeccakValue,
  CS: SmallConstraintSystem<V>,
{
  assert!(input.len() % 8 == 0, "Keccak-256 input must be byte-aligned");

  let rate = 1088; // Keccak-256 rate in bits
  let input_bits = input.len();

  // Compute padded length: always at least one block beyond the input
  let padded_len = ((input_bits / rate) + 1) * rate;

  // Build padded message: input bits + pad10*1
  let mut padded: Vec<SmallBoolean> = Vec::with_capacity(padded_len);
  padded.extend_from_slice(input);
  padded.resize(padded_len, SmallBoolean::Constant(false));
  // Keccak padding: set bit at input_bits (0x01 LSB) and at last bit of final rate block (0x80 MSB)
  padded[input_bits] = padded[input_bits].not();
  padded[padded_len - 1] = padded[padded_len - 1].not();

  // Initialize 1600-bit state to zero
  let mut state: Vec<SmallBoolean> = (0..1600)
    .map(|_| SmallBoolean::Constant(false))
    .collect();

  let num_blocks = padded_len / rate;
  for block_idx in 0..num_blocks {
    let block_start = block_idx * rate;
    let block = &padded[block_start..block_start + rate];

    if block_idx == 0 {
      // First block: state is all zeros, XOR = identity
      for (i, bit) in block.iter().enumerate() {
        state[i] = bit.clone();
      }
    } else {
      // Subsequent blocks: need real XOR gates for non-constant bits
      for (i, bit) in block.iter().enumerate() {
        if matches!(bit, SmallBoolean::Constant(false)) {
          continue; // XOR with 0 is no-op
        }
        state[i] = SmallBoolean::xor(
          cs.namespace(|| format!("absorb_b{block_idx}_{i}")).inner,
          &state[i],
          bit,
        )?;
      }
    }

    // Apply Keccak-f[1600] permutation
    let mut ns = SmallConstraintSystem::<V>::namespace(cs, || format!("keccak_f_{block_idx}"));
    keccak_f::<V, _>(&mut ns, &mut state)?;
  }

  // Squeeze: output first 256 bits
  Ok(state[..256].to_vec())
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::small_constraint_system::SmallShapeCS;
  use sha3::{Digest, Keccak256};

  /// Convert bytes to SmallBoolean bits (LSB-first per byte, matching Keccak standard).
  fn bytes_to_small_bits_lsb(bytes: &[u8]) -> Vec<SmallBoolean> {
    bytes
      .iter()
      .flat_map(|byte| {
        (0..8).map(move |i| SmallBoolean::constant((byte >> i) & 1 == 1))
      })
      .collect()
  }

  /// Convert SmallBoolean bits (LSB-first per byte) to bytes.
  fn small_bits_to_bytes_lsb(bits: &[SmallBoolean]) -> Vec<u8> {
    assert!(bits.len() % 8 == 0);
    bits
      .chunks(8)
      .map(|chunk| {
        chunk.iter().enumerate().fold(0u8, |acc, (i, bit)| {
          let b = bit.get_value().unwrap_or(false);
          acc | ((b as u8) << i)
        })
      })
      .collect()
  }

  #[test]
  fn test_small_keccak256_shape() {
    let mut cs = SmallShapeCS::<i8>::new();
    let input: Vec<SmallBoolean> = (0..512)
      .map(|_| SmallBoolean::constant(false))
      .collect();
    let hash_bits = small_keccak256::<i8, _>(&mut cs, &input).unwrap();
    assert_eq!(hash_bits.len(), 256);
  }

  #[test]
  fn test_small_keccak256_correctness() {
    // Test various input sizes including single-block and multi-block
    for &size in &[32, 64, 128, 136, 137, 256] {
      let mut cs = SmallShapeCS::<i8>::new();
      let input_bytes: Vec<u8> = (0..size).map(|i| i as u8).collect();
      let input_bits = bytes_to_small_bits_lsb(&input_bytes);
      let hash_bits = small_keccak256::<i8, _>(&mut cs, &input_bits).unwrap();
      let hash_bytes = small_bits_to_bytes_lsb(&hash_bits);

      let expected = Keccak256::digest(&input_bytes);
      assert_eq!(
        &hash_bytes[..],
        &expected[..],
        "Keccak-256 hash mismatch for {size}-byte input"
      );
    }
  }
}
