//! RS (Rao-Sandelius) Shuffle Gadget Implementation (Bellpepper version)
//!
//! This module implements the RS shuffle algorithm as bellpepper R1CS gadgets.
//! The shuffle is a stable-partition shuffle with bucket-local constraints
//! and grand-product permutation checks.
//!
//! # Overview
//!
//! The RS shuffle proves that a shuffler has correctly shuffled and re-encrypted
//! a deck of ElGamal ciphertexts. The proof demonstrates:
//!
//! 1. **Bit derivation**: Random bits are deterministically derived from a public seed
//! 2. **Stable partition**: Each level partitions elements by their bit while preserving order
//! 3. **Permutation validity**: Grand product checks ensure the shuffle is a valid permutation
//! 4. **Re-encryption**: Output ciphertexts are valid re-encryptions of shuffled inputs
//!
//! # Configuration
//!
//! - Curve pair: Pallas/Vesta (bellpepper-native)
//! - Deck size: 52 cards (N)
//! - Shuffle levels: 5 (LEVELS)
//! - Total split bits: 260 (N × LEVELS)

/// Number of ciphertexts (deck size)
pub const N: usize = 52;

/// Depth of shuffle levels
pub const LEVELS: usize = 5;

/// Total number of split bits needed (N * LEVELS)
pub const BITS_NEEDED: usize = N * LEVELS; // 260 split bits total

pub mod data_structures;
pub mod encryption;
pub mod native;
pub mod permutation;

// Re-export main types
pub use data_structures::{
  ElGamalCiphertext, ElGamalCiphertextVar, PermutationWitnessTrace, PermutationWitnessTraceVar,
  SortedRow, SortedRowVar, UnsortedRow, UnsortedRowVar,
};

pub use encryption::{
  native_reencrypt_parallel, reencrypt_deck_bp, rerandomize_ciphertext_bp,
  NativeReencryptionData,
};
pub use native::{prepare_rs_witness_trace, run_rs_shuffle_permutation, RSShuffleTrace};
