//! RS (Rao-Sandelius) Shuffle Gadget Implementation
//!
//! This module implements the RS shuffle algorithm as an arkworks R1CS gadget.
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
//! - Curve pair: BN254 (SNARK) / Grumpkin (ElGamal inside circuit)
//! - Deck size: 52 cards (N)
//! - Shuffle levels: 5 (LEVELS)
//! - Total split bits: 260 (N × LEVELS)

/// Number of ciphertexts (deck size)
pub const N: usize = 52;

/// Depth of shuffle levels
pub const LEVELS: usize = 5;

/// Total number of split bits needed (N * LEVELS)
pub const BITS_NEEDED: usize = N * LEVELS; // 260 split bits total

pub mod bit_generation;
pub mod data_structures;
pub mod encryption;
pub mod gadget;
pub mod native;
pub mod permutation;
pub mod poseidon;

// Re-export main types and functions
pub use data_structures::{
    ElGamalCiphertext, ElGamalCiphertextVar, PermutationWitnessTrace, PermutationWitnessTraceVar,
    SortedRow, SortedRowVar, UnsortedRow, UnsortedRowVar,
};

pub use gadget::{rs_shuffle, rs_shuffle_with_reencryption};

pub use native::{prepare_rs_witness_trace, run_rs_shuffle_permutation, RSShuffleTrace};

pub use encryption::ElGamalEncryption;

pub use poseidon::poseidon_config;
