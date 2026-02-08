//! Poseidon hash configuration for bit generation
//!
//! This module provides a standard Poseidon sponge configuration
//! with 128-bit security for deriving shuffle bits from a seed.

use ark_crypto_primitives::sponge::poseidon::{find_poseidon_ark_and_mds, PoseidonConfig};
use ark_ff::PrimeField;

/// Returns a Poseidon sponge configuration with 128-bit security.
///
/// Configuration parameters:
/// - Full rounds: 8
/// - Partial rounds: 57
/// - Alpha (S-box exponent): 5
/// - Rate: 2
/// - Capacity: 1
pub fn poseidon_config<F: PrimeField>() -> PoseidonConfig<F> {
    const FULL_ROUNDS: usize = 8;
    const PARTIAL_ROUNDS: usize = 57;
    const ALPHA: u64 = 5;
    const RATE: usize = 2;
    const CAPACITY: usize = 1;

    let (ark, mds) = find_poseidon_ark_and_mds::<F>(
        F::MODULUS_BIT_SIZE as u64,
        RATE,
        FULL_ROUNDS as u64,
        PARTIAL_ROUNDS as u64,
        0,
    );

    PoseidonConfig {
        full_rounds: FULL_ROUNDS,
        partial_rounds: PARTIAL_ROUNDS,
        alpha: ALPHA,
        ark,
        mds,
        rate: RATE,
        capacity: CAPACITY,
    }
}
