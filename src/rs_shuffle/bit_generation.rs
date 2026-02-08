//! Bit generation for RS shuffle using Poseidon hash
//!
//! This module handles the generation of pseudorandom bits for the RS shuffle algorithm.
//! It derives deterministic bits from a seed using Poseidon sponge.
//!
//! Provides both native and SNARK circuit implementations.

use super::poseidon::poseidon_config;
use ark_crypto_primitives::sponge::{
    constraints::CryptographicSpongeVar, poseidon::constraints::PoseidonSpongeVar,
    poseidon::PoseidonSponge, Absorb, CryptographicSponge,
};
use ark_ff::{BigInteger, PrimeField};
use ark_r1cs_std::{boolean::Boolean, fields::fp::FpVar, prelude::*};
use ark_relations::r1cs::{ConstraintSystemRef, SynthesisError};

/// Derive split bits from random seed using Poseidon hash (native version)
///
/// Dynamically determines how many field elements are needed based on the field size,
/// draws them from Poseidon, and fills a N×LEVELS matrix.
///
/// # Returns
/// - The bit matrix [[bool; N]; LEVELS]
/// - The number of field element samples that were drawn
pub fn derive_split_bits<F, const N: usize, const LEVELS: usize>(
    seed: F,
) -> ([[bool; N]; LEVELS], usize)
where
    F: PrimeField + Absorb,
{
    // Calculate how many field elements we need
    let field_bits = F::MODULUS_BIT_SIZE as usize;
    let usable_bits_per_element = field_bits.saturating_sub(2); // Trim first and last bits
    let total_bits_needed = N * LEVELS;
    let num_elements_needed =
        (total_bits_needed + usable_bits_per_element - 1) / usable_bits_per_element;

    // Create Poseidon sponge and absorb seed
    let config = poseidon_config::<F>();
    let mut sponge = PoseidonSponge::new(&config);
    sponge.absorb(&seed);

    // Squeeze the required number of field elements
    let random_values: Vec<F> = sponge.squeeze_field_elements(num_elements_needed);

    // Convert field elements to bits and collect into a single stream
    let mut bit_stream = Vec::new();

    for value in random_values.iter() {
        // Get bit decomposition of the field element (LSB first)
        let value_bigint = value.into_bigint();
        let mut value_bits = Vec::with_capacity(F::MODULUS_BIT_SIZE as usize);

        // Extract bits from the BigInteger representation
        for i in 0..(F::MODULUS_BIT_SIZE as usize) {
            value_bits.push(value_bigint.get_bit(i));
        }

        // Trim first and last bits for uniformity
        if value_bits.len() > 2 {
            bit_stream.extend_from_slice(&value_bits[1..value_bits.len() - 1]);
        }
    }

    // Fill the N×LEVELS matrix
    let bit_matrix = std::array::from_fn(|level| {
        std::array::from_fn(|i| {
            let bit_index = level * N + i;
            if bit_index < bit_stream.len() {
                bit_stream[bit_index]
            } else {
                false // Default to false if we somehow run out of bits
            }
        })
    });

    (bit_matrix, num_elements_needed)
}

/// SNARK circuit version: Derive split bits from seed using Poseidon hash
///
/// This is the constraint-generating version that creates R1CS constraints.
///
/// # Arguments
/// * `cs` - The constraint system reference
/// * `seed` - The seed as a field variable (typically a public input)
/// * `num_samples` - The number of field elements to squeeze from Poseidon
///
/// # Returns
/// A N×LEVELS matrix of Boolean circuit variables representing the derived bits
pub fn derive_split_bits_gadget<F, const N: usize, const LEVELS: usize>(
    cs: ConstraintSystemRef<F>,
    seed: &FpVar<F>,
    num_samples: usize,
) -> Result<[[Boolean<F>; N]; LEVELS], SynthesisError>
where
    F: PrimeField + Absorb,
{
    // Create Poseidon sponge in circuit
    let config = poseidon_config::<F>();
    let mut sponge = PoseidonSpongeVar::new(cs.clone(), &config);

    // Absorb the seed
    sponge.absorb(&seed)?;

    // Squeeze the specified number of field elements
    let random_values = sponge.squeeze_field_elements(num_samples)?;

    // Collect bits into a single stream, trimming first and last from each
    let mut bit_stream: Vec<Boolean<F>> = Vec::new();

    for value in random_values.iter() {
        let value_bits: Vec<Boolean<F>> = value.to_bits_le()?;
        let len = value_bits.len();

        // Trim first and last bits
        if len > 2 {
            bit_stream.extend_from_slice(&value_bits[1..len - 1]);
        }
    }

    // Fill the N×LEVELS matrix
    let result = std::array::from_fn(|level| {
        std::array::from_fn(|i| {
            let bit_index = level * N + i;
            if bit_index < bit_stream.len() {
                bit_stream[bit_index].clone()
            } else {
                Boolean::constant(false)
            }
        })
    });

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr as TestField;
    use ark_relations::r1cs::ConstraintSystem;

    const N: usize = 52;
    const LEVELS: usize = 5;

    #[test]
    fn test_derive_split_bits_dimensions() {
        let seed = TestField::from(12345u64);
        let (bits_mat, num_samples) = derive_split_bits::<TestField, N, LEVELS>(seed);

        // Check dimensions
        assert_eq!(bits_mat.len(), LEVELS);
        for level in &bits_mat {
            assert_eq!(level.len(), N);
        }

        // Check that we calculated samples correctly
        assert!(num_samples > 0);
    }

    #[test]
    fn test_derive_split_bits_deterministic() {
        let seed = TestField::from(98765u64);

        // Generate bits twice with same seed
        let (bits_mat1, _) = derive_split_bits::<TestField, N, LEVELS>(seed);
        let (bits_mat2, _) = derive_split_bits::<TestField, N, LEVELS>(seed);

        // Should be identical
        for level in 0..LEVELS {
            for i in 0..N {
                assert_eq!(bits_mat1[level][i], bits_mat2[level][i]);
            }
        }
    }

    #[test]
    fn test_derive_split_bits_circuit_consistency() {
        // Test that circuit and native versions produce the same bits
        let seed = TestField::from(7777u64);

        // Native version
        let (native_bits, num_samples) = derive_split_bits::<TestField, N, LEVELS>(seed);

        // Circuit version
        let cs = ConstraintSystem::<TestField>::new_ref();
        let seed_var =
            FpVar::new_input(cs.clone(), || Ok(seed)).expect("Failed to allocate seed");
        let circuit_bits =
            derive_split_bits_gadget::<TestField, N, LEVELS>(cs.clone(), &seed_var, num_samples)
                .expect("Circuit execution failed");

        // Compare the values
        for level in 0..LEVELS {
            for i in 0..N {
                let native_bit = native_bits[level][i];
                let circuit_bit = circuit_bits[level][i]
                    .value()
                    .expect("Failed to get circuit bit value");
                assert_eq!(
                    native_bit, circuit_bit,
                    "Bit mismatch at level {}, position {}",
                    level, i
                );
            }
        }

        // Verify constraints are satisfied
        assert!(
            cs.is_satisfied().unwrap(),
            "Constraints should be satisfied"
        );
    }
}
