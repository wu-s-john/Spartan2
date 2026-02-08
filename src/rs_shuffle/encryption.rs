//! ElGamal encryption gadgets for RS shuffle
//!
//! This module provides circuit gadgets for ElGamal re-encryption,
//! which is used to make the shuffle unlinkable.

use super::data_structures::ElGamalCiphertextVar;
use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use ark_r1cs_std::{fields::fp::FpVar, groups::CurveVar, prelude::ToBitsGadget};
use ark_relations::r1cs::{ConstraintSystemRef, SynthesisError};

/// ElGamal encryption operations for circuit use
pub struct ElGamalEncryption<C: CurveGroup> {
    /// Precomputed powers of the generator for efficient fixed-base scalar multiplication
    /// generator_powers[i] = 2^i · G
    pub generator_powers: Vec<C>,
}

impl<C: CurveGroup> ElGamalEncryption<C>
where
    C::BaseField: PrimeField,
{
    /// Create a new ElGamalEncryption instance with precomputed generator powers
    ///
    /// # Arguments
    /// * `generator_powers` - Precomputed powers [G, 2G, 4G, 8G, ...] for the bit length of the field
    pub fn new(generator_powers: Vec<C>) -> Self {
        Self { generator_powers }
    }

    /// Precompute generator powers for a given field size
    ///
    /// Returns [G, 2G, 4G, 8G, ...] for each bit position
    pub fn precompute_generator_powers() -> Vec<C> {
        let num_bits = C::BaseField::MODULUS_BIT_SIZE as usize;
        let mut powers = Vec::with_capacity(num_bits);
        let mut current = C::generator();

        for _ in 0..num_bits {
            powers.push(current);
            current = current.double();
        }

        powers
    }

    /// Re-randomize a ciphertext in-circuit
    ///
    /// Computes: (c1 + r·G, c2 + r·PK) from (c1, c2)
    ///
    /// # Arguments
    /// * `cs` - Constraint system reference
    /// * `ciphertext` - The input ciphertext to re-randomize
    /// * `rerandomization` - The randomization scalar r
    /// * `public_key` - The recipient's public key PK
    ///
    /// # Returns
    /// The re-randomized ciphertext
    pub fn rerandomize_ciphertext<CV>(
        &self,
        _cs: ConstraintSystemRef<C::BaseField>,
        ciphertext: &ElGamalCiphertextVar<C, CV>,
        rerandomization: &FpVar<C::BaseField>,
        public_key: &CV,
    ) -> Result<ElGamalCiphertextVar<C, CV>, SynthesisError>
    where
        CV: CurveVar<C, C::BaseField>,
    {
        // Convert randomization to bits for scalar multiplication
        let r_bits = rerandomization.to_bits_le()?;

        // Fixed-base multiplication: r · G using precomputed powers
        let mut r_g = CV::zero();
        r_g.precomputed_base_scalar_mul_le(r_bits.iter().zip(&self.generator_powers))?;

        // Variable-base multiplication: r · PK
        let r_pk = public_key.scalar_mul_le(r_bits.iter())?;

        // c1' = c1 + r·G
        let c1_prime = ciphertext.c1.clone() + r_g;

        // c2' = c2 + r·PK
        let c2_prime = ciphertext.c2.clone() + r_pk;

        Ok(ElGamalCiphertextVar::new(c1_prime, c2_prime))
    }

    /// Re-encrypt a deck of cards with new randomization values
    ///
    /// # Arguments
    /// * `cs` - Constraint system reference
    /// * `input_deck` - The shuffled ciphertexts before re-encryption
    /// * `encryption_randomizations` - Random scalars for each ciphertext
    /// * `shuffler_pk` - The shuffler's public key
    ///
    /// # Returns
    /// Vector of re-encrypted ciphertexts
    pub fn reencrypt_deck<CV, const N: usize>(
        &self,
        cs: ConstraintSystemRef<C::BaseField>,
        input_deck: &[ElGamalCiphertextVar<C, CV>; N],
        encryption_randomizations: &[FpVar<C::BaseField>; N],
        shuffler_pk: &CV,
    ) -> Result<[ElGamalCiphertextVar<C, CV>; N], SynthesisError>
    where
        CV: CurveVar<C, C::BaseField>,
    {
        let results: Vec<ElGamalCiphertextVar<C, CV>> = input_deck
            .iter()
            .zip(encryption_randomizations.iter())
            .map(|(ct, r)| self.rerandomize_ciphertext(cs.clone(), ct, r, shuffler_pk))
            .collect::<Result<Vec<_>, _>>()?;

        // Convert Vec to array
        results
            .try_into()
            .map_err(|_| SynthesisError::Unsatisfiable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rs_shuffle::data_structures::ElGamalCiphertext;
    use ark_bn254::Fr as BaseField;
    use ark_ec::{CurveConfig, PrimeGroup};
    use ark_ff::{BigInteger, UniformRand};
    use ark_grumpkin::{GrumpkinConfig, Projective as GrumpkinProjective};
    use ark_r1cs_std::alloc::AllocVar;
    use ark_r1cs_std::groups::curves::short_weierstrass::ProjectiveVar;
    use ark_r1cs_std::prelude::AllocationMode;
    use ark_r1cs_std::R1CSVar;
    use ark_relations::r1cs::ConstraintSystem;

    type GrumpkinVar = ProjectiveVar<GrumpkinConfig, FpVar<BaseField>>;

    #[test]
    fn test_rerandomize_ciphertext() {
        let mut rng = ark_std::test_rng();
        let cs = ConstraintSystem::<BaseField>::new_ref();

        // Generate keys
        let sk = <GrumpkinConfig as CurveConfig>::ScalarField::rand(&mut rng);
        let pk = GrumpkinProjective::generator() * sk;

        // Create a ciphertext
        let message = <GrumpkinConfig as CurveConfig>::ScalarField::from(42u64);
        let randomness = <GrumpkinConfig as CurveConfig>::ScalarField::rand(&mut rng);
        let ct = ElGamalCiphertext::encrypt_scalar(message, randomness, pk);

        // Precompute generator powers
        let generator_powers = ElGamalEncryption::<GrumpkinProjective>::precompute_generator_powers();
        let elgamal = ElGamalEncryption::new(generator_powers);

        // Allocate circuit variables
        let ct_var = ElGamalCiphertextVar::<GrumpkinProjective, GrumpkinVar>::new_variable(
            cs.clone(),
            || Ok(&ct),
            AllocationMode::Witness,
        )
        .expect("Failed to allocate ciphertext");

        let pk_var = GrumpkinVar::new_variable(cs.clone(), || Ok(pk), AllocationMode::Input)
            .expect("Failed to allocate public key");

        // Rerandomization value - use a random BaseField element directly
        // For the circuit, we represent the scalar in the circuit field (BaseField = BN254 Fr)
        // which equals Grumpkin's base field. The bit decomposition works the same in both.
        let rerand_base = BaseField::rand(&mut rng);
        let rerand_var = FpVar::new_witness(cs.clone(), || Ok(rerand_base))
            .expect("Failed to allocate rerandomization");

        // For native computation, convert to Grumpkin's scalar field
        // Both fields have ~254 bits, so we can safely convert via bytes
        use ark_ff::PrimeField as _;
        let rerand_scalar = <GrumpkinConfig as CurveConfig>::ScalarField::from_le_bytes_mod_order(
            &rerand_base.into_bigint().to_bytes_le(),
        );

        // Perform rerandomization in circuit
        let ct_rerand = elgamal
            .rerandomize_ciphertext(cs.clone(), &ct_var, &rerand_var, &pk_var)
            .expect("Rerandomization failed");

        // Compute expected result natively
        let expected = ct.add_encryption_layer(rerand_scalar, &pk);

        // Verify the result matches
        let c1_result = ct_rerand.c1.value().expect("Failed to get c1 value");
        let c2_result = ct_rerand.c2.value().expect("Failed to get c2 value");

        assert_eq!(c1_result, expected.c1, "c1 mismatch");
        assert_eq!(c2_result, expected.c2, "c2 mismatch");

        // Verify constraint satisfaction
        assert!(cs.is_satisfied().unwrap(), "Constraints not satisfied");
    }
}
