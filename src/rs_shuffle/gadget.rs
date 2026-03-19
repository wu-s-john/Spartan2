//! RS shuffle verification gadgets for SNARK circuits
//!
//! This module contains the main circuit gadgets for verifying RS shuffle operations.

use super::bit_generation::derive_split_bits_gadget;
use super::data_structures::{
    ElGamalCiphertextVar, PermutationWitnessTraceVar, SortedRowVar, UnsortedRowVar,
};
use super::encryption::ElGamalEncryption;
use super::permutation::{check_grand_product, IndexPositionPair, IndexedElGamalCiphertext};
use ark_crypto_primitives::sponge::Absorb;
use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use ark_r1cs_std::{
    boolean::Boolean, eq::EqGadget, fields::fp::FpVar, fields::FieldVar, groups::CurveVar,
    prelude::*,
};
use ark_relations::r1cs::{ConstraintSystemRef, SynthesisError};
use std::ops::Not;

/// Verify RS shuffle constraints for indices only (without ElGamal ciphertexts)
///
/// This function verifies that a shuffle was performed correctly on indices by:
/// 1. Checking row-local constraints at each level
/// 2. Verifying permutation consistency at each level
/// 3. Ensuring indices are preserved through the shuffle
///
/// # Arguments
/// * `cs` - Constraint system reference
/// * `indices_init` - Initial indices (as SNARK variables)
/// * `indices_after_shuffle` - Final shuffled indices (as SNARK variables)
/// * `witness` - The witness data containing the shuffle permutation
/// * `alpha` - First Fiat-Shamir challenge
/// * `beta` - Second Fiat-Shamir challenge
pub fn rs_shuffle_indices<F, const N: usize, const LEVELS: usize>(
    cs: ConstraintSystemRef<F>,
    indices_init: &[FpVar<F>],
    indices_after_shuffle: &[FpVar<F>],
    witness: &PermutationWitnessTraceVar<F, N, LEVELS>,
    alpha: &FpVar<F>,
    beta: &FpVar<F>,
) -> Result<(), SynthesisError>
where
    F: PrimeField,
{
    // 1. Create indexed values for initial indices
    let values_initial: Vec<IndexPositionPair<F>> = witness.uns_levels[0]
        .iter()
        .zip(indices_init.iter())
        .map(|(row, idx)| {
            row.idx.enforce_equal(idx)?;
            Ok(IndexPositionPair::new(row.idx.clone(), FpVar::zero()))
        })
        .collect::<Result<Vec<_>, SynthesisError>>()?;

    // 2. Create indexed values for final indices
    let values_final: Vec<IndexPositionPair<F>> = witness.sorted_levels[LEVELS - 1]
        .iter()
        .zip(indices_after_shuffle.iter())
        .map(|(row, idx)| {
            row.idx.enforce_equal(idx)?;
            Ok(IndexPositionPair::new(row.idx.clone(), FpVar::zero()))
        })
        .collect::<Result<Vec<_>, SynthesisError>>()?;

    // 3. Verify each shuffle level
    for level in 0..LEVELS {
        let unsorted = &witness.uns_levels[level];
        let sorted_arr = &witness.sorted_levels[level];

        verify_shuffle_level::<F, N>(cs.clone(), unsorted, sorted_arr, alpha, beta)?;
    }

    // 4. Final permutation check using just indices
    check_grand_product::<F, IndexPositionPair<F>, 2>(
        cs.clone(),
        &values_initial,
        &values_final,
        &[alpha.clone(), beta.clone()],
    )?;

    Ok(())
}

/// Verify RS shuffle constraints with ElGamal ciphertexts
///
/// This function verifies that a shuffle was performed correctly by:
/// 1. Checking row-local constraints at each level
/// 2. Verifying permutation consistency at each level
/// 3. Ensuring ElGamal ciphertexts are preserved through the shuffle
pub fn rs_shuffle<C, CV, const N: usize, const LEVELS: usize>(
    cs: ConstraintSystemRef<C::BaseField>,
    ct_init: &[ElGamalCiphertextVar<C, CV>; N],
    ct_after_shuffle: &[ElGamalCiphertextVar<C, CV>; N],
    witness: &PermutationWitnessTraceVar<C::BaseField, N, LEVELS>,
    alpha: &FpVar<C::BaseField>,
    beta: &FpVar<C::BaseField>,
) -> Result<(), SynthesisError>
where
    C: CurveGroup,
    C::BaseField: PrimeField,
    CV: CurveVar<C, C::BaseField>,
{
    // 1. Create indexed ciphertexts for initial state
    let ciphertexts_initial: Vec<IndexedElGamalCiphertext<C, CV>> = witness.uns_levels[0]
        .iter()
        .zip(ct_init.iter())
        .map(|(row, ct)| IndexedElGamalCiphertext::new(row.idx.clone(), ct.clone()))
        .collect();

    // 2. Create indexed ciphertexts for final state
    let ciphertexts_final: Vec<IndexedElGamalCiphertext<C, CV>> = witness.sorted_levels[LEVELS - 1]
        .iter()
        .zip(ct_after_shuffle.iter())
        .map(|(row, ct)| IndexedElGamalCiphertext::new(row.idx.clone(), ct.clone()))
        .collect();

    // 3. Compute challenge powers for ciphertext compression
    let beta_2 = beta * beta;
    let beta_3 = &beta_2 * beta;
    let beta_4 = &beta_3 * beta;

    // 4. Verify each shuffle level
    for level in 0..LEVELS {
        let unsorted = &witness.uns_levels[level];
        let sorted_arr = &witness.sorted_levels[level];

        verify_shuffle_level::<_, N>(cs.clone(), unsorted, sorted_arr, alpha, beta)?;
    }

    // 5. Grand product permutation check for ciphertexts
    check_grand_product::<C::BaseField, IndexedElGamalCiphertext<C, CV>, 5>(
        cs.clone(),
        &ciphertexts_initial,
        &ciphertexts_final,
        &[
            alpha.clone(),
            beta.clone(),
            beta_2,
            beta_3,
            beta_4,
        ],
    )?;

    Ok(())
}

/// Verify RS shuffle with re-encryption
///
/// This is the main entry point that proves:
/// 1. Bits are correctly derived from seed via Poseidon
/// 2. Each level performs valid stable-partition shuffle
/// 3. Grand product permutation check passes
/// 4. Output ciphertexts = rerandomized shuffled input
pub fn rs_shuffle_with_reencryption<C, CV, const N: usize, const LEVELS: usize>(
    cs: ConstraintSystemRef<C::BaseField>,
    // Public inputs
    seed: &FpVar<C::BaseField>,
    ct_input: &[ElGamalCiphertextVar<C, CV>; N],
    ct_shuffled: &[ElGamalCiphertextVar<C, CV>; N],
    ct_output: &[ElGamalCiphertextVar<C, CV>; N],
    shuffler_pk: &CV,
    alpha: &FpVar<C::BaseField>,
    beta: &FpVar<C::BaseField>,
    // Private witnesses
    witness: &PermutationWitnessTraceVar<C::BaseField, N, LEVELS>,
    rerandomizations: &[FpVar<C::BaseField>; N],
    // Constants
    num_samples: usize,
    generator_powers: &[C],
) -> Result<(), SynthesisError>
where
    C: CurveGroup,
    C::BaseField: PrimeField + Absorb,
    CV: CurveVar<C, C::BaseField>,
{
    // ═══════════════════════════════════════════════════════════════
    // Step 1: Derive bits from seed (proves bits match public seed)
    // ═══════════════════════════════════════════════════════════════
    let bits_mat = derive_split_bits_gadget::<C::BaseField, N, LEVELS>(cs.clone(), seed, num_samples)?;

    // Verify that witness bits match derived bits
    for level in 0..LEVELS {
        for i in 0..N {
            witness.bits_mat[level][i].enforce_equal(&bits_mat[level][i])?;
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Step 2: Verify the shuffle from input to shuffled ciphertexts
    // ═══════════════════════════════════════════════════════════════
    rs_shuffle::<C, CV, N, LEVELS>(cs.clone(), ct_input, ct_shuffled, witness, alpha, beta)?;

    // ═══════════════════════════════════════════════════════════════
    // Step 3: Verify re-encryption from shuffled to output
    // ═══════════════════════════════════════════════════════════════
    let elgamal = ElGamalEncryption::new(generator_powers.to_vec());
    let reencrypted =
        elgamal.reencrypt_deck::<CV, N>(cs.clone(), ct_shuffled, rerandomizations, shuffler_pk)?;

    // Verify output matches re-encrypted result
    for i in 0..N {
        reencrypted[i].c1.enforce_equal(&ct_output[i].c1)?;
        reencrypted[i].c2.enforce_equal(&ct_output[i].c2)?;
    }

    Ok(())
}

/// Verify row-local constraints for one level
///
/// Checks:
/// 1. Initial counters are zero at start of each bucket
/// 2. Counter evolution (zeros/ones counts)
/// 3. Bucket constants stay constant within bucket
/// 4. Final tallies match at end of each bucket
/// 5. Destination formula is correctly computed
fn verify_row_constraints<F, const N: usize>(
    cs: ConstraintSystemRef<F>,
    unsorted: &[UnsortedRowVar<F>; N],
) -> Result<Vec<IndexPositionPair<F>>, SynthesisError>
where
    F: PrimeField,
{
    let mut idx_next_pos_pairs = Vec::new();
    let zero = FpVar::<F>::zero();

    for i in 0..N {
        let u = &unsorted[i];
        let u_prev = if i > 0 { Some(&unsorted[i - 1]) } else { None };
        let u_next = if i + 1 < N {
            Some(&unsorted[i + 1])
        } else {
            None
        };

        let one = FpVar::<F>::one();
        let bit_as_fp: FpVar<F> = u.bit.clone().into();
        let one_minus_bit = &one - &bit_as_fp;

        // ═══════════════════════════════════════════════════════════════════
        // Constraint 1: Initial counters at start of each bucket
        // First row of each bucket must have num_zeros = 0 and num_ones = 0
        // ═══════════════════════════════════════════════════════════════════
        let is_first_in_bucket = if let Some(prev) = u_prev {
            // First in bucket if previous row has different bucket_id
            prev.bucket_id.is_eq(&u.bucket_id)?.not()
        } else {
            // First row is always first in its bucket
            Boolean::constant(true)
        };

        // Conditional constraint: is_first_in_bucket => num_zeros == 0
        let zero_check = is_first_in_bucket.clone().select(&u.num_zeros, &zero)?;
        zero_check.enforce_equal(&zero)?;

        // Conditional constraint: is_first_in_bucket => num_ones == 0
        let one_check = is_first_in_bucket.select(&u.num_ones, &zero)?;
        one_check.enforce_equal(&zero)?;

        // ═══════════════════════════════════════════════════════════════════
        // Constraint 2: Counter evolution within bucket
        // z_{i+1} = z_i + (1 - b_i) when same bucket
        // o_{i+1} = o_i + b_i when same bucket
        // ═══════════════════════════════════════════════════════════════════
        if let Some(next) = u_next {
            let same_bucket = u.bucket_id.is_eq(&next.bucket_id)?;

            // When same_bucket: enforce next.num_zeros == u.num_zeros + (1 - bit)
            // When different bucket: enforce next.num_zeros == 0 (handled by is_first_in_bucket)
            let expected_next_zeros = &u.num_zeros + &one_minus_bit;
            let diff_zeros = &next.num_zeros - &expected_next_zeros;
            let conditional_diff_zeros = same_bucket.clone().select(&diff_zeros, &zero)?;
            conditional_diff_zeros.enforce_equal(&zero)?;

            // When same_bucket: enforce next.num_ones == u.num_ones + bit
            let expected_next_ones = &u.num_ones + &bit_as_fp;
            let diff_ones = &next.num_ones - &expected_next_ones;
            let conditional_diff_ones = same_bucket.clone().select(&diff_ones, &zero)?;
            conditional_diff_ones.enforce_equal(&zero)?;

            // ═══════════════════════════════════════════════════════════════
            // Constraint 3: Bucket constants stay constant within bucket
            // ═══════════════════════════════════════════════════════════════
            let total_zeros_diff = &u.total_zeros_in_bucket - &next.total_zeros_in_bucket;
            let conditional_total_zeros_diff = same_bucket.clone().select(&total_zeros_diff, &zero)?;
            conditional_total_zeros_diff.enforce_equal(&zero)?;

            let length_diff = &u.bucket_length - &next.bucket_length;
            let conditional_length_diff = same_bucket.select(&length_diff, &zero)?;
            conditional_length_diff.enforce_equal(&zero)?;
        }

        // ═══════════════════════════════════════════════════════════════════
        // Constraint 4: Final tallies at end of each bucket
        // At the last row in a bucket:
        //   num_zeros + (1 - bit) == total_zeros_in_bucket
        //   num_ones + bit == bucket_length - total_zeros_in_bucket
        // ═══════════════════════════════════════════════════════════════════
        let is_last_in_bucket = if let Some(next) = u_next {
            u.bucket_id.is_eq(&next.bucket_id)?.not()
        } else {
            Boolean::constant(true)
        };

        // Conditional: is_last => (num_zeros + (1-bit) == total_zeros)
        let zeros_tally_diff = (&u.num_zeros + &one_minus_bit) - &u.total_zeros_in_bucket;
        let conditional_zeros_tally = is_last_in_bucket.clone().select(&zeros_tally_diff, &zero)?;
        conditional_zeros_tally.enforce_equal(&zero)?;

        // Conditional: is_last => (num_ones + bit == bucket_length - total_zeros)
        let expected_total_ones = &u.bucket_length - &u.total_zeros_in_bucket;
        let ones_tally_diff = (&u.num_ones + &bit_as_fp) - expected_total_ones;
        let conditional_ones_tally = is_last_in_bucket.select(&ones_tally_diff, &zero)?;
        conditional_ones_tally.enforce_equal(&zero)?;

        // ═══════════════════════════════════════════════════════════════════
        // Constraint 5: Destination formula
        // next_pos = base + z_i + b_i·(Z - z_i + o_i)
        // where base = pos - (z_i + o_i)
        // ═══════════════════════════════════════════════════════════════════
        let pos = FpVar::new_constant(cs.clone(), F::from(i as u64))?;
        let base = &pos - (&u.num_zeros + &u.num_ones);
        let offset =
            &u.num_zeros + &bit_as_fp * (&u.total_zeros_in_bucket - &u.num_zeros + &u.num_ones);
        let expected_dest = base + offset;
        u.next_pos.enforce_equal(&expected_dest)?;

        // Collect pairs for multiset check
        idx_next_pos_pairs.push(IndexPositionPair::new(u.idx.clone(), u.next_pos.clone()));
    }

    Ok(idx_next_pos_pairs)
}

/// Verify one level of the RS shuffle
///
/// This includes:
/// 1. Row-local constraint verification
/// 2. Building the sorted index-position pairs
/// 3. Grand product permutation check
fn verify_shuffle_level<F, const N: usize>(
    cs: ConstraintSystemRef<F>,
    unsorted: &[UnsortedRowVar<F>; N],
    sorted: &[SortedRowVar<F>; N],
    alpha: &FpVar<F>,
    beta: &FpVar<F>,
) -> Result<(), SynthesisError>
where
    F: PrimeField,
{
    // Step 1: Verify row-local constraints
    let idx_next_pos_pairs = verify_row_constraints::<F, N>(cs.clone(), unsorted)?;

    // Step 2: Build right-side pairs from sorted array
    let idx_pos_pairs: Vec<IndexPositionPair<F>> = sorted
        .iter()
        .enumerate()
        .map(|(j, sr)| {
            IndexPositionPair::new(
                sr.idx.clone(),
                FpVar::new_constant(cs.clone(), F::from(j as u64)).unwrap(),
            )
        })
        .collect();

    // Step 3: Grand product permutation check for this level
    check_grand_product::<F, IndexPositionPair<F>, 2>(
        cs.clone(),
        &idx_next_pos_pairs,
        &idx_pos_pairs,
        &[alpha.clone(), beta.clone()],
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rs_shuffle::data_structures::ElGamalCiphertext;
    use crate::rs_shuffle::native::run_rs_shuffle_permutation;
    use ark_bn254::Fr as BaseField;
    use ark_ec::{CurveConfig, PrimeGroup};
    use ark_ff::UniformRand;
    use ark_grumpkin::{GrumpkinConfig, Projective as GrumpkinProjective};
    use ark_r1cs_std::alloc::AllocVar;
    use ark_r1cs_std::groups::curves::short_weierstrass::ProjectiveVar;
    use ark_r1cs_std::prelude::AllocationMode;
    use ark_relations::r1cs::ConstraintSystem;

    type GrumpkinVar = ProjectiveVar<GrumpkinConfig, FpVar<BaseField>>;

    const N: usize = 8;
    const LEVELS: usize = 3;

    #[test]
    fn test_rs_shuffle_circuit() {
        let mut rng = ark_std::test_rng();
        let cs = ConstraintSystem::<BaseField>::new_ref();

        // Generate keys
        let sk = <GrumpkinConfig as CurveConfig>::ScalarField::rand(&mut rng);
        let pk = GrumpkinProjective::generator() * sk;

        // Create initial ciphertexts
        let ct_init: [ElGamalCiphertext<GrumpkinProjective>; N] = std::array::from_fn(|i| {
            let message = <GrumpkinConfig as CurveConfig>::ScalarField::from(i as u64);
            let randomness = <GrumpkinConfig as CurveConfig>::ScalarField::rand(&mut rng);
            ElGamalCiphertext::encrypt_scalar(message, randomness, pk)
        });

        // Run native shuffle
        let seed = BaseField::from(42u64);
        let rs_trace = run_rs_shuffle_permutation::<BaseField, _, N, LEVELS>(seed, &ct_init);

        // Allocate circuit variables
        let ct_init_vars: [ElGamalCiphertextVar<GrumpkinProjective, GrumpkinVar>; N] =
            std::array::from_fn(|i| {
                ElGamalCiphertextVar::new_variable(
                    cs.clone(),
                    || Ok(&ct_init[i]),
                    AllocationMode::Witness,
                )
                .expect("Failed to allocate initial ciphertext")
            });

        let ct_shuffled_vars: [ElGamalCiphertextVar<GrumpkinProjective, GrumpkinVar>; N] =
            std::array::from_fn(|i| {
                ElGamalCiphertextVar::new_variable(
                    cs.clone(),
                    || Ok(&rs_trace.permuted_output[i]),
                    AllocationMode::Witness,
                )
                .expect("Failed to allocate shuffled ciphertext")
            });

        let witness_var = PermutationWitnessTraceVar::new_variable(
            cs.clone(),
            || Ok(&rs_trace.witness_trace),
            AllocationMode::Witness,
        )
        .expect("Failed to allocate witness");

        let alpha = FpVar::new_input(cs.clone(), || Ok(BaseField::from(17u64)))
            .expect("Failed to allocate alpha");
        let beta = FpVar::new_input(cs.clone(), || Ok(BaseField::from(23u64)))
            .expect("Failed to allocate beta");

        // Run the shuffle verification
        rs_shuffle::<GrumpkinProjective, GrumpkinVar, N, LEVELS>(
            cs.clone(),
            &ct_init_vars,
            &ct_shuffled_vars,
            &witness_var,
            &alpha,
            &beta,
        )
        .expect("rs_shuffle failed");

        // Verify constraints are satisfied
        assert!(
            cs.is_satisfied().unwrap(),
            "Constraints should be satisfied for valid shuffle"
        );
    }
}
