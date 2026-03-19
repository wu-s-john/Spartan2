//! Grand-product permutation checks for RS shuffle
//!
//! This module implements the multiset equality check using grand products.
//! The key insight is that two sets are equal iff:
//!   ∏(r - a_i) = ∏(r - b_j)
//! for a random challenge r.

use super::data_structures::ElGamalCiphertextVar;
use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use ark_r1cs_std::{
    eq::EqGadget,
    fields::{fp::FpVar, FieldVar},
    groups::CurveVar,
};
use ark_relations::r1cs::{ConstraintSystemRef, SynthesisError};

/// Trait for types that can be compressed into a field element for permutation products
///
/// The generic parameter NUM_CHALLENGES represents the number of random challenges
/// needed to compress the type into a single field element.
pub trait PermutationProduct<F: PrimeField, const NUM_CHALLENGES: usize> {
    /// Compress the element using random challenges
    fn product(&self, challenges: &[FpVar<F>; NUM_CHALLENGES]) -> FpVar<F>;
}

/// Represents a pair of (index, position) for multiset equality checks
#[derive(Clone)]
pub struct IndexPositionPair<F: PrimeField> {
    pub idx: FpVar<F>,
    pub pos: FpVar<F>,
}

impl<F: PrimeField> IndexPositionPair<F> {
    pub fn new(idx: FpVar<F>, pos: FpVar<F>) -> Self {
        Self { idx, pos }
    }

    /// Compress the pair using challenges α and β
    /// Returns: α·idx + β·pos
    pub fn compress(&self, alpha: &FpVar<F>, beta: &FpVar<F>) -> FpVar<F> {
        alpha * &self.idx + beta * &self.pos
    }
}

impl<F: PrimeField> PermutationProduct<F, 2> for IndexPositionPair<F> {
    fn product(&self, challenges: &[FpVar<F>; 2]) -> FpVar<F> {
        self.compress(&challenges[0], &challenges[1])
    }
}

/// Indexed ElGamal ciphertext for permutation checks
///
/// Combines an index with an ElGamal ciphertext for grand product checks
/// that verify the shuffle preserves the multiset of ciphertexts.
pub struct IndexedElGamalCiphertext<C, CV>
where
    C: CurveGroup,
    C::BaseField: PrimeField,
    CV: CurveVar<C, C::BaseField>,
{
    pub idx: FpVar<C::BaseField>,
    pub ciphertext: ElGamalCiphertextVar<C, CV>,
}

impl<C, CV> IndexedElGamalCiphertext<C, CV>
where
    C: CurveGroup,
    C::BaseField: PrimeField,
    CV: CurveVar<C, C::BaseField>,
{
    pub fn new(idx: FpVar<C::BaseField>, ciphertext: ElGamalCiphertextVar<C, CV>) -> Self {
        Self { idx, ciphertext }
    }
}

impl<C, CV> PermutationProduct<C::BaseField, 5> for IndexedElGamalCiphertext<C, CV>
where
    C: CurveGroup,
    C::BaseField: PrimeField,
    CV: CurveVar<C, C::BaseField>,
{
    fn product(&self, challenges: &[FpVar<C::BaseField>; 5]) -> FpVar<C::BaseField> {
        // Convert curve points to constraint field elements
        // We use to_constraint_field which gives us the x, y coordinates
        let c1_fields = self.ciphertext.c1.to_constraint_field().unwrap();
        let c2_fields = self.ciphertext.c2.to_constraint_field().unwrap();

        // Compress: α₀·idx + α₁·c1.x + α₂·c1.y + α₃·c2.x + α₄·c2.y
        &challenges[0] * &self.idx
            + &challenges[1] * &c1_fields[0]
            + &challenges[2] * &c1_fields[1]
            + &challenges[3] * &c2_fields[0]
            + &challenges[4] * &c2_fields[1]
    }
}

/// Check multiset equality via grand product
///
/// Proves that the left and right collections represent the same multiset
/// by checking: ∏ compress(left[i]) = ∏ compress(right[j])
///
/// # Type Parameters
/// - `F`: The prime field type
/// - `T`: The type of elements being compared (must implement PermutationProduct)
/// - `NUM_CHALLENGES`: Number of random challenges for compression
///
/// # Security
/// This check is sound with high probability when challenges are derived
/// via Fiat-Shamir from a hash of the public inputs.
pub fn check_grand_product<F, T, const NUM_CHALLENGES: usize>(
    _cs: ConstraintSystemRef<F>,
    left: &[T],
    right: &[T],
    challenges: &[FpVar<F>; NUM_CHALLENGES],
) -> Result<(), SynthesisError>
where
    F: PrimeField,
    T: PermutationProduct<F, NUM_CHALLENGES>,
{
    // Compute left product: ∏ compress(left[i])
    let mut prod_left = FpVar::<F>::one();
    for item in left {
        let compressed = item.product(challenges);
        prod_left *= &compressed;
    }

    // Compute right product: ∏ compress(right[j])
    let mut prod_right = FpVar::<F>::one();
    for item in right {
        let compressed = item.product(challenges);
        prod_right *= &compressed;
    }

    // Enforce equality: ∏ left = ∏ right
    prod_left.enforce_equal(&prod_right)?;

    Ok(())
}

/// Implementation of PermutationProduct for FpVar (single element)
///
/// Used for simple permutation checks where elements are field elements.
/// Uses the formula: (challenge - element) for the product.
impl<F: PrimeField> PermutationProduct<F, 1> for FpVar<F> {
    fn product(&self, challenges: &[FpVar<F>; 1]) -> FpVar<F> {
        // For permutation check: (r - element)
        // This allows checking ∏(r - a_i) = ∏(r - b_i)
        &challenges[0] - self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr as TestField;
    use ark_r1cs_std::alloc::AllocVar;
    use ark_relations::r1cs::ConstraintSystem;

    #[test]
    fn test_grand_product_same_multiset() {
        let cs = ConstraintSystem::<TestField>::new_ref();

        // Create two identical sets in different order
        let values_left = vec![1u64, 2, 3, 4, 5];
        let values_right = vec![3u64, 1, 5, 2, 4]; // Same elements, different order

        let left: Vec<FpVar<TestField>> = values_left
            .iter()
            .map(|&v| {
                FpVar::new_witness(cs.clone(), || Ok(TestField::from(v)))
                    .expect("Failed to allocate")
            })
            .collect();

        let right: Vec<FpVar<TestField>> = values_right
            .iter()
            .map(|&v| {
                FpVar::new_witness(cs.clone(), || Ok(TestField::from(v)))
                    .expect("Failed to allocate")
            })
            .collect();

        // Random challenge
        let challenge = FpVar::new_input(cs.clone(), || Ok(TestField::from(12345u64)))
            .expect("Failed to allocate challenge");

        // Check should pass
        check_grand_product::<TestField, _, 1>(cs.clone(), &left, &right, &[challenge])
            .expect("Grand product check failed");

        assert!(cs.is_satisfied().unwrap());
    }

    #[test]
    fn test_grand_product_different_multiset() {
        let cs = ConstraintSystem::<TestField>::new_ref();

        // Create two different sets
        let values_left = vec![1u64, 2, 3, 4, 5];
        let values_right = vec![1u64, 2, 3, 4, 6]; // Different element

        let left: Vec<FpVar<TestField>> = values_left
            .iter()
            .map(|&v| {
                FpVar::new_witness(cs.clone(), || Ok(TestField::from(v)))
                    .expect("Failed to allocate")
            })
            .collect();

        let right: Vec<FpVar<TestField>> = values_right
            .iter()
            .map(|&v| {
                FpVar::new_witness(cs.clone(), || Ok(TestField::from(v)))
                    .expect("Failed to allocate")
            })
            .collect();

        let challenge = FpVar::new_input(cs.clone(), || Ok(TestField::from(12345u64)))
            .expect("Failed to allocate challenge");

        // This will create constraints that should NOT be satisfied
        check_grand_product::<TestField, _, 1>(cs.clone(), &left, &right, &[challenge])
            .expect("Grand product check should create constraints");

        // The constraint system should NOT be satisfied
        assert!(!cs.is_satisfied().unwrap());
    }

    #[test]
    fn test_index_position_pair() {
        let cs = ConstraintSystem::<TestField>::new_ref();

        let pair = IndexPositionPair::new(
            FpVar::new_witness(cs.clone(), || Ok(TestField::from(3u64))).unwrap(),
            FpVar::new_witness(cs.clone(), || Ok(TestField::from(7u64))).unwrap(),
        );

        let alpha = FpVar::new_input(cs.clone(), || Ok(TestField::from(2u64))).unwrap();
        let beta = FpVar::new_input(cs.clone(), || Ok(TestField::from(5u64))).unwrap();

        let compressed = pair.compress(&alpha, &beta);

        // Expected: 2*3 + 5*7 = 6 + 35 = 41
        let expected =
            FpVar::new_constant(cs.clone(), TestField::from(41u64)).expect("Failed to create");
        compressed.enforce_equal(&expected).unwrap();

        assert!(cs.is_satisfied().unwrap());
    }
}
