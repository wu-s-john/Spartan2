//! Grand-product permutation checks for RS shuffle (Bellpepper version)
//!
//! This module implements the multiset equality check using grand products.
//! The key insight is that two sets are equal iff:
//!   ∏(r - a_i) = ∏(r - b_j)
//! for a random challenge r.

use crate::traits::Engine;
use bellpepper_core::{ConstraintSystem, SynthesisError, num::AllocatedNum};
use ff::PrimeField;

/// Trait for types that can be compressed into a field element for permutation products
///
/// The generic parameter NUM_CHALLENGES represents the number of random challenges
/// needed to compress the type into a single field element.
pub trait PermutationProduct<F: PrimeField, const NUM_CHALLENGES: usize> {
  /// Compress the element using random challenges
  fn product<CS: ConstraintSystem<F>>(
    &self,
    cs: CS,
    challenges: &[AllocatedNum<F>; NUM_CHALLENGES],
  ) -> Result<AllocatedNum<F>, SynthesisError>;
}

/// Represents a pair of (index, position) for multiset equality checks
#[derive(Clone)]
pub struct IndexPositionPair<F: PrimeField> {
  pub idx: AllocatedNum<F>,
  pub pos: AllocatedNum<F>,
}

impl<F: PrimeField> IndexPositionPair<F> {
  pub fn new(idx: AllocatedNum<F>, pos: AllocatedNum<F>) -> Self {
    Self { idx, pos }
  }

  /// Compress the pair using challenges α and β
  /// Returns: α·idx + β·pos
  pub fn compress<CS: ConstraintSystem<F>>(
    &self,
    mut cs: CS,
    alpha: &AllocatedNum<F>,
    beta: &AllocatedNum<F>,
  ) -> Result<AllocatedNum<F>, SynthesisError> {
    // Compute alpha * idx
    let alpha_idx = AllocatedNum::alloc(cs.namespace(|| "alpha_idx"), || {
      let a = alpha.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let i = self
        .idx
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?;
      Ok(a * i)
    })?;
    cs.enforce(
      || "alpha * idx",
      |lc| lc + alpha.get_variable(),
      |lc| lc + self.idx.get_variable(),
      |lc| lc + alpha_idx.get_variable(),
    );

    // Compute beta * pos
    let beta_pos = AllocatedNum::alloc(cs.namespace(|| "beta_pos"), || {
      let b = beta.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let p = self
        .pos
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?;
      Ok(b * p)
    })?;
    cs.enforce(
      || "beta * pos",
      |lc| lc + beta.get_variable(),
      |lc| lc + self.pos.get_variable(),
      |lc| lc + beta_pos.get_variable(),
    );

    // Return alpha_idx + beta_pos
    let result = AllocatedNum::alloc(cs.namespace(|| "result"), || {
      let ai = alpha_idx
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?;
      let bp = beta_pos
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?;
      Ok(ai + bp)
    })?;
    cs.enforce(
      || "result = alpha_idx + beta_pos",
      |lc| lc + alpha_idx.get_variable() + beta_pos.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc + result.get_variable(),
    );

    Ok(result)
  }
}

impl<F: PrimeField> PermutationProduct<F, 2> for IndexPositionPair<F> {
  fn product<CS: ConstraintSystem<F>>(
    &self,
    cs: CS,
    challenges: &[AllocatedNum<F>; 2],
  ) -> Result<AllocatedNum<F>, SynthesisError> {
    self.compress(cs, &challenges[0], &challenges[1])
  }
}

/// Indexed ElGamal ciphertext for permutation checks (uses coordinates)
pub struct IndexedCiphertext<F: PrimeField> {
  pub idx: AllocatedNum<F>,
  pub c1_x: AllocatedNum<F>,
  pub c1_y: AllocatedNum<F>,
  pub c2_x: AllocatedNum<F>,
  pub c2_y: AllocatedNum<F>,
}

impl<F: PrimeField> IndexedCiphertext<F> {
  pub fn new<E: Engine<Base = F>>(
    idx: AllocatedNum<F>,
    ciphertext: &super::data_structures::ElGamalCiphertextVar<E>,
  ) -> Self {
    Self {
      idx,
      c1_x: ciphertext.c1.x.clone(),
      c1_y: ciphertext.c1.y.clone(),
      c2_x: ciphertext.c2.x.clone(),
      c2_y: ciphertext.c2.y.clone(),
    }
  }
}

impl<F: PrimeField> PermutationProduct<F, 5> for IndexedCiphertext<F> {
  fn product<CS: ConstraintSystem<F>>(
    &self,
    mut cs: CS,
    challenges: &[AllocatedNum<F>; 5],
  ) -> Result<AllocatedNum<F>, SynthesisError> {
    // Compress: α₀·idx + α₁·c1.x + α₂·c1.y + α₃·c2.x + α₄·c2.y
    // We'll compute this incrementally

    // First: α₀·idx
    let term0 = AllocatedNum::alloc(cs.namespace(|| "term0"), || {
      let c = challenges[0]
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?;
      let v = self
        .idx
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?;
      Ok(c * v)
    })?;
    cs.enforce(
      || "α₀·idx",
      |lc| lc + challenges[0].get_variable(),
      |lc| lc + self.idx.get_variable(),
      |lc| lc + term0.get_variable(),
    );

    // α₁·c1.x
    let term1 = AllocatedNum::alloc(cs.namespace(|| "term1"), || {
      let c = challenges[1]
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?;
      let v = self
        .c1_x
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?;
      Ok(c * v)
    })?;
    cs.enforce(
      || "α₁·c1_x",
      |lc| lc + challenges[1].get_variable(),
      |lc| lc + self.c1_x.get_variable(),
      |lc| lc + term1.get_variable(),
    );

    // α₂·c1.y
    let term2 = AllocatedNum::alloc(cs.namespace(|| "term2"), || {
      let c = challenges[2]
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?;
      let v = self
        .c1_y
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?;
      Ok(c * v)
    })?;
    cs.enforce(
      || "α₂·c1_y",
      |lc| lc + challenges[2].get_variable(),
      |lc| lc + self.c1_y.get_variable(),
      |lc| lc + term2.get_variable(),
    );

    // α₃·c2.x
    let term3 = AllocatedNum::alloc(cs.namespace(|| "term3"), || {
      let c = challenges[3]
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?;
      let v = self
        .c2_x
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?;
      Ok(c * v)
    })?;
    cs.enforce(
      || "α₃·c2_x",
      |lc| lc + challenges[3].get_variable(),
      |lc| lc + self.c2_x.get_variable(),
      |lc| lc + term3.get_variable(),
    );

    // α₄·c2.y
    let term4 = AllocatedNum::alloc(cs.namespace(|| "term4"), || {
      let c = challenges[4]
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?;
      let v = self
        .c2_y
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?;
      Ok(c * v)
    })?;
    cs.enforce(
      || "α₄·c2_y",
      |lc| lc + challenges[4].get_variable(),
      |lc| lc + self.c2_y.get_variable(),
      |lc| lc + term4.get_variable(),
    );

    // Sum all terms
    let result = AllocatedNum::alloc(cs.namespace(|| "sum"), || {
      let t0 = term0.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let t1 = term1.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let t2 = term2.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let t3 = term3.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let t4 = term4.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      Ok(t0 + t1 + t2 + t3 + t4)
    })?;
    cs.enforce(
      || "sum all terms",
      |lc| {
        lc + term0.get_variable()
          + term1.get_variable()
          + term2.get_variable()
          + term3.get_variable()
          + term4.get_variable()
      },
      |lc| lc + CS::one(),
      |lc| lc + result.get_variable(),
    );

    Ok(result)
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
pub fn check_grand_product<F, T, CS, const NUM_CHALLENGES: usize>(
  mut cs: CS,
  left: &[T],
  right: &[T],
  challenges: &[AllocatedNum<F>; NUM_CHALLENGES],
) -> Result<(), SynthesisError>
where
  F: PrimeField,
  T: PermutationProduct<F, NUM_CHALLENGES>,
  CS: ConstraintSystem<F>,
{
  assert_eq!(left.len(), right.len(), "Arrays must have equal length");

  // Compute left product: ∏ compress(left[i])
  let mut prod_left = AllocatedNum::alloc(cs.namespace(|| "init_prod_left"), || Ok(F::ONE))?;
  cs.enforce(
    || "init prod_left = 1",
    |lc| lc + prod_left.get_variable(),
    |lc| lc + CS::one(),
    |lc| lc + CS::one(),
  );

  for (i, item) in left.iter().enumerate() {
    let compressed = item.product(cs.namespace(|| format!("compress_left_{}", i)), challenges)?;
    let new_prod = AllocatedNum::alloc(cs.namespace(|| format!("prod_left_{}", i)), || {
      let p = prod_left
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?;
      let c = compressed
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?;
      Ok(p * c)
    })?;
    cs.enforce(
      || format!("prod_left_{} = prod * compressed", i),
      |lc| lc + prod_left.get_variable(),
      |lc| lc + compressed.get_variable(),
      |lc| lc + new_prod.get_variable(),
    );
    prod_left = new_prod;
  }

  // Compute right product: ∏ compress(right[j])
  let mut prod_right = AllocatedNum::alloc(cs.namespace(|| "init_prod_right"), || Ok(F::ONE))?;
  cs.enforce(
    || "init prod_right = 1",
    |lc| lc + prod_right.get_variable(),
    |lc| lc + CS::one(),
    |lc| lc + CS::one(),
  );

  for (i, item) in right.iter().enumerate() {
    let compressed = item.product(cs.namespace(|| format!("compress_right_{}", i)), challenges)?;
    let new_prod = AllocatedNum::alloc(cs.namespace(|| format!("prod_right_{}", i)), || {
      let p = prod_right
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?;
      let c = compressed
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?;
      Ok(p * c)
    })?;
    cs.enforce(
      || format!("prod_right_{} = prod * compressed", i),
      |lc| lc + prod_right.get_variable(),
      |lc| lc + compressed.get_variable(),
      |lc| lc + new_prod.get_variable(),
    );
    prod_right = new_prod;
  }

  // Enforce equality: ∏ left = ∏ right
  cs.enforce(
    || "prod_left == prod_right",
    |lc| lc + prod_left.get_variable() - prod_right.get_variable(),
    |lc| lc + CS::one(),
    |lc| lc,
  );

  Ok(())
}

/// Implementation of PermutationProduct for AllocatedNum (single element)
///
/// Used for simple permutation checks where elements are field elements.
/// Uses the formula: (challenge - element) for the product.
impl<F: PrimeField> PermutationProduct<F, 1> for AllocatedNum<F> {
  fn product<CS: ConstraintSystem<F>>(
    &self,
    mut cs: CS,
    challenges: &[AllocatedNum<F>; 1],
  ) -> Result<AllocatedNum<F>, SynthesisError> {
    // For permutation check: (r - element)
    // This allows checking ∏(r - a_i) = ∏(r - b_i)
    let result = AllocatedNum::alloc(cs.namespace(|| "r - element"), || {
      let r = challenges[0]
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?;
      let e = self.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      Ok(r - e)
    })?;
    cs.enforce(
      || "result = r - element",
      |lc| lc + challenges[0].get_variable() - self.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc + result.get_variable(),
    );
    Ok(result)
  }
}
