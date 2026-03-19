//! Parallel ElGamal re-encryption for RS shuffle (native + gadget)
//!
//! This module provides both native curve arithmetic and bellpepper gadget synthesis
//! for re-encrypting a deck of ElGamal ciphertexts. Both layers are parallelized
//! with rayon since all 52 cards are independent.

use super::data_structures::{ElGamalCiphertext, ElGamalCiphertextVar};
use crate::{
  gadgets::{ecc::AllocatedPoint, utils::le_bits_to_num},
  traits::{Engine, Group},
};
use bellpepper_core::{
  boolean::AllocatedBit, num::AllocatedNum, ConstraintSystem, Index, LinearCombination,
  SynthesisError, Variable,
};
use ff::{PrimeField, PrimeFieldBits};
use rayon::prelude::*;

// ============================================================================
// Witness-only constraint system parameterized by field (not Engine)
// ============================================================================

/// A lightweight witness-only constraint system parameterized directly by a field.
///
/// Unlike `SatisfyingAssignment<E>` which implements `ConstraintSystem<E::Scalar>`,
/// this type implements `ConstraintSystem<F>` for any `PrimeField F`, making it
/// suitable for gadgets that operate on `E::Base`.
struct WitnessCS<F: PrimeField> {
  input_assignment: Vec<F>,
  aux_assignment: Vec<F>,
}

impl<F: PrimeField> ConstraintSystem<F> for WitnessCS<F> {
  type Root = Self;

  fn new() -> Self {
    Self {
      input_assignment: vec![F::ONE],
      aux_assignment: vec![],
    }
  }

  fn alloc<Fn, A, AR>(&mut self, _: A, f: Fn) -> Result<Variable, SynthesisError>
  where
    Fn: FnOnce() -> Result<F, SynthesisError>,
    A: FnOnce() -> AR,
    AR: Into<String>,
  {
    self.aux_assignment.push(f()?);
    Ok(Variable(Index::Aux(self.aux_assignment.len() - 1)))
  }

  fn alloc_input<Fn, A, AR>(&mut self, _: A, f: Fn) -> Result<Variable, SynthesisError>
  where
    Fn: FnOnce() -> Result<F, SynthesisError>,
    A: FnOnce() -> AR,
    AR: Into<String>,
  {
    self.input_assignment.push(f()?);
    Ok(Variable(Index::Input(self.input_assignment.len() - 1)))
  }

  fn enforce<A, AR, LA, LB, LC>(&mut self, _: A, _a: LA, _b: LB, _c: LC)
  where
    A: FnOnce() -> AR,
    AR: Into<String>,
    LA: FnOnce(LinearCombination<F>) -> LinearCombination<F>,
    LB: FnOnce(LinearCombination<F>) -> LinearCombination<F>,
    LC: FnOnce(LinearCombination<F>) -> LinearCombination<F>,
  {
    // No-op: we only care about witness values
  }

  fn push_namespace<NR, N>(&mut self, _: N)
  where
    NR: Into<String>,
    N: FnOnce() -> NR,
  {
  }

  fn pop_namespace(&mut self) {}

  fn get_root(&mut self) -> &mut Self::Root {
    self
  }

  fn is_witness_generator(&self) -> bool {
    true
  }

  fn is_extensible() -> bool {
    true
  }

  fn extend(&mut self, other: &Self) {
    self
      .input_assignment
      .extend(&other.input_assignment[1..]);
    self.aux_assignment.extend(other.aux_assignment.clone());
  }

  fn extend_inputs(&mut self, new_inputs: &[F]) {
    self.input_assignment.extend(new_inputs);
  }

  fn extend_aux(&mut self, new_aux: &[F]) {
    self.aux_assignment.extend(new_aux);
  }

  fn allocate_empty(&mut self, aux_n: usize, inputs_n: usize) -> (&mut [F], &mut [F]) {
    let allocated_aux = {
      let i = self.aux_assignment.len();
      self.aux_assignment.resize(aux_n + i, F::ZERO);
      &mut self.aux_assignment[i..]
    };
    let allocated_inputs = {
      let i = self.input_assignment.len();
      self.input_assignment.resize(inputs_n + i, F::ZERO);
      &mut self.input_assignment[i..]
    };
    (allocated_aux, allocated_inputs)
  }

  fn inputs_slice(&self) -> &[F] {
    &self.input_assignment
  }

  fn aux_slice(&self) -> &[F] {
    &self.aux_assignment
  }
}

// ============================================================================
// Native parallel re-encryption
// ============================================================================

/// Pre-computed native re-encryption results for witness generation
#[derive(Clone, Debug)]
pub struct NativeReencryptionData<E: Engine, const N: usize> {
  /// Output ciphertexts after re-encryption
  pub output_ciphertexts: [ElGamalCiphertext<E>; N],
  /// r·G results for each card (x, y)
  pub r_g: [(E::Base, E::Base); N],
  /// r·PK results for each card (x, y)
  pub r_pk: [(E::Base, E::Base); N],
}

/// Perform scalar multiplication natively: scalar * point
/// Uses double-and-add with Jacobian projective coordinates (handles all edge cases).
fn native_scalar_mul<F: PrimeField + PrimeFieldBits>(
  scalar: F,
  point_x: F,
  point_y: F,
  curve_a: F,
) -> (F, F) {
  let bits: Vec<bool> = scalar.to_le_bits().into_iter().collect::<Vec<_>>();
  let highest = bits.iter().rposition(|b| *b).unwrap_or(0);

  // Jacobian: (X, Y, Z) represents affine (X/Z², Y/Z³)
  let mut ax = point_x;
  let mut ay = point_y;
  let mut az = F::ONE;

  for i in (0..highest).rev() {
    // Double in Jacobian
    let (dx, dy, dz) = jacobian_double(ax, ay, az, curve_a);
    ax = dx;
    ay = dy;
    az = dz;

    if bits[i] {
      // Add affine point in mixed Jacobian+affine
      let (sx, sy, sz) = jacobian_add_affine(ax, ay, az, point_x, point_y, curve_a);
      ax = sx;
      ay = sy;
      az = sz;
    }
  }

  // Convert back to affine
  let z_inv = az.invert().unwrap();
  let z_inv2 = z_inv.square();
  let z_inv3 = z_inv2 * z_inv;
  (ax * z_inv2, ay * z_inv3)
}

/// Jacobian point doubling: 2(X,Y,Z) -> (X',Y',Z')
/// Formula for a=0 curves (Pallas/Vesta): simplified since a=0
fn jacobian_double<F: PrimeField>(x: F, y: F, z: F, a: F) -> (F, F, F) {
  if y == F::ZERO {
    return (F::ONE, F::ONE, F::ZERO); // point at infinity
  }
  let xx = x.square();
  let yy = y.square();
  let yyyy = yy.square();
  let zz = z.square();

  let s = ((x + yy).square() - xx - yyyy).double(); // 2*((X+YY)²-XX-YYYY)
  let m = if a == F::ZERO {
    xx + xx + xx // 3*XX (when a=0)
  } else {
    xx + xx + xx + a * zz.square() // 3*XX + a*ZZ²
  };
  let t = m.square() - s - s; // M² - 2*S

  let x3 = t;
  let y3 = m * (s - t) - yyyy.double().double().double(); // M*(S-T) - 8*YYYY
  let z3 = (y + z).square() - yy - zz; // (Y+Z)² - YY - ZZ

  (x3, y3, z3)
}

/// Mixed Jacobian+affine addition: (X1,Y1,Z1) + (x2,y2) -> (X3,Y3,Z3)
/// Handles the case where the points are equal (falls back to doubling).
fn jacobian_add_affine<F: PrimeField>(
  x1: F,
  y1: F,
  z1: F,
  x2: F,
  y2: F,
  curve_a: F,
) -> (F, F, F) {
  if z1 == F::ZERO {
    return (x2, y2, F::ONE); // infinity + P = P
  }

  let z1z1 = z1.square();
  let u2 = x2 * z1z1;
  let s2 = y2 * z1 * z1z1;

  let h = u2 - x1;
  let r = s2 - y1;

  if h == F::ZERO {
    if r == F::ZERO {
      // Points are equal: use doubling
      return jacobian_double(x1, y1, z1, curve_a);
    }
    // Point and its negation: return infinity
    return (F::ONE, F::ONE, F::ZERO);
  }

  let hh = h.square();
  let hhh = hh * h;
  let v = x1 * hh;

  let x3 = r.square() - hhh - v - v;
  let y3 = r * (v - x3) - y1 * hhh;
  let z3 = z1 * h;

  (x3, y3, z3)
}

/// Affine point addition: P + Q (handles P == Q by using Jacobian internally)
fn affine_add_safe<F: PrimeField>(x1: F, y1: F, x2: F, y2: F, a: F) -> (F, F) {
  // Use Jacobian to handle all cases
  let (rx, ry, rz) = jacobian_add_affine(x1, y1, F::ONE, x2, y2, a);
  let z_inv = rz.invert().unwrap();
  let z_inv2 = z_inv.square();
  let z_inv3 = z_inv2 * z_inv;
  (rx * z_inv2, ry * z_inv3)
}

/// Re-encrypt all N ciphertexts in parallel using rayon
pub fn native_reencrypt_parallel<E: Engine, const N: usize>(
  ciphertexts: &[ElGamalCiphertext<E>; N],
  randomizations: &[E::Base; N],
  pk: (E::Base, E::Base),
  generator: (E::Base, E::Base),
) -> NativeReencryptionData<E, N> {
  let (a, _, _, _) = E::GE::group_params();

  let results: Vec<(ElGamalCiphertext<E>, (E::Base, E::Base), (E::Base, E::Base))> = (0..N)
    .into_par_iter()
    .map(|i| {
      let r = randomizations[i];
      let ct = &ciphertexts[i];

      let (rg_x, rg_y) = native_scalar_mul(r, generator.0, generator.1, a);
      let (rpk_x, rpk_y) = native_scalar_mul(r, pk.0, pk.1, a);

      let (c1p_x, c1p_y) = affine_add_safe(ct.c1_x, ct.c1_y, rg_x, rg_y, a);
      let (c2p_x, c2p_y) = affine_add_safe(ct.c2_x, ct.c2_y, rpk_x, rpk_y, a);

      let out_ct = ElGamalCiphertext::new(c1p_x, c1p_y, c2p_x, c2p_y);
      (out_ct, (rg_x, rg_y), (rpk_x, rpk_y))
    })
    .collect();

  let mut output_ciphertexts = Vec::with_capacity(N);
  let mut r_g = Vec::with_capacity(N);
  let mut r_pk = Vec::with_capacity(N);

  for (ct, rg, rpk) in results {
    output_ciphertexts.push(ct);
    r_g.push(rg);
    r_pk.push(rpk);
  }

  NativeReencryptionData {
    output_ciphertexts: output_ciphertexts.try_into().ok().unwrap(),
    r_g: r_g.try_into().ok().unwrap(),
    r_pk: r_pk.try_into().ok().unwrap(),
  }
}

// ============================================================================
// Single-card gadget
// ============================================================================

/// Re-encrypt one ciphertext — the core gadget.
///
/// Per card:
/// 1. Bit-decompose randomization scalar (NUM_BITS AllocatedBit, constrained via le_bits_to_num)
/// 2. Scalar mul r·G (generator allocated as point)
/// 3. Scalar mul r·PK
/// 4. c1' = c1 + r·G, c2' = c2 + r·PK
pub fn rerandomize_ciphertext_bp<E: Engine, CS: ConstraintSystem<E::Base>>(
  mut cs: CS,
  ct: &ElGamalCiphertextVar<E>,
  randomization: &AllocatedNum<E::Base>,
  pk: &AllocatedPoint<E>,
  generator_coords: (E::Base, E::Base),
) -> Result<ElGamalCiphertextVar<E>, SynthesisError> {
  let num_bits = E::Base::NUM_BITS as usize;

  // Step 1: Bit-decompose the randomization scalar
  let r_bits = alloc_scalar_bits::<E, _>(cs.namespace(|| "r_bits"), randomization, num_bits)?;

  // Step 2: r·G
  let gen_point = AllocatedPoint::<E>::alloc(
    cs.namespace(|| "generator"),
    Some((generator_coords.0, generator_coords.1, false)),
  )?;
  let r_g = gen_point.scalar_mul(cs.namespace(|| "r_G"), &r_bits)?;

  // Step 3: r·PK
  let r_pk = pk.scalar_mul(cs.namespace(|| "r_PK"), &r_bits)?;

  // Step 4: c1' = c1 + r·G
  let c1_prime = ct.c1.add(cs.namespace(|| "c1_plus_rG"), &r_g)?;

  // Step 5: c2' = c2 + r·PK
  let c2_prime = ct.c2.add(cs.namespace(|| "c2_plus_rPK"), &r_pk)?;

  Ok(ElGamalCiphertextVar::new(c1_prime, c2_prime))
}

/// Bit-decompose a scalar and constrain via le_bits_to_num
fn alloc_scalar_bits<E: Engine, CS: ConstraintSystem<E::Base>>(
  mut cs: CS,
  scalar: &AllocatedNum<E::Base>,
  num_bits: usize,
) -> Result<Vec<AllocatedBit>, SynthesisError> {
  let scalar_val = scalar.get_value();

  let bits_le: Vec<Option<bool>> = if let Some(val) = scalar_val {
    val
      .to_le_bits()
      .into_iter()
      .take(num_bits)
      .map(Some)
      .collect()
  } else {
    vec![None; num_bits]
  };

  let mut allocated_bits = Vec::with_capacity(num_bits);
  for (i, bit) in bits_le.iter().enumerate() {
    let ab = AllocatedBit::alloc(cs.namespace(|| format!("bit_{}", i)), *bit)?;
    allocated_bits.push(ab);
  }

  // Constrain: bits reconstruct the scalar
  let reconstructed = le_bits_to_num(cs.namespace(|| "le_bits_to_num"), &allocated_bits)?;

  // Enforce equality
  cs.enforce(
    || "scalar_bits_match",
    |lc| lc + scalar.get_variable() - reconstructed.get_variable(),
    |lc| lc + CS::one(),
    |lc| lc,
  );

  Ok(allocated_bits)
}

// ============================================================================
// Parallel gadget synthesis for full deck
// ============================================================================

/// Re-encrypt deck with parallel witness generation.
///
/// When `cs.is_witness_generator()` is true (prove phase):
/// Uses parallel witness synthesis on separate `WitnessCS` instances,
/// pre-allocates aux slots via `allocate_empty`, and fills them in.
///
/// When `cs.is_witness_generator()` is false (setup phase):
/// Runs serially to capture correct constraints.
pub fn reencrypt_deck_bp<E: Engine, CS: ConstraintSystem<E::Base>, const N: usize>(
  cs: &mut CS,
  input_deck: &[ElGamalCiphertextVar<E>; N],
  randomizations: &[AllocatedNum<E::Base>; N],
  pk: &AllocatedPoint<E>,
  _native_data: &NativeReencryptionData<E, N>,
  generator_coords: (E::Base, E::Base),
) -> Result<(), SynthesisError> {
  if cs.is_witness_generator() {
    reencrypt_deck_parallel_witness::<E, CS, N>(
      cs,
      input_deck,
      randomizations,
      pk,
      generator_coords,
    )
  } else {
    reencrypt_deck_serial::<E, CS, N>(cs, input_deck, randomizations, pk, generator_coords)?;
    Ok(())
  }
}

/// Serial re-encryption — used during shape (setup) phase
fn reencrypt_deck_serial<E: Engine, CS: ConstraintSystem<E::Base>, const N: usize>(
  cs: &mut CS,
  input_deck: &[ElGamalCiphertextVar<E>; N],
  randomizations: &[AllocatedNum<E::Base>; N],
  pk: &AllocatedPoint<E>,
  generator_coords: (E::Base, E::Base),
) -> Result<[ElGamalCiphertextVar<E>; N], SynthesisError> {
  let mut results = Vec::with_capacity(N);
  for i in 0..N {
    let ct = rerandomize_ciphertext_bp::<E, _>(
      cs.namespace(|| format!("reencrypt_{}", i)),
      &input_deck[i],
      &randomizations[i],
      pk,
      generator_coords,
    )?;
    results.push(ct);
  }

  results
    .try_into()
    .map_err(|_| SynthesisError::Unsatisfiable)
}

/// Parallel witness generation for re-encryption.
///
/// Runs each card's gadget on a separate `WitnessCS` (where `enforce()` is a no-op),
/// then fills pre-allocated aux slots via `allocate_empty`. This produces exactly
/// the same aux variable count as the serial path — no extra allocations.
fn reencrypt_deck_parallel_witness<E: Engine, CS: ConstraintSystem<E::Base>, const N: usize>(
  cs: &mut CS,
  input_deck: &[ElGamalCiphertextVar<E>; N],
  randomizations: &[AllocatedNum<E::Base>; N],
  pk: &AllocatedPoint<E>,
  generator_coords: (E::Base, E::Base),
) -> Result<(), SynthesisError> {
  let pk_x_val = pk.x.get_value().ok_or(SynthesisError::AssignmentMissing)?;
  let pk_y_val = pk.y.get_value().ok_or(SynthesisError::AssignmentMissing)?;

  struct CardInputs<F: PrimeField> {
    ct_c1_x: F,
    ct_c1_y: F,
    ct_c2_x: F,
    ct_c2_y: F,
    randomization: F,
  }

  let card_inputs: Vec<CardInputs<E::Base>> = (0..N)
    .map(|i| {
      Ok(CardInputs {
        ct_c1_x: input_deck[i].c1.x.get_value().ok_or(SynthesisError::AssignmentMissing)?,
        ct_c1_y: input_deck[i].c1.y.get_value().ok_or(SynthesisError::AssignmentMissing)?,
        ct_c2_x: input_deck[i].c2.x.get_value().ok_or(SynthesisError::AssignmentMissing)?,
        ct_c2_y: input_deck[i].c2.y.get_value().ok_or(SynthesisError::AssignmentMissing)?,
        randomization: randomizations[i].get_value().ok_or(SynthesisError::AssignmentMissing)?,
      })
    })
    .collect::<Result<Vec<_>, SynthesisError>>()?;

  // Run all N gadgets in parallel on separate WitnessCS instances
  let mini_witnesses: Vec<Vec<E::Base>> = card_inputs
    .par_iter()
    .enumerate()
    .map(|(i, inputs)| {
      let mut mini_cs = WitnessCS::<E::Base>::new();

      let ct_var = ElGamalCiphertextVar::<E>::alloc(
        mini_cs.namespace(|| "ct"),
        &ElGamalCiphertext::new(inputs.ct_c1_x, inputs.ct_c1_y, inputs.ct_c2_x, inputs.ct_c2_y),
      )
      .expect("alloc ct");

      let r_var = AllocatedNum::alloc(mini_cs.namespace(|| "r"), || Ok(inputs.randomization))
        .expect("alloc r");

      let pk_var = AllocatedPoint::<E>::alloc(
        mini_cs.namespace(|| "pk"),
        Some((pk_x_val, pk_y_val, false)),
      )
      .expect("alloc pk");

      // Record how many aux vars were allocated for inputs (ct + r + pk).
      // These already exist on the main CS — we must skip them when copying.
      let input_aux_count = mini_cs.aux_assignment.len();

      let _result = rerandomize_ciphertext_bp::<E, _>(
        mini_cs.namespace(|| format!("reencrypt_{}", i)),
        &ct_var,
        &r_var,
        &pk_var,
        generator_coords,
      )
      .expect("rerandomize failed");

      // Only return the gadget's own variables, not the re-allocated inputs
      mini_cs.aux_assignment[input_aux_count..].to_vec()
    })
    .collect();

  // Pre-allocate exactly the right number of aux slots, then fill them.
  // This matches the serial path's variable count: each card produces K aux vars,
  // serial allocates them one-by-one, we pre-allocate N*K and copy in bulk.
  let total_aux: usize = mini_witnesses.iter().map(|w| w.len()).sum();
  let (aux_slice, _) = cs.allocate_empty(total_aux, 0);

  let mut offset = 0;
  for mini_w in &mini_witnesses {
    aux_slice[offset..offset + mini_w.len()].copy_from_slice(mini_w);
    offset += mini_w.len();
  }

  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::PallasHyraxEngine;
  use crate::traits::Group;
  use bellpepper_core::test_cs::TestConstraintSystem;
  use ff::Field;
  use std::time::Instant;

  type E = PallasHyraxEngine;
  type Base = <E as Engine>::Base;

  fn find_point_on_curve(a: Base, b: Base) -> (Base, Base) {
    let mut x = Base::ONE;
    loop {
      let rhs = x.cube() + a * x + b;
      if let Some(y_val) = Option::from(rhs.sqrt()) {
        return (x, y_val);
      }
      x += Base::ONE;
    }
  }

  fn make_random_ciphertext() -> ElGamalCiphertext<E> {
    let (a, b, _, _) = <E as Engine>::GE::group_params();
    let (c1_x, c1_y) = find_point_on_curve(a, b);
    let mut x = c1_x + Base::ONE;
    let (c2_x, c2_y) = loop {
      let rhs = x.cube() + a * x + b;
      if let Some(y) = Option::from(rhs.sqrt()) {
        break (x, y);
      }
      x += Base::ONE;
    };
    ElGamalCiphertext::new(c1_x, c1_y, c2_x, c2_y)
  }

  #[test]
  fn test_single_card_rerandomize() {
    let mut cs = TestConstraintSystem::<Base>::new();

    let (a, b, _, _) = <E as Engine>::GE::group_params();
    let gen_coords = find_point_on_curve(a, b);

    let ct_native = make_random_ciphertext();
    let ct_var =
      ElGamalCiphertextVar::<E>::alloc(cs.namespace(|| "ct"), &ct_native).expect("alloc ct");

    let r = Base::from(42u64);
    let r_var = AllocatedNum::alloc(cs.namespace(|| "r"), || Ok(r)).expect("alloc r");

    let pk_native = find_point_on_curve(a, b);
    let pk_var = AllocatedPoint::<E>::alloc(
      cs.namespace(|| "pk"),
      Some((pk_native.0, pk_native.1, false)),
    )
    .expect("alloc pk");

    let result = rerandomize_ciphertext_bp::<E, _>(
      cs.namespace(|| "reencrypt"),
      &ct_var,
      &r_var,
      &pk_var,
      gen_coords,
    );

    assert!(result.is_ok(), "Gadget failed: {:?}", result.err());

    let num_constraints = cs.num_constraints();
    println!("Single card re-encryption constraints: {}", num_constraints);

    assert!(
      cs.is_satisfied(),
      "Constraints not satisfied! First unsatisfied: {:?}",
      cs.which_is_unsatisfied()
    );
  }

  #[test]
  fn test_native_reencrypt_parallel() {
    let (a, b, _, _) = <E as Engine>::GE::group_params();
    let gen_coords = find_point_on_curve(a, b);

    const N: usize = 4;
    let mut ciphertexts = Vec::with_capacity(N);
    let mut randomizations = Vec::with_capacity(N);

    for _ in 0..N {
      ciphertexts.push(make_random_ciphertext());
      randomizations.push(Base::from(7u64));
    }

    let ct_arr: [ElGamalCiphertext<E>; N] = ciphertexts.try_into().ok().unwrap();
    let r_arr: [Base; N] = randomizations.try_into().ok().unwrap();
    let pk = find_point_on_curve(a, b);

    let result = native_reencrypt_parallel::<E, N>(&ct_arr, &r_arr, pk, gen_coords);

    // Verify all outputs are valid curve points
    for i in 0..N {
      let ct = &result.output_ciphertexts[i];
      let rhs1 = ct.c1_x.cube() + a * ct.c1_x + b;
      let lhs1 = ct.c1_y.square();
      assert_eq!(lhs1, rhs1, "c1 not on curve for card {}", i);

      let rhs2 = ct.c2_x.cube() + a * ct.c2_x + b;
      let lhs2 = ct.c2_y.square();
      assert_eq!(lhs2, rhs2, "c2 not on curve for card {}", i);
    }
  }

  #[test]
  fn test_native_serial_vs_parallel_equivalence() {
    let (a, b, _, _) = <E as Engine>::GE::group_params();
    let gen_coords = find_point_on_curve(a, b);
    let pk = find_point_on_curve(a, b);

    const N: usize = 8;
    let mut ciphertexts = Vec::with_capacity(N);
    let mut randomizations = Vec::with_capacity(N);

    for i in 0..N {
      ciphertexts.push(make_random_ciphertext());
      randomizations.push(Base::from((i + 1) as u64));
    }

    let ct_arr: [ElGamalCiphertext<E>; N] = ciphertexts.try_into().ok().unwrap();
    let r_arr: [Base; N] = randomizations.try_into().ok().unwrap();

    let parallel_result = native_reencrypt_parallel::<E, N>(&ct_arr, &r_arr, pk, gen_coords);

    // Compare with serial computation
    for i in 0..N {
      let r = r_arr[i];
      let ct = &ct_arr[i];
      let (rg_x, rg_y) = native_scalar_mul(r, gen_coords.0, gen_coords.1, a);
      let (rpk_x, rpk_y) = native_scalar_mul(r, pk.0, pk.1, a);
      let (c1p_x, c1p_y) = affine_add_safe(ct.c1_x, ct.c1_y, rg_x, rg_y, a);
      let (c2p_x, c2p_y) = affine_add_safe(ct.c2_x, ct.c2_y, rpk_x, rpk_y, a);

      assert_eq!(parallel_result.output_ciphertexts[i].c1_x, c1p_x);
      assert_eq!(parallel_result.output_ciphertexts[i].c1_y, c1p_y);
      assert_eq!(parallel_result.output_ciphertexts[i].c2_x, c2p_x);
      assert_eq!(parallel_result.output_ciphertexts[i].c2_y, c2p_y);
    }
  }

  #[test]
  fn test_performance_native_reencrypt() {
    let (a, b, _, _) = <E as Engine>::GE::group_params();
    let gen_coords = find_point_on_curve(a, b);
    let pk = find_point_on_curve(a, b);

    const N: usize = 52;
    let mut ciphertexts = Vec::with_capacity(N);
    let mut randomizations = Vec::with_capacity(N);

    for i in 0..N {
      ciphertexts.push(make_random_ciphertext());
      randomizations.push(Base::from((i + 1) as u64));
    }

    let ct_arr: [ElGamalCiphertext<E>; N] = ciphertexts.try_into().ok().unwrap();
    let r_arr: [Base; N] = randomizations.try_into().ok().unwrap();

    // Serial timing
    let serial_start = Instant::now();
    for i in 0..N {
      let r = r_arr[i];
      let ct = &ct_arr[i];
      let (rg_x, rg_y) = native_scalar_mul(r, gen_coords.0, gen_coords.1, a);
      let (rpk_x, rpk_y) = native_scalar_mul(r, pk.0, pk.1, a);
      let _ = affine_add_safe(ct.c1_x, ct.c1_y, rg_x, rg_y, a);
      let _ = affine_add_safe(ct.c2_x, ct.c2_y, rpk_x, rpk_y, a);
    }
    let serial_time = serial_start.elapsed();

    // Parallel timing
    let parallel_start = Instant::now();
    let _result = native_reencrypt_parallel::<E, N>(&ct_arr, &r_arr, pk, gen_coords);
    let parallel_time = parallel_start.elapsed();

    let speedup = serial_time.as_secs_f64() / parallel_time.as_secs_f64();

    println!("\n=== Native Re-encryption Performance (N={}) ===", N);
    println!("Serial:   {:?}", serial_time);
    println!("Parallel: {:?}", parallel_time);
    println!("Speedup:  {:.2}x", speedup);
    println!("Threads:  {}", rayon::current_num_threads());
  }

  #[test]
  fn test_parallel_witness_synthesis() {
    let (a, b, _, _) = <E as Engine>::GE::group_params();
    let gen_coords = find_point_on_curve(a, b);

    // Use a different point for PK to avoid P=Q edge cases in gadget
    let mut pk_x = gen_coords.0 + Base::from(10u64);
    let pk_coords = loop {
      let rhs = pk_x.cube() + a * pk_x + b;
      if let Some(y) = Option::from(rhs.sqrt()) {
        break (pk_x, y);
      }
      pk_x += Base::ONE;
    };

    const N: usize = 4;
    let mut ct_natives = Vec::with_capacity(N);
    let mut r_natives = Vec::with_capacity(N);

    for i in 0..N {
      ct_natives.push(make_random_ciphertext());
      r_natives.push(Base::from((i + 10) as u64));
    }

    let ct_arr: [ElGamalCiphertext<E>; N] = ct_natives.clone().try_into().ok().unwrap();
    let r_arr: [Base; N] = r_natives.clone().try_into().ok().unwrap();

    // Pre-compute native results
    let native_data =
      native_reencrypt_parallel::<E, N>(&ct_arr, &r_arr, pk_coords, gen_coords);

    // Test serial gadget path (TestConstraintSystem is NOT a witness generator)
    let mut cs_serial = TestConstraintSystem::<Base>::new();

    let mut input_deck_serial = Vec::with_capacity(N);
    let mut rand_vars_serial = Vec::with_capacity(N);
    for i in 0..N {
      input_deck_serial.push(
        ElGamalCiphertextVar::<E>::alloc(
          cs_serial.namespace(|| format!("ct_{}", i)),
          &ct_arr[i],
        )
        .unwrap(),
      );
      rand_vars_serial.push(
        AllocatedNum::alloc(cs_serial.namespace(|| format!("r_{}", i)), || Ok(r_arr[i])).unwrap(),
      );
    }
    let pk_serial = AllocatedPoint::<E>::alloc(
      cs_serial.namespace(|| "pk"),
      Some((pk_coords.0, pk_coords.1, false)),
    )
    .unwrap();

    let input_deck_arr: [ElGamalCiphertextVar<E>; N] =
      input_deck_serial.try_into().ok().unwrap();
    let rand_arr: [AllocatedNum<Base>; N] = rand_vars_serial.try_into().ok().unwrap();

    let serial_result = reencrypt_deck_bp::<E, _, N>(
      &mut cs_serial,
      &input_deck_arr,
      &rand_arr,
      &pk_serial,
      &native_data,
      gen_coords,
    );

    assert!(serial_result.is_ok(), "Serial gadget failed: {:?}", serial_result.err());

    // Verify serial constraints satisfied
    assert!(
      cs_serial.is_satisfied(),
      "Serial constraints not satisfied! {:?}",
      cs_serial.which_is_unsatisfied()
    );

    let serial_num_constraints = cs_serial.num_constraints();
    println!("Serial deck re-encryption: {} constraints", serial_num_constraints);

    // Test parallel witness path using WitnessCS
    let mut cs_parallel = WitnessCS::<Base>::new();

    let mut input_deck_par = Vec::with_capacity(N);
    let mut rand_vars_par = Vec::with_capacity(N);
    for i in 0..N {
      input_deck_par.push(
        ElGamalCiphertextVar::<E>::alloc(
          cs_parallel.namespace(|| format!("ct_{}", i)),
          &ct_arr[i],
        )
        .unwrap(),
      );
      rand_vars_par.push(
        AllocatedNum::alloc(cs_parallel.namespace(|| format!("r_{}", i)), || Ok(r_arr[i]))
          .unwrap(),
      );
    }
    let pk_par = AllocatedPoint::<E>::alloc(
      cs_parallel.namespace(|| "pk"),
      Some((pk_coords.0, pk_coords.1, false)),
    )
    .unwrap();

    let input_deck_arr_par: [ElGamalCiphertextVar<E>; N] =
      input_deck_par.try_into().ok().unwrap();
    let rand_arr_par: [AllocatedNum<Base>; N] = rand_vars_par.try_into().ok().unwrap();

    let par_result = reencrypt_deck_bp::<E, _, N>(
      &mut cs_parallel,
      &input_deck_arr_par,
      &rand_arr_par,
      &pk_par,
      &native_data,
      gen_coords,
    );

    assert!(par_result.is_ok(), "Parallel gadget failed: {:?}", par_result.err());

    let par_aux_len = cs_parallel.aux_assignment.len();
    println!("Parallel witness synthesis: {} aux values", par_aux_len);
  }

  #[test]
  fn test_parallel_vs_serial_witness_performance() {
    let (a, b, _, _) = <E as Engine>::GE::group_params();
    let gen_coords = find_point_on_curve(a, b);

    let mut pk_x = gen_coords.0 + Base::from(10u64);
    let pk_coords = loop {
      let rhs = pk_x.cube() + a * pk_x + b;
      if let Some(y) = Option::from(rhs.sqrt()) {
        break (pk_x, y);
      }
      pk_x += Base::ONE;
    };

    const N: usize = 52;
    let mut ct_natives = Vec::with_capacity(N);
    let mut r_natives = Vec::with_capacity(N);

    for i in 0..N {
      ct_natives.push(make_random_ciphertext());
      r_natives.push(Base::from((i + 10) as u64));
    }

    let ct_arr: [ElGamalCiphertext<E>; N] = ct_natives.try_into().ok().unwrap();
    let r_arr: [Base; N] = r_natives.try_into().ok().unwrap();

    let native_data =
      native_reencrypt_parallel::<E, N>(&ct_arr, &r_arr, pk_coords, gen_coords);

    // Serial witness timing
    let serial_start = Instant::now();
    {
      let mut cs = WitnessCS::<Base>::new();
      let mut deck = Vec::with_capacity(N);
      let mut rands = Vec::with_capacity(N);
      for i in 0..N {
        deck.push(
          ElGamalCiphertextVar::<E>::alloc(cs.namespace(|| format!("ct_{}", i)), &ct_arr[i])
            .unwrap(),
        );
        rands.push(
          AllocatedNum::alloc(cs.namespace(|| format!("r_{}", i)), || Ok(r_arr[i])).unwrap(),
        );
      }
      let pk = AllocatedPoint::<E>::alloc(
        cs.namespace(|| "pk"),
        Some((pk_coords.0, pk_coords.1, false)),
      )
      .unwrap();

      let deck_arr: [ElGamalCiphertextVar<E>; N] = deck.try_into().ok().unwrap();
      let rand_arr: [AllocatedNum<Base>; N] = rands.try_into().ok().unwrap();

      // Serial: run each card's gadget sequentially on the main CS
      for i in 0..N {
        let _ = rerandomize_ciphertext_bp::<E, _>(
          cs.namespace(|| format!("serial_reencrypt_{}", i)),
          &deck_arr[i],
          &rand_arr[i],
          &pk,
          gen_coords,
        )
        .unwrap();
      }
    }
    let serial_time = serial_start.elapsed();

    // Parallel witness timing
    let parallel_start = Instant::now();
    {
      let mut cs = WitnessCS::<Base>::new();
      let mut deck = Vec::with_capacity(N);
      let mut rands = Vec::with_capacity(N);
      for i in 0..N {
        deck.push(
          ElGamalCiphertextVar::<E>::alloc(cs.namespace(|| format!("ct_{}", i)), &ct_arr[i])
            .unwrap(),
        );
        rands.push(
          AllocatedNum::alloc(cs.namespace(|| format!("r_{}", i)), || Ok(r_arr[i])).unwrap(),
        );
      }
      let pk = AllocatedPoint::<E>::alloc(
        cs.namespace(|| "pk"),
        Some((pk_coords.0, pk_coords.1, false)),
      )
      .unwrap();

      let deck_arr: [ElGamalCiphertextVar<E>; N] = deck.try_into().ok().unwrap();
      let rand_arr: [AllocatedNum<Base>; N] = rands.try_into().ok().unwrap();

      let _ = reencrypt_deck_bp::<E, _, N>(
        &mut cs,
        &deck_arr,
        &rand_arr,
        &pk,
        &native_data,
        gen_coords,
      )
      .unwrap();
    }
    let parallel_time = parallel_start.elapsed();

    let speedup = serial_time.as_secs_f64() / parallel_time.as_secs_f64();

    println!("\n=== Gadget Witness Synthesis Performance (N={}) ===", N);
    println!("Serial:   {:?}", serial_time);
    println!("Parallel: {:?}", parallel_time);
    println!("Speedup:  {:.2}x", speedup);
    println!("Threads:  {}", rayon::current_num_threads());
  }
}
