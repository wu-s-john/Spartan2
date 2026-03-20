//! Parallel ElGamal re-encryption for RS shuffle (native + gadget)
//!
//! This module provides both native curve arithmetic and bellpepper gadget synthesis
//! for re-encrypting a deck of ElGamal ciphertexts. Both layers are parallelized
//! with rayon since all 52 cards are independent.

use super::data_structures::{ElGamalCiphertext, ElGamalCiphertextVar};
use crate::{
  gadgets::ecc::AllocatedPointNonInfinity,
  traits::{Engine, Group},
};
use bellpepper_core::{
  boolean::AllocatedBit, num::AllocatedNum, ConstraintSystem, Index, LinearCombination,
  SynthesisError, Variable,
};
use ff::{Field, PrimeField, PrimeFieldBits};
use rayon::prelude::*;

// ============================================================================
// Witness-only constraint system parameterized by field (not Engine)
// ============================================================================

/// A lightweight witness-only constraint system parameterized directly by a field.
///
/// Unlike `SatisfyingAssignment<E>` which implements `ConstraintSystem<E::Scalar>`,
/// this type implements `ConstraintSystem<F>` for any `PrimeField F`, making it
/// suitable for gadgets that operate on `E::Base`.
#[allow(dead_code)]
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

/// Precompute constant power table for fixed-base scalar multiplication.
/// Returns `powers[i] = 2^i · base` for `i = 0..num_bits`.
pub fn precompute_fixed_base_powers<F: PrimeField>(
  base: (F, F),
  curve_a: F,
  num_bits: usize,
) -> Vec<(F, F)> {
  let mut powers = Vec::with_capacity(num_bits);
  powers.push(base);
  for _ in 1..num_bits {
    let (x, y) = *powers.last().unwrap();
    let (dx, dy, dz) = jacobian_double(x, y, F::ONE, curve_a);
    let z_inv = dz.invert().unwrap();
    let z_inv2 = z_inv.square();
    let z_inv3 = z_inv2 * z_inv;
    powers.push((dx * z_inv2, dy * z_inv3));
  }
  powers
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
/// 1. Bit-decompose randomization scalar (constrained inline)
/// 2. Scalar mul r·G using fixed-base (generator powers are compile-time constants)
/// 3. Scalar mul r·PK using precomputed in-circuit power table
/// 4. c1' = c1 + r·G, c2' = c2 + r·PK
pub fn rerandomize_ciphertext_bp<E: Engine, CS: ConstraintSystem<E::Base>>(
  mut cs: CS,
  ct: &ElGamalCiphertextVar<E>,
  randomization: &AllocatedNum<E::Base>,
  pk_powers: &[AllocatedPointNonInfinity<E>],
  generator: &AllocatedPointNonInfinity<E>,
  gen_powers: &[(E::Base, E::Base)],
) -> Result<ElGamalCiphertextVar<E>, SynthesisError> {
  let num_bits = E::Base::NUM_BITS as usize;

  // Step 1: Bit-decompose the randomization scalar
  let r_bits = alloc_scalar_bits::<E, _>(cs.namespace(|| "r_bits"), randomization, num_bits)?;

  // Step 2: r·G using fixed-base scalar mul (all doublings precomputed as constants)
  let r_g = generator.scalar_mul_fixed_base(cs.namespace(|| "r_G"), &r_bits, gen_powers)?;

  // Step 3: r·PK using precomputed power table (doublings shared across all cards)
  let r_pk = AllocatedPointNonInfinity::scalar_mul_with_powers(
    cs.namespace(|| "r_PK"),
    &r_bits,
    pk_powers,
  )?;

  // Step 4: c1' = c1 + r·G using incomplete addition (safe: unrelated random points)
  let c1_prime = ct.c1.add_incomplete(cs.namespace(|| "c1_plus_rG"), &r_g)?;

  // Step 5: c2' = c2 + r·PK using incomplete addition (safe: unrelated random points)
  let c2_prime = ct.c2.add_incomplete(cs.namespace(|| "c2_plus_rPK"), &r_pk)?;

  Ok(ElGamalCiphertextVar::new(c1_prime, c2_prime))
}

/// Bit-decompose a scalar and constrain directly: Σ(2^i × bit_i) = scalar
/// This saves 1 variable + 1 constraint per call vs le_bits_to_num + enforce.
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

  // Constrain directly: Σ(2^i × bit_i) = scalar (no intermediate variable)
  let mut coeff = E::Base::ONE;
  cs.enforce(
    || "scalar_bits_match",
    |mut lc| {
      for bit in &allocated_bits {
        lc = lc + (coeff, bit.get_variable());
        coeff = coeff.double();
      }
      lc
    },
    |lc| lc + CS::one(),
    |lc| lc + scalar.get_variable(),
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
  pk: &AllocatedPointNonInfinity<E>,
  native_data: &NativeReencryptionData<E, N>,
  generator: &AllocatedPointNonInfinity<E>,
  gen_powers: &[(E::Base, E::Base)],
) -> Result<(), SynthesisError> {
  let num_bits = E::Base::NUM_BITS as usize;

  // Precompute PK power table in-circuit: pk_powers[i] = 2^i · PK
  // 254 doublings = 1,016 vars, shared across all N cards
  let mut pk_powers = Vec::with_capacity(num_bits);
  pk_powers.push(pk.clone());
  for i in 1..num_bits {
    let next = pk_powers[i - 1].double_incomplete(cs.namespace(|| format!("pk_power_{}", i)))?;
    pk_powers.push(next);
  }

  if cs.is_witness_generator() {
    reencrypt_deck_parallel_witness::<E, CS, N>(
      cs,
      input_deck,
      randomizations,
      &pk_powers,
      generator,
      gen_powers,
      native_data,
    )
  } else {
    reencrypt_deck_serial::<E, CS, N>(
      cs,
      input_deck,
      randomizations,
      &pk_powers,
      generator,
      gen_powers,
    )
  }
}

/// Serial re-encryption — used during shape (setup) phase.
/// Runs `rerandomize_ciphertext_bp` per card and inputizes the 4 output coords.
fn reencrypt_deck_serial<E: Engine, CS: ConstraintSystem<E::Base>, const N: usize>(
  cs: &mut CS,
  input_deck: &[ElGamalCiphertextVar<E>; N],
  randomizations: &[AllocatedNum<E::Base>; N],
  pk_powers: &[AllocatedPointNonInfinity<E>],
  generator: &AllocatedPointNonInfinity<E>,
  gen_powers: &[(E::Base, E::Base)],
) -> Result<(), SynthesisError> {
  for i in 0..N {
    let ct = rerandomize_ciphertext_bp::<E, _>(
      cs.namespace(|| format!("reencrypt_{}", i)),
      &input_deck[i],
      &randomizations[i],
      pk_powers,
      generator,
      gen_powers,
    )?;
    ct.c1.x.inputize(cs.namespace(|| format!("output_ct_{}_c1x", i)))?;
    ct.c1.y.inputize(cs.namespace(|| format!("output_ct_{}_c1y", i)))?;
    ct.c2.x.inputize(cs.namespace(|| format!("output_ct_{}_c2x", i)))?;
    ct.c2.y.inputize(cs.namespace(|| format!("output_ct_{}_c2y", i)))?;
  }
  Ok(())
}

/// Parallel witness generation for re-encryption.
///
/// Runs each card's gadget on a separate `WitnessCS` (where `enforce()` is a no-op),
/// then fills pre-allocated aux slots via `allocate_empty`. This produces exactly
/// the same aux variable count as the serial path — no extra allocations.
///
/// Phase 1: Extract native values from main CS handles (serial, cheap)
/// Phase 2: par_iter over N cards, each on its own WitnessCS (parallel, expensive)
/// Phase 3: Pre-allocate + copy aux values into main CS (serial, cheap)
/// Phase 4: Inputize output coords using aux variable references (serial, cheap)
fn reencrypt_deck_parallel_witness<E: Engine, CS: ConstraintSystem<E::Base>, const N: usize>(
  cs: &mut CS,
  input_deck: &[ElGamalCiphertextVar<E>; N],
  randomizations: &[AllocatedNum<E::Base>; N],
  pk_powers: &[AllocatedPointNonInfinity<E>],
  generator: &AllocatedPointNonInfinity<E>,
  gen_powers: &[(E::Base, E::Base)],
  native_data: &NativeReencryptionData<E, N>,
) -> Result<(), SynthesisError> {
  // === Phase 1: Extract native values (serial, cheap) ===
  let gen_x_val = generator.x.get_value().ok_or(SynthesisError::AssignmentMissing)?;
  let gen_y_val = generator.y.get_value().ok_or(SynthesisError::AssignmentMissing)?;

  let pk_power_vals: Vec<(E::Base, E::Base)> = pk_powers
    .iter()
    .map(|p| {
      Ok((
        p.x.get_value().ok_or(SynthesisError::AssignmentMissing)?,
        p.y.get_value().ok_or(SynthesisError::AssignmentMissing)?,
      ))
    })
    .collect::<Result<Vec<_>, SynthesisError>>()?;

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

  let num_bits = E::Base::NUM_BITS as usize;

  // === Phase 2: Parallel witness synthesis ===
  // Each card returns (gadget_aux_values, [c1x_off, c1y_off, c2x_off, c2y_off])
  let card_results: Vec<(Vec<E::Base>, [usize; 4])> = card_inputs
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

      // Allocate pk_powers in mini_cs (these are "inputs" — skipped when copying)
      let mut mini_pk_powers = Vec::with_capacity(num_bits);
      for (j, &(px, py)) in pk_power_vals.iter().enumerate() {
        let p = AllocatedPointNonInfinity::<E>::alloc(
          mini_cs.namespace(|| format!("pk_pow_{}", j)),
          Some((px, py)),
        )
        .expect("alloc pk_power");
        mini_pk_powers.push(p);
      }

      let gen_var = AllocatedPointNonInfinity::<E>::alloc(
        mini_cs.namespace(|| "gen"),
        Some((gen_x_val, gen_y_val)),
      )
      .expect("alloc gen");

      // Record how many aux vars were allocated for inputs (ct + r + pk_powers + gen).
      // These already exist on the main CS — we must skip them when copying.
      let input_aux_count = mini_cs.aux_assignment.len();

      let result = rerandomize_ciphertext_bp::<E, _>(
        mini_cs.namespace(|| format!("reencrypt_{}", i)),
        &ct_var,
        &r_var,
        &mini_pk_powers,
        &gen_var,
        gen_powers,
      )
      .expect("rerandomize failed");

      // Capture output coord offsets relative to gadget's own aux vars
      let extract_aux_offset = |var: Variable| -> usize {
        let Variable(Index::Aux(j)) = var else {
          panic!("expected aux variable");
        };
        j - input_aux_count
      };
      let offsets = [
        extract_aux_offset(result.c1.x.get_variable()),
        extract_aux_offset(result.c1.y.get_variable()),
        extract_aux_offset(result.c2.x.get_variable()),
        extract_aux_offset(result.c2.y.get_variable()),
      ];

      // Only return the gadget's own variables, not the re-allocated inputs
      (mini_cs.aux_assignment[input_aux_count..].to_vec(), offsets)
    })
    .collect();

  // === Phase 3: Pre-allocate and fill aux slots ===
  let vars_per_card = card_results[0].0.len();
  let base_aux_idx = cs.aux_slice().len();
  let total_aux = N * vars_per_card;
  let (aux_slice, _) = cs.allocate_empty(total_aux, 0);

  let mut offset = 0;
  for (aux_vals, _) in &card_results {
    aux_slice[offset..offset + aux_vals.len()].copy_from_slice(aux_vals);
    offset += aux_vals.len();
  }

  // === Phase 4: Inputize output coords ===
  // Use offsets from first card (deterministic gadget → same structure for all cards)
  let output_offsets = card_results[0].1;
  let coord_names = ["c1x", "c1y", "c2x", "c2y"];

  for i in 0..N {
    let ct = &native_data.output_ciphertexts[i];
    let coord_values = [ct.c1_x, ct.c1_y, ct.c2_x, ct.c2_y];

    for (k, &off) in output_offsets.iter().enumerate() {
      let aux_idx = base_aux_idx + i * vars_per_card + off;
      let aux_var = Variable(Index::Aux(aux_idx));
      let value = coord_values[k];

      let input_var = cs.alloc_input(
        || format!("output_ct_{}_{}", i, coord_names[k]),
        || Ok(value),
      )?;
      // enforce input_var * 1 = aux_var (no-op on SatisfyingAssignment,
      // but the constraint exists from setup's serial path)
      cs.enforce(
        || format!("inputize_output_ct_{}_{}", i, coord_names[k]),
        |lc| lc + input_var,
        |lc| lc + CS::one(),
        |lc| lc + aux_var,
      );
    }
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
    let num_bits = Base::NUM_BITS as usize;

    // Precompute generator power table (constants)
    let gen_powers = precompute_fixed_base_powers(gen_coords, a, num_bits);

    let ct_native = make_random_ciphertext();
    let ct_var =
      ElGamalCiphertextVar::<E>::alloc(cs.namespace(|| "ct"), &ct_native).expect("alloc ct");

    let r = Base::from(42u64);
    let r_var = AllocatedNum::alloc(cs.namespace(|| "r"), || Ok(r)).expect("alloc r");

    let pk_native = find_point_on_curve(a, b);
    let pk_var = AllocatedPointNonInfinity::<E>::alloc(
      cs.namespace(|| "pk"),
      Some((pk_native.0, pk_native.1)),
    )
    .expect("alloc pk");

    // Precompute PK power table in-circuit
    let mut pk_powers = Vec::with_capacity(num_bits);
    pk_powers.push(pk_var.clone());
    for i in 1..num_bits {
      let next = pk_powers[i - 1]
        .double_incomplete(cs.namespace(|| format!("pk_power_{}", i)))
        .expect("double pk");
      pk_powers.push(next);
    }

    let gen_var = AllocatedPointNonInfinity::<E>::alloc(
      cs.namespace(|| "gen"),
      Some(gen_coords),
    )
    .expect("alloc gen");

    let result = rerandomize_ciphertext_bp::<E, _>(
      cs.namespace(|| "reencrypt"),
      &ct_var,
      &r_var,
      &pk_powers,
      &gen_var,
      &gen_powers,
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
    let num_bits = Base::NUM_BITS as usize;
    let gen_powers = precompute_fixed_base_powers(gen_coords, a, num_bits);

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
    let pk_serial = AllocatedPointNonInfinity::<E>::alloc(
      cs_serial.namespace(|| "pk"),
      Some((pk_coords.0, pk_coords.1)),
    )
    .unwrap();
    let gen_serial = AllocatedPointNonInfinity::<E>::alloc(
      cs_serial.namespace(|| "gen"),
      Some(gen_coords),
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
      &gen_serial,
      &gen_powers,
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
    let pk_par = AllocatedPointNonInfinity::<E>::alloc(
      cs_parallel.namespace(|| "pk"),
      Some((pk_coords.0, pk_coords.1)),
    )
    .unwrap();
    let gen_par = AllocatedPointNonInfinity::<E>::alloc(
      cs_parallel.namespace(|| "gen"),
      Some(gen_coords),
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
      &gen_par,
      &gen_powers,
    );

    assert!(par_result.is_ok(), "Parallel gadget failed: {:?}", par_result.err());

    let par_aux_len = cs_parallel.aux_assignment.len();
    println!("Parallel witness synthesis: {} aux values", par_aux_len);
  }

  #[test]
  fn test_parallel_witness_satisfies_constraints() {
    let (a, b, _, _) = <E as Engine>::GE::group_params();
    let gen_coords = find_point_on_curve(a, b);
    let num_bits = Base::NUM_BITS as usize;
    let gen_powers = precompute_fixed_base_powers(gen_coords, a, num_bits);

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

    let native_data =
      native_reencrypt_parallel::<E, N>(&ct_arr, &r_arr, pk_coords, gen_coords);

    // 1. Serial path on TestConstraintSystem: records constraints + witness
    let mut cs_serial = TestConstraintSystem::<Base>::new();

    let mut input_deck = Vec::with_capacity(N);
    let mut rand_vars = Vec::with_capacity(N);
    for i in 0..N {
      input_deck.push(
        ElGamalCiphertextVar::<E>::alloc(
          cs_serial.namespace(|| format!("ct_{}", i)),
          &ct_arr[i],
        )
        .unwrap(),
      );
      rand_vars.push(
        AllocatedNum::alloc(cs_serial.namespace(|| format!("r_{}", i)), || Ok(r_arr[i])).unwrap(),
      );
    }
    let pk_var = AllocatedPointNonInfinity::<E>::alloc(
      cs_serial.namespace(|| "pk"),
      Some(pk_coords),
    )
    .unwrap();
    let gen_var = AllocatedPointNonInfinity::<E>::alloc(
      cs_serial.namespace(|| "gen"),
      Some(gen_coords),
    )
    .unwrap();

    let deck_arr: [ElGamalCiphertextVar<E>; N] = input_deck.try_into().ok().unwrap();
    let rand_arr: [AllocatedNum<Base>; N] = rand_vars.try_into().ok().unwrap();

    reencrypt_deck_bp::<E, _, N>(
      &mut cs_serial,
      &deck_arr,
      &rand_arr,
      &pk_var,
      &native_data,
      &gen_var,
      &gen_powers,
    )
    .expect("serial gadget failed");

    assert!(
      cs_serial.is_satisfied(),
      "Serial constraints not satisfied! {:?}",
      cs_serial.which_is_unsatisfied()
    );

    let serial_aux = cs_serial.scalar_aux();
    let serial_inputs = cs_serial.scalar_inputs();

    // 2. Parallel path on WitnessCS
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
    let pk_par = AllocatedPointNonInfinity::<E>::alloc(
      cs_parallel.namespace(|| "pk"),
      Some(pk_coords),
    )
    .unwrap();
    let gen_par = AllocatedPointNonInfinity::<E>::alloc(
      cs_parallel.namespace(|| "gen"),
      Some(gen_coords),
    )
    .unwrap();

    let deck_arr_par: [ElGamalCiphertextVar<E>; N] =
      input_deck_par.try_into().ok().unwrap();
    let rand_arr_par: [AllocatedNum<Base>; N] = rand_vars_par.try_into().ok().unwrap();

    reencrypt_deck_bp::<E, _, N>(
      &mut cs_parallel,
      &deck_arr_par,
      &rand_arr_par,
      &pk_par,
      &native_data,
      &gen_par,
      &gen_powers,
    )
    .expect("parallel gadget failed");

    let par_aux = &cs_parallel.aux_assignment;
    let par_inputs = &cs_parallel.input_assignment;

    // 3. Assert counts match
    assert_eq!(
      serial_aux.len(),
      par_aux.len(),
      "aux count mismatch: serial={}, parallel={}",
      serial_aux.len(),
      par_aux.len()
    );
    assert_eq!(
      serial_inputs.len(),
      par_inputs.len(),
      "input count mismatch: serial={}, parallel={}",
      serial_inputs.len(),
      par_inputs.len()
    );

    // 4. Assert values are identical (deterministic gadget → same witness)
    for (i, (s, p)) in serial_aux.iter().zip(par_aux.iter()).enumerate() {
      assert_eq!(s, p, "aux[{}] mismatch", i);
    }
    for (i, (s, p)) in serial_inputs.iter().zip(par_inputs.iter()).enumerate() {
      assert_eq!(s, p, "input[{}] mismatch", i);
    }

    // Since serial satisfies constraints and parallel produces identical values,
    // the parallel witness necessarily satisfies all constraints too.
    println!(
      "Parallel witness matches serial: {} aux, {} inputs",
      par_aux.len(),
      par_inputs.len()
    );
  }

  #[test]
  fn test_parallel_witness_end_to_end_proof() {
    use crate::provider::pasta::pallas;
    use crate::provider::VestaHyraxEngine;
    use crate::spartan::SpartanSNARK;
    use crate::traits::circuit::SpartanCircuit;
    use crate::traits::snark::R1CSSNARKTrait;

    // Spartan engine: PallasHyraxEngine
    // Circuit field: pallas::Scalar = vesta::Base
    // EC engine for in-circuit ops: VestaHyraxEngine
    type SpartanE = PallasHyraxEngine;
    type ECEngine = VestaHyraxEngine;
    type Scalar = pallas::Scalar;

    const N: usize = 4;

    fn find_vesta_point(start: Scalar) -> (Scalar, Scalar) {
      let (a, b, _, _) = <VestaHyraxEngine as Engine>::GE::group_params();
      let mut x = start;
      loop {
        let rhs = x.cube() + a * x + b;
        if let Some(y) = Option::from(rhs.sqrt()) {
          return (x, y);
        }
        x += Scalar::ONE;
      }
    }

    let (curve_a, _, _, _) = <ECEngine as Engine>::GE::group_params();
    let gen_coords = find_vesta_point(Scalar::ONE);
    let pk_coords = find_vesta_point(Scalar::from(100u64));
    let num_bits = Scalar::NUM_BITS as usize;
    let gen_powers = precompute_fixed_base_powers(gen_coords, curve_a, num_bits);

    let mut input_cts = Vec::with_capacity(N);
    for i in 0..N {
      let (c1_x, c1_y) = find_vesta_point(Scalar::from((i * 2 + 1) as u64));
      let (c2_x, c2_y) = find_vesta_point(Scalar::from((i * 2 + 200) as u64));
      input_cts.push(ElGamalCiphertext::<ECEngine>::new(c1_x, c1_y, c2_x, c2_y));
    }
    let ct_arr: [ElGamalCiphertext<ECEngine>; N] = input_cts.try_into().ok().unwrap();

    let mut randomizations = Vec::with_capacity(N);
    for i in 0..N {
      randomizations.push(Scalar::from((i + 10) as u64));
    }
    let rand_arr: [Scalar; N] = randomizations.try_into().ok().unwrap();

    let native_data =
      native_reencrypt_parallel::<ECEngine, N>(&ct_arr, &rand_arr, pk_coords, gen_coords);

    /// Minimal circuit that only does re-encryption (no permutation checks)
    #[derive(Clone)]
    struct ReencryptCircuit {
      input_cts: [ElGamalCiphertext<ECEngine>; N],
      randomizations: [Scalar; N],
      pk_coords: (Scalar, Scalar),
      gen_coords: (Scalar, Scalar),
      native_data: NativeReencryptionData<ECEngine, N>,
      gen_powers: Vec<(Scalar, Scalar)>,
    }

    impl SpartanCircuit<SpartanE> for ReencryptCircuit {
      fn public_values(&self) -> Result<Vec<Scalar>, SynthesisError> {
        let mut vals = Vec::with_capacity(4 * N);
        for ct in &self.native_data.output_ciphertexts {
          vals.push(ct.c1_x);
          vals.push(ct.c1_y);
          vals.push(ct.c2_x);
          vals.push(ct.c2_y);
        }
        Ok(vals)
      }

      fn shared<CS: ConstraintSystem<Scalar>>(
        &self,
        _cs: &mut CS,
      ) -> Result<Vec<AllocatedNum<Scalar>>, SynthesisError> {
        Ok(vec![])
      }

      fn precommitted<CS: ConstraintSystem<Scalar>>(
        &self,
        _cs: &mut CS,
        _shared: &[AllocatedNum<Scalar>],
      ) -> Result<Vec<AllocatedNum<Scalar>>, SynthesisError> {
        Ok(vec![])
      }

      fn num_challenges(&self) -> usize {
        0
      }

      fn synthesize<CS: ConstraintSystem<Scalar>>(
        &self,
        cs: &mut CS,
        _shared: &[AllocatedNum<Scalar>],
        _precommitted: &[AllocatedNum<Scalar>],
        _challenges: Option<&[Scalar]>,
      ) -> Result<(), SynthesisError> {
        let mut deck_vars = Vec::with_capacity(N);
        let mut rand_vars = Vec::with_capacity(N);
        for i in 0..N {
          deck_vars.push(ElGamalCiphertextVar::<ECEngine>::alloc(
            cs.namespace(|| format!("ct_{}", i)),
            &self.input_cts[i],
          )?);
          rand_vars.push(AllocatedNum::alloc(
            cs.namespace(|| format!("r_{}", i)),
            || Ok(self.randomizations[i]),
          )?);
        }

        let pk_var = AllocatedPointNonInfinity::<ECEngine>::alloc(
          cs.namespace(|| "pk"),
          Some(self.pk_coords),
        )?;
        let gen_var = AllocatedPointNonInfinity::<ECEngine>::alloc(
          cs.namespace(|| "gen"),
          Some(self.gen_coords),
        )?;

        let deck_arr: [ElGamalCiphertextVar<ECEngine>; N] =
          deck_vars.try_into().ok().unwrap();
        let rand_arr: [AllocatedNum<Scalar>; N] = rand_vars.try_into().ok().unwrap();

        reencrypt_deck_bp::<ECEngine, _, N>(
          cs,
          &deck_arr,
          &rand_arr,
          &pk_var,
          &self.native_data,
          &gen_var,
          &self.gen_powers,
        )
      }
    }

    let circuit = ReencryptCircuit {
      input_cts: ct_arr,
      randomizations: rand_arr,
      pk_coords,
      gen_coords,
      native_data,
      gen_powers,
    };

    // Setup (ShapeCS → serial path)
    let (pk, vk) =
      SpartanSNARK::<SpartanE>::setup(circuit.clone()).expect("Setup failed");

    // Prep prove + prove (SatisfyingAssignment → parallel path)
    let prep = SpartanSNARK::<SpartanE>::prep_prove(&pk, circuit.clone(), false)
      .expect("Prep prove failed");
    let proof = SpartanSNARK::<SpartanE>::prove(&pk, circuit, &prep, false)
      .expect("Prove failed");

    // Verify — if this passes, parallel witness is cryptographically valid
    proof.verify(&vk).expect("Verification failed — parallel witness produced invalid proof");
  }

  #[test]
  fn test_parallel_vs_serial_witness_performance() {
    let (a, b, _, _) = <E as Engine>::GE::group_params();
    let gen_coords = find_point_on_curve(a, b);
    let num_bits = Base::NUM_BITS as usize;
    let gen_powers = precompute_fixed_base_powers(gen_coords, a, num_bits);

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

    // Serial witness timing — use reencrypt_deck_bp which now includes pk_powers precomputation
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
      let pk = AllocatedPointNonInfinity::<E>::alloc(
        cs.namespace(|| "pk"),
        Some((pk_coords.0, pk_coords.1)),
      )
      .unwrap();
      let gen_pt = AllocatedPointNonInfinity::<E>::alloc(
        cs.namespace(|| "gen"),
        Some(gen_coords),
      )
      .unwrap();

      let deck_arr: [ElGamalCiphertextVar<E>; N] = deck.try_into().ok().unwrap();
      let rand_arr: [AllocatedNum<Base>; N] = rands.try_into().ok().unwrap();

      // Precompute pk_powers for serial test
      let mut pk_power_vars = Vec::with_capacity(num_bits);
      pk_power_vars.push(pk.clone());
      for i in 1..num_bits {
        let next = pk_power_vars[i - 1]
          .double_incomplete(cs.namespace(|| format!("pk_power_{}", i)))
          .unwrap();
        pk_power_vars.push(next);
      }

      // Serial: run each card's gadget sequentially on the main CS
      for i in 0..N {
        let _ = rerandomize_ciphertext_bp::<E, _>(
          cs.namespace(|| format!("serial_reencrypt_{}", i)),
          &deck_arr[i],
          &rand_arr[i],
          &pk_power_vars,
          &gen_pt,
          &gen_powers,
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
      let pk = AllocatedPointNonInfinity::<E>::alloc(
        cs.namespace(|| "pk"),
        Some((pk_coords.0, pk_coords.1)),
      )
      .unwrap();
      let gen_pt = AllocatedPointNonInfinity::<E>::alloc(
        cs.namespace(|| "gen"),
        Some(gen_coords),
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
        &gen_pt,
        &gen_powers,
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
