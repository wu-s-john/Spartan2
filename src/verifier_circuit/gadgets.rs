//! Gadgets for in-circuit verification of Spartan ZK proof components.
//!
//! These gadgets verify the NIFS scalar fold and relaxed R1CS satisfiability,
//! which are currently checked natively by `SpartanZkSNARK::verify()`.

use crate::{gadgets::ecc::AllocatedPoint, r1cs::R1CSShape, traits::Engine};
use bellpepper_core::{
  ConstraintSystem, SynthesisError,
  boolean::AllocatedBit,
  num::AllocatedNum,
};
use ff::Field;

/// Enforces that `(u_f, X_f)` is the correct scalar fold of relaxed instance
/// `(u1, X1)` with regular instance `X2` (where u2 = 1) using challenge `r`.
///
/// Constraints:
///   u_f = u1 + r                         (1 linear)
///   X_f[k] = X1[k] + r · X2[k]   ∀k     (|X| muls)
pub fn enforce_nifs_scalar_fold<E: Engine, CS: ConstraintSystem<E::Scalar>>(
  mut cs: CS,
  u1: &AllocatedNum<E::Scalar>,
  X1: &[AllocatedNum<E::Scalar>],
  X2: &[AllocatedNum<E::Scalar>],
  r: &AllocatedNum<E::Scalar>,
  u_f: &AllocatedNum<E::Scalar>,
  X_f: &[AllocatedNum<E::Scalar>],
) -> Result<(), SynthesisError> {
  assert_eq!(X1.len(), X2.len());
  assert_eq!(X1.len(), X_f.len());

  // u_f = u1 + r  ⟺  (u_f - u1 - r) · 1 = 0
  cs.enforce(
    || "u_fold",
    |lc| lc + u_f.get_variable() - u1.get_variable() - r.get_variable(),
    |lc| lc + CS::one(),
    |lc| lc,
  );

  // For each k: X_f[k] = X1[k] + r · X2[k]
  for (k, ((x1, x2), xf)) in X1.iter().zip(X2.iter()).zip(X_f.iter()).enumerate() {
    // delta_k = r · X2[k]
    let delta = AllocatedNum::alloc(cs.namespace(|| format!("delta_{k}")), || {
      let r_val = r.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let x2_val = x2.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      Ok(r_val * x2_val)
    })?;
    cs.enforce(
      || format!("r_mul_X2_{k}"),
      |lc| lc + r.get_variable(),
      |lc| lc + x2.get_variable(),
      |lc| lc + delta.get_variable(),
    );

    // X_f[k] = X1[k] + delta_k  ⟺  (X_f[k] - X1[k] - delta_k) · 1 = 0
    cs.enforce(
      || format!("X_fold_{k}"),
      |lc| lc + xf.get_variable() - x1.get_variable() - delta.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc,
    );
  }

  Ok(())
}

/// Enforces relaxed R1CS satisfiability for a known, constant shape.
///
/// Checks: A·z_f ∘ B·z_f = u_f · C·z_f + E_f
///
/// The matrices A, B, C from `shape` are embedded as constant coefficients
/// in the linear combinations — they cost zero constraints.
/// Each constraint row costs 2 multiplication constraints.
pub fn enforce_relaxed_r1cs_sat<E: Engine, CS: ConstraintSystem<E::Scalar>>(
  mut cs: CS,
  shape: &R1CSShape<E>,
  z_f: &[AllocatedNum<E::Scalar>],
  u_f: &AllocatedNum<E::Scalar>,
  E_f: &[AllocatedNum<E::Scalar>],
) -> Result<(), SynthesisError> {
  assert_eq!(z_f.len(), shape.num_vars + 1 + shape.num_io);
  assert_eq!(E_f.len(), shape.num_cons);

  for i in 0..shape.num_cons {
    // Compute witness values for L_i, R_i, O_i
    let l_val = compute_lc_value(&shape.A, i, z_f);
    let r_val = compute_lc_value(&shape.B, i, z_f);
    let o_val = compute_lc_value(&shape.C, i, z_f);

    // P_i = L_i · R_i
    let P_i = AllocatedNum::alloc(cs.namespace(|| format!("P_{i}")), || {
      let l = l_val.ok_or(SynthesisError::AssignmentMissing)?;
      let r = r_val.ok_or(SynthesisError::AssignmentMissing)?;
      Ok(l * r)
    })?;
    cs.enforce(
      || format!("LR_{i}"),
      |lc| build_lc::<E>(&shape.A, i, z_f, lc),
      |lc| build_lc::<E>(&shape.B, i, z_f, lc),
      |lc| lc + P_i.get_variable(),
    );

    // Q_i = u_f · O_i
    let Q_i = AllocatedNum::alloc(cs.namespace(|| format!("Q_{i}")), || {
      let u = u_f.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let o = o_val.ok_or(SynthesisError::AssignmentMissing)?;
      Ok(u * o)
    })?;
    cs.enforce(
      || format!("uO_{i}"),
      |lc| lc + u_f.get_variable(),
      |lc| build_lc::<E>(&shape.C, i, z_f, lc),
      |lc| lc + Q_i.get_variable(),
    );

    // P_i - Q_i = E_f[i]  ⟺  (P_i - Q_i - E_f[i]) · 1 = 0
    cs.enforce(
      || format!("PQE_{i}"),
      |lc| lc + P_i.get_variable() - Q_i.get_variable() - E_f[i].get_variable(),
      |lc| lc + CS::one(),
      |lc| lc,
    );
  }

  Ok(())
}

/// Build a linear combination from a sparse matrix row.
/// row i of matrix M: sum_j M[i,j] * z_f[j]
fn build_lc<E: Engine>(
  matrix: &crate::r1cs::SparseMatrix<E::Scalar>,
  row: usize,
  z_f: &[AllocatedNum<E::Scalar>],
  lc: bellpepper_core::LinearCombination<E::Scalar>,
) -> bellpepper_core::LinearCombination<E::Scalar> {
  let start = matrix.indptr[row];
  let end = matrix.indptr[row + 1];
  (start..end).fold(lc, |lc, pos| {
    let col = matrix.indices[pos];
    let val = matrix.data[pos];
    lc + (val, z_f[col].get_variable())
  })
}

/// Enforces that `G_out = G_running + r · G_new` using EC operations
/// native to the circuit's field.
///
/// This gadget operates over `E::Base` (the curve's coordinate field).
/// For Pallas points, `E::Base = F_p`, so this lives in a circuit over F_p.
///
/// Cost: ~254 constraints (scalar mul) + ~6 constraints (point add) + 2 (equality).
pub fn enforce_accumulator_fold<E: Engine, CS: ConstraintSystem<E::Base>>(
  mut cs: CS,
  g_running: &AllocatedPoint<E>,
  g_new: &AllocatedPoint<E>,
  r_bits: &[AllocatedBit],
  g_out: &AllocatedPoint<E>,
) -> Result<(), SynthesisError> {
  // R = r · G_new
  let r_scaled = g_new.scalar_mul(cs.namespace(|| "r_mul_g_new"), r_bits)?;

  // G_computed = G_running + R
  let g_computed = g_running.add(cs.namespace(|| "g_running_add_r"), &r_scaled)?;

  // Assert G_computed == G_out (coordinate equality)
  cs.enforce(
    || "g_out_x_eq",
    |lc| lc + g_computed.x.get_variable() - g_out.x.get_variable(),
    |lc| lc + CS::one(),
    |lc| lc,
  );
  cs.enforce(
    || "g_out_y_eq",
    |lc| lc + g_computed.y.get_variable() - g_out.y.get_variable(),
    |lc| lc + CS::one(),
    |lc| lc,
  );

  Ok(())
}

/// Pre-derived IPA challenges for in-circuit verification.
///
/// These are normally derived via Fiat-Shamir (Poseidon). Until Poseidon is
/// available in-circuit, the challenges are passed as witnesses and the
/// transcript binding is deferred to the native verifier.
pub struct IpaChallenges<F: ff::PrimeField> {
  /// ξ: challenge for combining P and S
  pub xi: AllocatedNum<F>,
  /// z: inner-product binding challenge
  pub z: AllocatedNum<F>,
  /// Per-round challenges u₀, ..., u_{k-1}
  pub u: Vec<AllocatedNum<F>>,
  /// Per-round inverse challenges u₀⁻¹, ..., u_{k-1}⁻¹
  pub u_inv: Vec<AllocatedNum<F>>,
}

/// Verifies the IPA MSM equation in-circuit:
///
/// ```text
/// C + ξ·S_comm - v·g₀ + Σⱼ(uⱼ⁻¹·Lⱼ + uⱼ·Rⱼ) - c·b₀·z·U - f·W - c·G = identity
/// ```
///
/// Operates over `E::Base` using `AllocatedPoint<E>` for EC arithmetic.
/// Challenges are passed as pre-derived witnesses (Poseidon binding deferred).
///
/// Cost: ~(2k+5) EC scalar muls + ~(2k+5) EC adds + k inversions + b₀ computation
pub fn enforce_ipa_msm_check<E: Engine, CS: ConstraintSystem<E::Base>>(
  mut cs: CS,
  // Public parameters
  params_g0: &AllocatedPoint<E>,
  params_u: &AllocatedPoint<E>,
  params_w: &AllocatedPoint<E>,
  // Proof elements
  commitment: &AllocatedPoint<E>,
  s_comm: &AllocatedPoint<E>,
  l_vec: &[AllocatedPoint<E>],
  r_vec: &[AllocatedPoint<E>],
  c_bits: &[AllocatedBit],
  f_bits: &[AllocatedBit],
  v_bits: &[AllocatedBit],
  g_claimed: &AllocatedPoint<E>,
  // Pre-derived challenges (will be replaced by in-circuit Poseidon later)
  challenges: &IpaChallenges<E::Base>,
  // Evaluation point x (for b₀ computation)
  x: &AllocatedNum<E::Base>,
) -> Result<(), SynthesisError>
where
  E::Base: ff::PrimeFieldBits,
{
  let k = l_vec.len();
  assert_eq!(r_vec.len(), k);
  assert_eq!(challenges.u.len(), k);
  assert_eq!(challenges.u_inv.len(), k);

  // ── Verify inversions: u_j · u_j⁻¹ = 1 ──
  for j in 0..k {
    cs.enforce(
      || format!("u_inv_check_{j}"),
      |lc| lc + challenges.u[j].get_variable(),
      |lc| lc + challenges.u_inv[j].get_variable(),
      |lc| lc + CS::one(),
    );
  }

  // ── Compute b₀ = ∏ⱼ (1 + uⱼ · x^{2^{k-1-j}}) ──
  // First compute x powers: x, x², x⁴, ..., x^{2^{k-1}}
  let mut x_powers = Vec::with_capacity(k);
  x_powers.push(x.clone());
  for t in 1..k {
    let prev = &x_powers[t - 1];
    let sq = AllocatedNum::alloc(cs.namespace(|| format!("x_pow2_{t}")), || {
      let v = prev.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      Ok(v * v)
    })?;
    cs.enforce(
      || format!("x_sq_{t}"),
      |lc| lc + prev.get_variable(),
      |lc| lc + prev.get_variable(),
      |lc| lc + sq.get_variable(),
    );
    x_powers.push(sq);
  }
  // x_powers[t] = x^{2^t}

  // Compute each factor: f_j = 1 + u_j · x^{2^{k-1-j}}
  // Then b₀ = ∏ f_j
  let mut b0 = AllocatedNum::alloc(cs.namespace(|| "b0_init"), || Ok(E::Base::ONE))?;
  cs.enforce(
    || "b0_init_eq_one",
    |lc| lc + b0.get_variable(),
    |lc| lc + CS::one(),
    |lc| lc + CS::one(),
  );

  for j in 0..k {
    // t_j = u_j · x^{2^{k-1-j}}
    let x_pow = &x_powers[k - 1 - j];
    let t_j = AllocatedNum::alloc(cs.namespace(|| format!("t_{j}")), || {
      let u = challenges.u[j].get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let xp = x_pow.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      Ok(u * xp)
    })?;
    cs.enforce(
      || format!("t_{j}_def"),
      |lc| lc + challenges.u[j].get_variable(),
      |lc| lc + x_pow.get_variable(),
      |lc| lc + t_j.get_variable(),
    );

    // f_j = 1 + t_j (linear, no constraint needed — just track the expression)
    // b0_new = b0 · (1 + t_j)
    let b0_new = AllocatedNum::alloc(cs.namespace(|| format!("b0_{j}")), || {
      let b = b0.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let t = t_j.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      Ok(b * (E::Base::ONE + t))
    })?;
    cs.enforce(
      || format!("b0_mul_{j}"),
      |lc| lc + b0.get_variable(),
      |lc| lc + CS::one() + t_j.get_variable(),
      |lc| lc + b0_new.get_variable(),
    );
    b0 = b0_new;
  }

  // ── Compute α = c · b₀ · z  (need c and z as field elements) ──
  // c is given as bits; reconstruct as field element
  let c_field = bits_to_num(cs.namespace(|| "c_field"), c_bits)?;
  let _f_field = bits_to_num(cs.namespace(|| "f_field"), f_bits)?;

  // c_b0 = c · b₀
  let c_b0 = AllocatedNum::alloc(cs.namespace(|| "c_b0"), || {
    let c = c_field.get_value().ok_or(SynthesisError::AssignmentMissing)?;
    let b = b0.get_value().ok_or(SynthesisError::AssignmentMissing)?;
    Ok(c * b)
  })?;
  cs.enforce(
    || "c_b0_def",
    |lc| lc + c_field.get_variable(),
    |lc| lc + b0.get_variable(),
    |lc| lc + c_b0.get_variable(),
  );

  // alpha = c_b0 · z
  let alpha = AllocatedNum::alloc(cs.namespace(|| "alpha"), || {
    let cb = c_b0.get_value().ok_or(SynthesisError::AssignmentMissing)?;
    let z = challenges.z.get_value().ok_or(SynthesisError::AssignmentMissing)?;
    Ok(cb * z)
  })?;
  cs.enforce(
    || "alpha_def",
    |lc| lc + c_b0.get_variable(),
    |lc| lc + challenges.z.get_variable(),
    |lc| lc + alpha.get_variable(),
  );

  // ── Decompose all scalars to bits for EC scalar mul ──
  let xi_bits = num_to_bits(cs.namespace(|| "xi_bits"), &challenges.xi)?;
  let alpha_bits = num_to_bits(cs.namespace(|| "alpha_bits"), &alpha)?;

  let mut u_bits = Vec::with_capacity(k);
  let mut u_inv_bits = Vec::with_capacity(k);
  for j in 0..k {
    u_bits.push(num_to_bits(
      cs.namespace(|| format!("u_bits_{j}")),
      &challenges.u[j],
    )?);
    u_inv_bits.push(num_to_bits(
      cs.namespace(|| format!("u_inv_bits_{j}")),
      &challenges.u_inv[j],
    )?);
  }

  // ── EC scalar multiplications ──

  // T_S = ξ · S_comm
  let t_s = s_comm.scalar_mul(cs.namespace(|| "xi_mul_s_comm"), &xi_bits)?;

  // T_v = v · g₀
  let t_v = params_g0.scalar_mul(cs.namespace(|| "v_mul_g0"), v_bits)?;

  // T_Lj = u_j⁻¹ · L[j], T_Rj = u_j · R[j]
  let mut t_lr = Vec::with_capacity(2 * k);
  for j in 0..k {
    let t_l =
      l_vec[j].scalar_mul(cs.namespace(|| format!("u_inv_mul_L_{j}")), &u_inv_bits[j])?;
    let t_r = r_vec[j].scalar_mul(cs.namespace(|| format!("u_mul_R_{j}")), &u_bits[j])?;
    t_lr.push(t_l);
    t_lr.push(t_r);
  }

  // T_U = α · U
  let t_u = params_u.scalar_mul(cs.namespace(|| "alpha_mul_U"), &alpha_bits)?;

  // T_W = f · W
  let t_w = params_w.scalar_mul(cs.namespace(|| "f_mul_W"), f_bits)?;

  // T_G = c · G_claimed
  let t_g = g_claimed.scalar_mul(cs.namespace(|| "c_mul_G"), c_bits)?;

  // ── Sum all terms: C + T_S - T_v + Σ(T_Lj + T_Rj) - T_U - T_W - T_G ──

  // Start with commitment
  let mut sum = commitment.clone();

  // + T_S
  sum = sum.add(cs.namespace(|| "add_t_s"), &t_s)?;

  // - T_v
  let neg_t_v = t_v.negate(cs.namespace(|| "neg_t_v"))?;
  sum = sum.add(cs.namespace(|| "sub_t_v"), &neg_t_v)?;

  // + T_Lj + T_Rj for each round
  for (idx, t) in t_lr.iter().enumerate() {
    sum = sum.add(cs.namespace(|| format!("add_t_lr_{idx}")), t)?;
  }

  // - T_U
  let neg_t_u = t_u.negate(cs.namespace(|| "neg_t_u"))?;
  sum = sum.add(cs.namespace(|| "sub_t_u"), &neg_t_u)?;

  // - T_W
  let neg_t_w = t_w.negate(cs.namespace(|| "neg_t_w"))?;
  sum = sum.add(cs.namespace(|| "sub_t_w"), &neg_t_w)?;

  // - T_G
  let neg_t_g = t_g.negate(cs.namespace(|| "neg_t_g"))?;
  sum = sum.add(cs.namespace(|| "sub_t_g"), &neg_t_g)?;

  // ── Check sum is identity (point at infinity) ──
  cs.enforce(
    || "sum_is_infinity",
    |lc| lc + sum.is_infinity.get_variable(),
    |lc| lc + CS::one(),
    |lc| lc + CS::one(),
  );

  Ok(())
}

/// Reconstruct a field element from allocated bits (little-endian).
fn bits_to_num<F: ff::PrimeField, CS: ConstraintSystem<F>>(
  mut cs: CS,
  bits: &[AllocatedBit],
) -> Result<AllocatedNum<F>, SynthesisError> {
  let mut value = Some(F::ZERO);
  let mut coeff = F::ONE;
  for bit in bits {
    if let Some(ref mut v) = value {
      if bit.get_value().unwrap_or(false) {
        *v += coeff;
      }
    }
    coeff = coeff.double();
  }

  let num = AllocatedNum::alloc(cs.namespace(|| "bits_to_num"), || {
    value.ok_or(SynthesisError::AssignmentMissing)
  })?;

  // Enforce: num = Σ bit_i · 2^i
  let mut lc_sum = bellpepper_core::LinearCombination::<F>::zero();
  let mut coeff = F::ONE;
  for bit in bits {
    lc_sum = lc_sum + (coeff, bit.get_variable());
    coeff = coeff.double();
  }
  cs.enforce(
    || "bits_to_num_check",
    |_| lc_sum,
    |lc| lc + CS::one(),
    |lc| lc + num.get_variable(),
  );

  Ok(num)
}

/// Decompose an allocated field element into bits (little-endian).
fn num_to_bits<F: ff::PrimeFieldBits, CS: ConstraintSystem<F>>(
  mut cs: CS,
  num: &AllocatedNum<F>,
) -> Result<Vec<AllocatedBit>, SynthesisError> {
  let bits_val: Vec<Option<bool>> = match num.get_value() {
    Some(v) => v.to_le_bits().iter().by_vals().map(Some).collect(),
    None => vec![None; F::NUM_BITS as usize],
  };

  let mut bits = Vec::with_capacity(F::NUM_BITS as usize);
  for (i, b) in bits_val.iter().enumerate().take(F::NUM_BITS as usize) {
    let bit = AllocatedBit::alloc(cs.namespace(|| format!("bit_{i}")), *b)?;
    bits.push(bit);
  }

  // Enforce reconstruction: num = Σ bit_i · 2^i
  let mut lc_sum = bellpepper_core::LinearCombination::<F>::zero();
  let mut coeff = F::ONE;
  for bit in &bits {
    lc_sum = lc_sum + (coeff, bit.get_variable());
    coeff = coeff.double();
  }
  cs.enforce(
    || "num_to_bits_check",
    |_| lc_sum,
    |lc| lc + CS::one(),
    |lc| lc + num.get_variable(),
  );

  Ok(bits)
}

/// Compute the value of a linear combination from a sparse matrix row.
fn compute_lc_value<F: ff::PrimeField>(
  matrix: &crate::r1cs::SparseMatrix<F>,
  row: usize,
  z_f: &[AllocatedNum<F>],
) -> Option<F> {
  let start = matrix.indptr[row];
  let end = matrix.indptr[row + 1];
  let mut acc = F::ZERO;
  for pos in start..end {
    let col = matrix.indices[pos];
    let val = matrix.data[pos];
    acc += val * z_f[col].get_value()?;
  }
  Some(acc)
}
