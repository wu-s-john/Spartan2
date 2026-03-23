//! Log-sized Inner Product Argument (IPA) polynomial commitment scheme
//! with Halo-style accumulation.
//!
//! This implements the IPA from halo2: k rounds of folding produce an O(log n)
//! proof for opening a univariate polynomial commitment. The Guard/Accumulator
//! types support deferred verification for recursive proof composition.

use crate::{
  errors::SpartanError,
  provider::traits::{DlogGroup, DlogGroupExt},
  traits::{
    Engine,
    transcript::TranscriptEngineTrait,
  },
};
use ff::Field;
use rand_core::OsRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Compute b₀ = ∏ⱼ (1 + u_{k-1-j} · x^{2^j}).
///
/// `challenges` are stored in round order `[u₀, u₁, ..., u_{k-1}]`.
fn compute_b<F: Field>(x: &F, challenges: &[F]) -> F {
  let mut result = F::ONE;
  let mut cur = *x;
  for u_j in challenges.iter().rev() {
    result *= F::ONE + *u_j * cur;
    cur = cur.square();
  }
  result
}

/// Compute the coefficient vector s of `g(X) = init · ∏ⱼ (1 + u_{k-1-j} · X^{2^j})`.
///
/// Output has length `2^k`. `challenges` are in round order `[u₀, ..., u_{k-1}]`.
fn compute_s<F: Field>(challenges: &[F], init: &F) -> Vec<F> {
  assert!(!challenges.is_empty());
  let n = 1 << challenges.len();
  let mut v = vec![F::ZERO; n];
  v[0] = *init;

  for (len, u_j) in challenges
    .iter()
    .rev()
    .enumerate()
    .map(|(i, u_j)| (1 << i, u_j))
  {
    let (left, right) = v.split_at_mut(len);
    let right = &mut right[0..len];
    right.copy_from_slice(left);
    for val in right.iter_mut() {
      *val *= u_j;
    }
  }
  v
}

/// Parallel inner product of two vectors.
fn inner_product<F: Field + Send + Sync>(a: &[F], b: &[F]) -> F {
  assert_eq!(a.len(), b.len());
  a.par_iter()
    .zip(b.par_iter())
    .map(|(ai, bi)| *ai * *bi)
    .reduce(|| F::ZERO, |x, y| x + y)
}

/// Evaluate a polynomial (given as coefficients) at point x.
fn eval_polynomial<F: Field>(poly: &[F], x: &F) -> F {
  let mut result = F::ZERO;
  let mut power = F::ONE;
  for coeff in poly {
    result += *coeff * power;
    power *= x;
  }
  result
}

// ---------------------------------------------------------------------------
// DeferredMSM — lazy multi-scalar multiplication accumulator
// ---------------------------------------------------------------------------

/// A deferred multi-scalar multiplication that collects (scalar, point) pairs
/// and evaluates them lazily. Supports scaling and merging for batch verification.
pub struct DeferredMSM<E: Engine>
where
  E::GE: DlogGroup,
{
  scalars: Vec<E::Scalar>,
  bases: Vec<<E::GE as DlogGroup>::AffineGroupElement>,
}

impl<E: Engine> DeferredMSM<E>
where
  E::GE: DlogGroupExt,
{
  /// Create an empty MSM accumulator.
  pub fn new() -> Self {
    Self {
      scalars: Vec::new(),
      bases: Vec::new(),
    }
  }

  /// Add a term: `scalar * point` (projective).
  pub fn add_term(&mut self, scalar: E::Scalar, point: &E::GE) {
    self.scalars.push(scalar);
    self.bases.push(point.affine());
  }

  /// Add a term: `scalar * point` (affine).
  pub fn add_term_affine(
    &mut self,
    scalar: E::Scalar,
    point: &<E::GE as DlogGroup>::AffineGroupElement,
  ) {
    self.scalars.push(scalar);
    self.bases.push(point.clone());
  }

  /// Scale all accumulated terms by `factor`.
  pub fn scale(&mut self, factor: &E::Scalar) {
    self.scalars.par_iter_mut().for_each(|s| *s *= factor);
  }

  /// Merge another MSM into this one.
  pub fn add_msm(&mut self, other: &Self) {
    self.scalars.extend_from_slice(&other.scalars);
    self.bases.extend_from_slice(&other.bases);
  }

  /// Evaluate the MSM, returning the resulting group element.
  pub fn eval(&self) -> Result<E::GE, SpartanError> {
    if self.scalars.is_empty() {
      return Ok(E::GE::zero());
    }
    E::GE::vartime_multiscalar_mul(&self.scalars, &self.bases)
  }

  /// Check whether the MSM evaluates to the identity (zero) point.
  pub fn is_identity(&self) -> Result<bool, SpartanError> {
    Ok(self.eval()? == E::GE::zero())
  }

  /// Batch-verify multiple MSMs: scale each by a random factor,
  /// sum them, and check the result is the identity.
  pub fn batch_verify(msms: Vec<Self>) -> Result<bool, SpartanError> {
    let mut combined = Self::new();
    for mut msm in msms {
      let r = E::Scalar::random(&mut OsRng);
      msm.scale(&r);
      combined.add_msm(&msm);
    }
    combined.is_identity()
  }
}

// ---------------------------------------------------------------------------
// IpaParams — universal reference string
// ---------------------------------------------------------------------------

/// Parameters (URS) for the log-sized IPA scheme.
///
/// Contains `n = 2^k` generators for polynomial commitments, plus
/// dedicated blinding (`W`) and inner-product binding (`U`) points.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct IpaParams<E: Engine>
where
  E::GE: DlogGroup,
{
  /// log₂ of the polynomial degree bound.
  pub k: u32,
  /// `n = 2^k`, the polynomial degree bound.
  pub n: usize,
  /// Generator vector of length n.
  pub g: Vec<<E::GE as DlogGroup>::AffineGroupElement>,
  /// Blinding generator W.
  pub w: <E::GE as DlogGroup>::AffineGroupElement,
  /// Inner product binding point U.
  pub u: <E::GE as DlogGroup>::AffineGroupElement,
}

impl<E: Engine> IpaParams<E>
where
  E::GE: DlogGroupExt,
{
  /// Create new parameters for polynomials of degree `< 2^k`.
  pub fn new(k: u32) -> Self {
    let n = 1usize << k;
    let g = E::GE::from_label(b"ipa_halo2_g", n);
    let w_points = E::GE::from_label(b"ipa_halo2_w", 1);
    let u_points = E::GE::from_label(b"ipa_halo2_u", 1);
    IpaParams {
      k,
      n,
      g,
      w: w_points[0].clone(),
      u: u_points[0].clone(),
    }
  }

  /// Commit to a polynomial with the given blinding factor.
  ///
  /// Returns `Commit(poly) = MSM(poly, g) + [blind] · W`.
  pub fn commit(&self, poly: &[E::Scalar], blind: &E::Scalar) -> Result<E::GE, SpartanError> {
    assert!(poly.len() <= self.n);
    let comm = E::GE::vartime_multiscalar_mul(poly, &self.g[..poly.len()])?;
    Ok(comm + E::GE::group(&self.w) * *blind)
  }
}

// ---------------------------------------------------------------------------
// IpaProof — the log-sized opening proof
// ---------------------------------------------------------------------------

/// A log-sized IPA opening proof.
///
/// Proves that a committed polynomial P evaluates to v at point x.
/// Proof size is O(log n): 2k group elements (L, R) plus 2 scalars (c, f)
/// and 1 group element (S commitment).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct IpaProof<E: Engine>
where
  E::GE: DlogGroup,
{
  /// Commitment to the random blinding polynomial S.
  s_comm: E::GE,
  /// Left cross-term commitments, one per round.
  l_vec: Vec<E::GE>,
  /// Right cross-term commitments, one per round.
  r_vec: Vec<E::GE>,
  /// Final folded scalar c = p'[0] after k rounds.
  c: E::Scalar,
  /// Accumulated synthetic blinding factor.
  f: E::Scalar,
}

impl<E: Engine> IpaProof<E>
where
  E::GE: DlogGroupExt,
{
  /// Create an opening proof that polynomial `poly` (committed as `commitment`)
  /// evaluates to `v` at point `x`.
  #[allow(clippy::too_many_arguments)]
  pub fn create(
    params: &IpaParams<E>,
    transcript: &mut E::TE,
    poly: &[E::Scalar],
    blind: &E::Scalar,
    x: &E::Scalar,
    v: &E::Scalar,
    commitment: &E::GE,
  ) -> Result<Self, SpartanError> {
    let k = params.k as usize;
    let n = params.n;
    assert_eq!(poly.len(), n);

    // Domain-separate and absorb public inputs
    transcript.dom_sep(b"ipa_halo2");
    transcript.absorb(b"commitment", commitment);
    transcript.absorb(b"x", x);
    transcript.absorb(b"v", v);

    // Step 1: Sample random polynomial S with S(x) = 0
    let mut s_poly: Vec<E::Scalar> = (0..n).map(|_| E::Scalar::random(&mut OsRng)).collect();
    let s_at_x = eval_polynomial(&s_poly, x);
    s_poly[0] -= s_at_x; // now S(x) = 0
    let blind_s = E::Scalar::random(&mut OsRng);
    let s_comm = params.commit(&s_poly, &blind_s)?;

    transcript.absorb(b"s_comm", &s_comm);

    // Step 2: Squeeze challenges xi, z
    let xi = transcript.squeeze(b"xi")?;
    let z = transcript.squeeze(b"z")?;

    // Step 3: Form P' = P + xi * S, then subtract evaluation to get root at x
    let mut p_prime: Vec<E::Scalar> = poly
      .par_iter()
      .zip(s_poly.par_iter())
      .map(|(p, s)| *p + xi * *s)
      .collect();
    let v_prime = eval_polynomial(&p_prime, x);
    p_prime[0] -= v_prime;
    let blind_prime = *blind + xi * blind_s;

    // Accumulated blinding factor
    let mut f_acc = blind_prime;

    // Step 4: Build b = [1, x, x², ..., x^{n-1}]
    let mut b: Vec<E::Scalar> = Vec::with_capacity(n);
    {
      let mut cur = E::Scalar::ONE;
      for _ in 0..n {
        b.push(cur);
        cur *= x;
      }
    }

    // Mutable generators — folded each round
    let mut g_prime: Vec<<E::GE as DlogGroup>::AffineGroupElement> = params.g.clone();

    let mut l_vec = Vec::with_capacity(k);
    let mut r_vec = Vec::with_capacity(k);

    // Step 5: k rounds of folding
    for j in 0..k {
      let half = 1 << (k - j - 1);

      // Compute cross-terms
      let ip_l = inner_product(&p_prime[half..], &b[..half]);
      let ip_r = inner_product(&p_prime[..half], &b[half..]);

      let blind_l = E::Scalar::random(&mut OsRng);
      let blind_r = E::Scalar::random(&mut OsRng);

      // L_j = MSM(p_hi, g_lo) + [ip_l * z] U + [blind_l] W
      let l_msm = E::GE::vartime_multiscalar_mul(&p_prime[half..], &g_prime[..half])?;
      let l_j =
        l_msm + E::GE::group(&params.u) * (ip_l * z) + E::GE::group(&params.w) * blind_l;

      // R_j = MSM(p_lo, g_hi) + [ip_r * z] U + [blind_r] W
      let r_msm = E::GE::vartime_multiscalar_mul(&p_prime[..half], &g_prime[half..])?;
      let r_j =
        r_msm + E::GE::group(&params.u) * (ip_r * z) + E::GE::group(&params.w) * blind_r;

      transcript.absorb(b"L", &l_j);
      transcript.absorb(b"R", &r_j);

      let u_j = transcript.squeeze(b"u")?;
      let u_j_inv = u_j.invert().unwrap();

      l_vec.push(l_j);
      r_vec.push(r_j);

      // Fold p' and b
      let mut p_new = Vec::with_capacity(half);
      let mut b_new = Vec::with_capacity(half);
      for i in 0..half {
        p_new.push(p_prime[i] + u_j_inv * p_prime[i + half]);
        b_new.push(b[i] + u_j * b[i + half]);
      }
      p_prime = p_new;
      b = b_new;

      // Fold generators: g'_i = g_lo_i + u_j * g_hi_i
      let g_folded: Vec<_> = (0..half)
        .into_par_iter()
        .map(|i| {
          (E::GE::group(&g_prime[i]) + E::GE::group(&g_prime[i + half]) * u_j).affine()
        })
        .collect();
      g_prime = g_folded;

      // Update blinding: f += u_j^{-1} * blind_l + u_j * blind_r
      f_acc += u_j_inv * blind_l + u_j * blind_r;
    }

    assert_eq!(p_prime.len(), 1);
    let c = p_prime[0];

    Ok(IpaProof {
      s_comm,
      l_vec,
      r_vec,
      c,
      f: f_acc,
    })
  }

  /// Verify the opening proof, returning a [`Guard`] with the deferred
  /// `[-c]G'₀` term for accumulation.
  pub fn verify(
    &self,
    params: &IpaParams<E>,
    transcript: &mut E::TE,
    commitment: &E::GE,
    x: &E::Scalar,
    v: &E::Scalar,
  ) -> Result<Guard<E>, SpartanError> {
    let k = params.k as usize;

    if self.l_vec.len() != k || self.r_vec.len() != k {
      return Err(SpartanError::InvalidPCS {
        reason: format!(
          "IPA proof has {} L and {} R elements, expected {}",
          self.l_vec.len(),
          self.r_vec.len(),
          k
        ),
      });
    }

    // Replay transcript
    transcript.dom_sep(b"ipa_halo2");
    transcript.absorb(b"commitment", commitment);
    transcript.absorb(b"x", x);
    transcript.absorb(b"v", v);
    transcript.absorb(b"s_comm", &self.s_comm);

    let xi = transcript.squeeze(b"xi")?;
    let z = transcript.squeeze(b"z")?;

    // Build MSM: start with P' = commitment + [xi] S_comm
    let mut msm = DeferredMSM::<E>::new();
    msm.add_term(E::Scalar::ONE, commitment);
    msm.add_term(xi, &self.s_comm);

    // P' also includes -[v] g₀ (the constant term adjustment).
    // In halo2, this is msm.add_constant_term(-v), which subtracts v from the g₀ scalar.
    // Here, we add [-v] * g₀.
    msm.add_term_affine(-*v, &params.g[0]);

    // Process k rounds
    let mut challenges = Vec::with_capacity(k);
    let mut challenges_inv = Vec::with_capacity(k);

    for j in 0..k {
      transcript.absorb(b"L", &self.l_vec[j]);
      transcript.absorb(b"R", &self.r_vec[j]);

      let u_j = transcript.squeeze(b"u")?;
      let u_j_inv = u_j.invert().unwrap();

      msm.add_term(u_j_inv, &self.l_vec[j]);
      msm.add_term(u_j, &self.r_vec[j]);

      challenges.push(u_j);
      challenges_inv.push(u_j_inv);
    }

    // Compute b₀ = ∏(1 + u_{k-1-j} · x^{2^j})
    let b0 = compute_b(x, &challenges);

    // Add [-c·b₀·z] U + [-f] W
    let neg_c = -self.c;
    msm.add_term_affine(neg_c * b0 * z, &params.u);
    msm.add_term_affine(-self.f, &params.w);

    Ok(Guard {
      msm,
      neg_c,
      challenges,
      g: params.g.clone(),
    })
  }
}

// ---------------------------------------------------------------------------
// Guard — deferred verification state
// ---------------------------------------------------------------------------

/// A guard holding partially-verified IPA state.
///
/// The MSM is complete except for the `[-c] G'₀` term where
/// `G'₀ = ⟨s, g⟩`. Three resolution paths are available:
/// - [`use_challenges`](Guard::use_challenges): immediate full verification
/// - [`use_g`](Guard::use_g): accept a claimed G and defer its check
/// - [`compute_g`](Guard::compute_g): compute the correct G (expensive)
pub struct Guard<E: Engine>
where
  E::GE: DlogGroup,
{
  msm: DeferredMSM<E>,
  neg_c: E::Scalar,
  challenges: Vec<E::Scalar>,
  g: Vec<<E::GE as DlogGroup>::AffineGroupElement>,
}

impl<E: Engine> Guard<E>
where
  E::GE: DlogGroupExt,
{
  /// Immediate verification: compute `s`, perform the full MSM `G'₀ = ⟨s, g⟩`,
  /// add `[-c]G'₀` to the accumulated MSM, and return the complete MSM.
  ///
  /// This is `O(n)` work.
  pub fn use_challenges(mut self) -> Result<DeferredMSM<E>, SpartanError> {
    let s = compute_s(&self.challenges, &E::Scalar::ONE);
    let g_prime = E::GE::vartime_multiscalar_mul(&s, &self.g)?;
    self.msm.add_term(self.neg_c, &g_prime);
    Ok(self.msm)
  }

  /// Accept a claimed `G` point, add `[-c]G` to the MSM, and return both
  /// the complete MSM (for immediate checking) and an [`Accumulator`]
  /// that records the deferred claim `G = ⟨s, g⟩`.
  ///
  /// This is `O(1)` work.
  pub fn use_g(mut self, g_claimed: &E::GE) -> (DeferredMSM<E>, Accumulator<E>) {
    self.msm.add_term(self.neg_c, g_claimed);
    let acc = Accumulator {
      g: *g_claimed,
      challenges: self.challenges,
    };
    (self.msm, acc)
  }

  /// Compute the correct `G'₀ = ⟨s, g⟩` from the generators and challenges.
  ///
  /// This is `O(n)` work.
  pub fn compute_g(&self) -> Result<E::GE, SpartanError> {
    let s = compute_s(&self.challenges, &E::Scalar::ONE);
    E::GE::vartime_multiscalar_mul(&s, &self.g)
  }
}

// ---------------------------------------------------------------------------
// Accumulator — deferred G verification claim
// ---------------------------------------------------------------------------

/// An accumulator for deferred IPA verification.
///
/// Records the claim that `g = ⟨s(challenges), params.g⟩`. The expensive
/// `O(n)` check is deferred to the final [`decide`](Accumulator::decide) step.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Accumulator<E: Engine>
where
  E::GE: DlogGroup,
{
  /// The claimed folded generator `G = ⟨s, g⟩`.
  pub g: E::GE,
  /// The IPA round challenges `[u₀, ..., u_{k-1}]`.
  pub challenges: Vec<E::Scalar>,
}

impl<E: Engine> Accumulator<E>
where
  E::GE: DlogGroupExt,
{
  /// The decider: verify that `self.g == ⟨s(challenges), params.g⟩`.
  ///
  /// This performs the deferred `O(n)` MSM.
  pub fn decide(&self, params: &IpaParams<E>) -> Result<bool, SpartanError> {
    let s = compute_s(&self.challenges, &E::Scalar::ONE);
    let expected = E::GE::vartime_multiscalar_mul(&s, &params.g)?;
    Ok(expected == self.g)
  }

  /// Fold two accumulators into an [`AccumulatedS`] using challenge `r`.
  ///
  /// Computes `G_new = G₁ + r·G₂` and `s_new = s(ch₁) + r·s(ch₂)`.
  pub fn fold(acc1: &Self, acc2: &Self, r: &E::Scalar) -> AccumulatedS<E> {
    let s1 = compute_s(&acc1.challenges, &E::Scalar::ONE);
    let s2 = compute_s(&acc2.challenges, &E::Scalar::ONE);

    let s_new: Vec<E::Scalar> = s1
      .par_iter()
      .zip(s2.par_iter())
      .map(|(a, b)| *a + *r * *b)
      .collect();

    let g_new = acc1.g + acc2.g * *r;

    AccumulatedS {
      s: s_new,
      g_acc: g_new,
    }
  }

  /// Fold N accumulators into one [`AccumulatedS`] using Fiat-Shamir challenges.
  ///
  /// The first accumulator enters unscaled; each subsequent one is scaled by
  /// a fresh challenge squeezed from the transcript.
  pub fn fold_all(
    accumulators: &[Self],
    transcript: &mut E::TE,
  ) -> Result<AccumulatedS<E>, SpartanError> {
    assert!(accumulators.len() >= 2);

    for acc in accumulators {
      transcript.absorb(b"G", &acc.g);
    }

    let mut result = AccumulatedS {
      s: compute_s(&accumulators[0].challenges, &E::Scalar::ONE),
      g_acc: accumulators[0].g,
    };

    for acc in &accumulators[1..] {
      let r = transcript.squeeze(b"r_acc")?;
      result.fold_next(acc, &r);
    }

    Ok(result)
  }
}

// ---------------------------------------------------------------------------
// AccumulatedS — folded accumulator state
// ---------------------------------------------------------------------------

/// Accumulated s-vector state for the decider.
///
/// After folding multiple accumulators, the s vector loses its tensor-product
/// structure (it can no longer be represented as k challenges) and must be
/// stored explicitly as a length-n vector.
pub struct AccumulatedS<E: Engine>
where
  E::GE: DlogGroup,
{
  /// Combined s-vector: `s₁ + r₂·s₂ + r₃·s₃ + ...`, length `2^k`.
  pub s: Vec<E::Scalar>,
  /// Accumulated G point: `G₁ + r₂·G₂ + r₃·G₃ + ...`.
  pub g_acc: E::GE,
}

impl<E: Engine> AccumulatedS<E>
where
  E::GE: DlogGroupExt,
{
  /// Fold another accumulator into this state using challenge `r`.
  pub fn fold_next(&mut self, acc: &Accumulator<E>, r: &E::Scalar) {
    let s_acc = compute_s(&acc.challenges, &E::Scalar::ONE);
    self
      .s
      .par_iter_mut()
      .zip(s_acc.par_iter())
      .for_each(|(si, sa)| *si += *r * *sa);
    self.g_acc = self.g_acc + acc.g * *r;
  }

  /// The decider: verify that `g_acc == MSM(s, params.g)`.
  ///
  /// Performs one `O(n)` MSM regardless of how many accumulators were folded.
  pub fn decide(&self, params: &IpaParams<E>) -> Result<bool, SpartanError> {
    let expected = E::GE::vartime_multiscalar_mul(&self.s, &params.g)?;
    Ok(expected == self.g_acc)
  }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::PallasHyraxEngine;
  use ff::Field;
  use rand_core::OsRng;

  type E = PallasHyraxEngine;
  type Scalar = <E as Engine>::Scalar;

  /// Helper: random polynomial of degree n-1
  fn random_poly(n: usize) -> Vec<Scalar> {
    (0..n).map(|_| Scalar::random(&mut OsRng)).collect()
  }

  /// Helper: evaluate polynomial at point
  fn eval_poly(poly: &[Scalar], x: &Scalar) -> Scalar {
    eval_polynomial(poly, x)
  }

  #[test]
  fn test_compute_b_consistency() {
    // compute_b(x, u) should equal inner_product(compute_s(u, 1), [1, x, x², ...])
    for k in [1, 2, 4, 6] {
      let challenges: Vec<Scalar> = (0..k).map(|_| Scalar::random(&mut OsRng)).collect();
      let x = Scalar::random(&mut OsRng);

      let b = compute_b(&x, &challenges);
      let s = compute_s(&challenges, &Scalar::ONE);
      let n = 1 << k;
      let mut powers = Vec::with_capacity(n);
      let mut cur = Scalar::ONE;
      for _ in 0..n {
        powers.push(cur);
        cur *= x;
      }
      let b_via_ip = inner_product(&s, &powers);

      assert_eq!(b, b_via_ip, "compute_b and compute_s disagree for k={k}");
    }
  }

  #[test]
  fn test_compute_s_structure() {
    let k = 3;
    let challenges: Vec<Scalar> = (0..k).map(|_| Scalar::random(&mut OsRng)).collect();
    let s = compute_s(&challenges, &Scalar::ONE);

    assert_eq!(s.len(), 1 << k);
    // s[0] should be 1 (init)
    assert_eq!(s[0], Scalar::ONE);
  }

  #[test]
  fn test_params_setup() {
    for k in [1, 4, 8] {
      let params = IpaParams::<E>::new(k);
      assert_eq!(params.g.len(), 1 << k);
      assert_eq!(params.n, 1 << k);
      assert_eq!(params.k, k);
    }
  }

  #[test]
  fn test_commitment_deterministic() {
    let params = IpaParams::<E>::new(4);
    let poly = random_poly(params.n);
    let blind = Scalar::random(&mut OsRng);

    let c1 = params.commit(&poly, &blind).unwrap();
    let c2 = params.commit(&poly, &blind).unwrap();
    assert_eq!(c1, c2);
  }

  #[test]
  fn test_prove_verify_roundtrip() {
    for k in [1, 4, 8] {
      let params = IpaParams::<E>::new(k);
      let poly = random_poly(params.n);
      let blind = Scalar::random(&mut OsRng);
      let commitment = params.commit(&poly, &blind).unwrap();

      let x = Scalar::random(&mut OsRng);
      let v = eval_poly(&poly, &x);

      // Prove
      let mut prover_transcript =
        <E as Engine>::TE::new(b"test_ipa");
      let proof = IpaProof::<E>::create(
        &params,
        &mut prover_transcript,
        &poly,
        &blind,
        &x,
        &v,
        &commitment,
      )
      .expect("proof creation should succeed");

      // Verify
      let mut verifier_transcript =
        <E as Engine>::TE::new(b"test_ipa");
      let guard = proof
        .verify(&params, &mut verifier_transcript, &commitment, &x, &v)
        .expect("verification should succeed");

      // Full check via use_challenges
      let msm = guard.use_challenges().expect("use_challenges should succeed");
      assert!(
        msm.is_identity().expect("MSM eval should succeed"),
        "MSM should be identity for k={k}"
      );
    }
  }

  #[test]
  fn test_wrong_eval_rejects() {
    let k = 4;
    let params = IpaParams::<E>::new(k);
    let poly = random_poly(params.n);
    let blind = Scalar::random(&mut OsRng);
    let commitment = params.commit(&poly, &blind).unwrap();

    let x = Scalar::random(&mut OsRng);
    let v = eval_poly(&poly, &x);
    let v_wrong = v + Scalar::ONE;

    // Prove with correct value
    let mut pt = <E as Engine>::TE::new(b"test_ipa");
    let proof =
      IpaProof::<E>::create(&params, &mut pt, &poly, &blind, &x, &v, &commitment).unwrap();

    // Verify with wrong value — transcript diverges, verification should fail
    let mut vt = <E as Engine>::TE::new(b"test_ipa");
    let result = proof.verify(&params, &mut vt, &commitment, &x, &v_wrong);

    // The transcript will diverge (different v absorbed), so the MSM won't be identity.
    // But verify() itself returns Ok(Guard) — the check happens at MSM eval.
    if let Ok(guard) = result {
      let msm = guard.use_challenges().unwrap();
      assert!(
        !msm.is_identity().unwrap(),
        "MSM should NOT be identity for wrong evaluation"
      );
    }
    // If verify itself errors, that's also acceptable
  }

  #[test]
  fn test_use_g_with_compute_g() {
    let k = 4;
    let params = IpaParams::<E>::new(k);
    let poly = random_poly(params.n);
    let blind = Scalar::random(&mut OsRng);
    let commitment = params.commit(&poly, &blind).unwrap();

    let x = Scalar::random(&mut OsRng);
    let v = eval_poly(&poly, &x);

    let mut pt = <E as Engine>::TE::new(b"test_ipa");
    let proof =
      IpaProof::<E>::create(&params, &mut pt, &poly, &blind, &x, &v, &commitment).unwrap();

    let mut vt = <E as Engine>::TE::new(b"test_ipa");
    let guard = proof
      .verify(&params, &mut vt, &commitment, &x, &v)
      .unwrap();

    // Compute the correct G
    let g = guard.compute_g().unwrap();

    // Use it
    let (msm, acc) = guard.use_g(&g);
    assert!(
      msm.is_identity().unwrap(),
      "MSM should be identity with correct G"
    );

    // Accumulator should also verify
    assert!(acc.decide(&params).unwrap(), "Accumulator decide should pass");
  }

  #[test]
  fn test_use_g_wrong_g_rejects() {
    let k = 4;
    let params = IpaParams::<E>::new(k);
    let poly = random_poly(params.n);
    let blind = Scalar::random(&mut OsRng);
    let commitment = params.commit(&poly, &blind).unwrap();

    let x = Scalar::random(&mut OsRng);
    let v = eval_poly(&poly, &x);

    let mut pt = <E as Engine>::TE::new(b"test_ipa");
    let proof =
      IpaProof::<E>::create(&params, &mut pt, &poly, &blind, &x, &v, &commitment).unwrap();

    let mut vt = <E as Engine>::TE::new(b"test_ipa");
    let guard = proof
      .verify(&params, &mut vt, &commitment, &x, &v)
      .unwrap();

    // Use a random (wrong) G
    let wrong_g = <E as Engine>::GE::generator();
    let (msm, acc) = guard.use_g(&wrong_g);

    assert!(
      !msm.is_identity().unwrap(),
      "MSM should NOT be identity with wrong G"
    );
    assert!(
      !acc.decide(&params).unwrap(),
      "Accumulator decide should fail with wrong G"
    );
  }

  #[test]
  fn test_accumulator_decide() {
    let k = 4;
    let params = IpaParams::<E>::new(k);
    let poly = random_poly(params.n);
    let blind = Scalar::random(&mut OsRng);
    let commitment = params.commit(&poly, &blind).unwrap();

    let x = Scalar::random(&mut OsRng);
    let v = eval_poly(&poly, &x);

    let mut pt = <E as Engine>::TE::new(b"test_ipa");
    let proof =
      IpaProof::<E>::create(&params, &mut pt, &poly, &blind, &x, &v, &commitment).unwrap();

    let mut vt = <E as Engine>::TE::new(b"test_ipa");
    let guard = proof
      .verify(&params, &mut vt, &commitment, &x, &v)
      .unwrap();

    let g = guard.compute_g().unwrap();
    let (_msm, acc) = guard.use_g(&g);

    assert!(acc.decide(&params).unwrap());
  }

  #[test]
  fn test_batch_verify_multiple() {
    let k = 4;
    let params = IpaParams::<E>::new(k);

    let mut msms = Vec::new();
    for _ in 0..3 {
      let poly = random_poly(params.n);
      let blind = Scalar::random(&mut OsRng);
      let commitment = params.commit(&poly, &blind).unwrap();
      let x = Scalar::random(&mut OsRng);
      let v = eval_poly(&poly, &x);

      let mut pt = <E as Engine>::TE::new(b"test_ipa");
      let proof =
        IpaProof::<E>::create(&params, &mut pt, &poly, &blind, &x, &v, &commitment).unwrap();

      let mut vt = <E as Engine>::TE::new(b"test_ipa");
      let guard = proof
        .verify(&params, &mut vt, &commitment, &x, &v)
        .unwrap();
      let msm = guard.use_challenges().unwrap();
      msms.push(msm);
    }

    assert!(
      DeferredMSM::<E>::batch_verify(msms).unwrap(),
      "Batch verification of 3 valid proofs should pass"
    );
  }

  /// End-to-end accumulation test:
  /// 1. Create N proofs for different polynomials
  /// 2. Verify each proof individually, using use_g to defer G computation
  /// 3. Batch-verify all the MSMs together
  /// 4. Verify each accumulator via decide()
  /// All steps must pass.
  #[test]
  fn test_full_accumulation_flow() {
    let num_proofs = 5;
    let k = 4;
    let params = IpaParams::<E>::new(k);

    let mut msms = Vec::new();
    let mut accumulators = Vec::new();

    for i in 0..num_proofs {
      // Each proof opens a different polynomial at a different point
      let poly = random_poly(params.n);
      let blind = Scalar::random(&mut OsRng);
      let commitment = params.commit(&poly, &blind).unwrap();
      let x = Scalar::random(&mut OsRng);
      let v = eval_poly(&poly, &x);

      // Prove
      let mut pt = <E as Engine>::TE::new(b"test_ipa");
      let proof =
        IpaProof::<E>::create(&params, &mut pt, &poly, &blind, &x, &v, &commitment).unwrap();

      // Verify — get Guard
      let mut vt = <E as Engine>::TE::new(b"test_ipa");
      let guard = proof
        .verify(&params, &mut vt, &commitment, &x, &v)
        .expect(&format!("verification of proof {i} should succeed"));

      // Compute the correct G (this is what the prover would provide)
      let g = guard.compute_g().expect("compute_g should succeed");

      // Use deferred path: accept G, get MSM + Accumulator
      let (msm, acc) = guard.use_g(&g);
      msms.push(msm);
      accumulators.push(acc);
    }

    // Step 1: Batch-verify all MSMs (cheap check — ensures each proof's
    // verification equation holds, assuming the claimed G points are correct)
    assert!(
      DeferredMSM::<E>::batch_verify(msms).unwrap(),
      "Batch MSM verification of {num_proofs} proofs should pass"
    );

    // Step 2: Verify each accumulator (the deferred O(n) decider)
    for (i, acc) in accumulators.iter().enumerate() {
      assert!(
        acc.decide(&params).unwrap(),
        "Accumulator {i} decide should pass"
      );
    }
  }

  /// Same as above but one proof has a wrong G — batch MSM should fail.
  #[test]
  fn test_full_accumulation_flow_one_bad() {
    let num_proofs = 4;
    let k = 4;
    let params = IpaParams::<E>::new(k);

    let mut msms = Vec::new();
    let mut accumulators = Vec::new();

    for i in 0..num_proofs {
      let poly = random_poly(params.n);
      let blind = Scalar::random(&mut OsRng);
      let commitment = params.commit(&poly, &blind).unwrap();
      let x = Scalar::random(&mut OsRng);
      let v = eval_poly(&poly, &x);

      let mut pt = <E as Engine>::TE::new(b"test_ipa");
      let proof =
        IpaProof::<E>::create(&params, &mut pt, &poly, &blind, &x, &v, &commitment).unwrap();

      let mut vt = <E as Engine>::TE::new(b"test_ipa");
      let guard = proof
        .verify(&params, &mut vt, &commitment, &x, &v)
        .expect(&format!("verification of proof {i} should succeed"));

      let g = guard.compute_g().expect("compute_g should succeed");

      // Inject a wrong G for the last proof
      let g_to_use = if i == num_proofs - 1 {
        <E as Engine>::GE::generator() // wrong!
      } else {
        g
      };

      let (msm, acc) = guard.use_g(&g_to_use);
      msms.push(msm);
      accumulators.push(acc);
    }

    // Batch MSM check should fail because one G is wrong
    assert!(
      !DeferredMSM::<E>::batch_verify(msms).unwrap(),
      "Batch MSM should fail with one bad G"
    );

    // The bad accumulator's decide should also fail
    assert!(
      !accumulators.last().unwrap().decide(&params).unwrap(),
      "Last accumulator decide should fail"
    );

    // But the good accumulators' decide should pass
    for (i, acc) in accumulators.iter().take(num_proofs - 1).enumerate() {
      assert!(
        acc.decide(&params).unwrap(),
        "Accumulator {i} decide should pass (it was correct)"
      );
    }
  }

  // ─── Accumulator fold tests ──────────────────────────────────────────

  /// Helper: create an IPA proof and return the accumulator with correct G.
  /// Generic over engine.
  fn make_accumulator<EE: Engine>(
    params: &IpaParams<EE>,
  ) -> (Accumulator<EE>, DeferredMSM<EE>)
  where
    EE::GE: DlogGroupExt,
  {
    let poly: Vec<EE::Scalar> = (0..params.n)
      .map(|_| EE::Scalar::random(&mut OsRng))
      .collect();
    let blind = EE::Scalar::random(&mut OsRng);
    let commitment = params.commit(&poly, &blind).unwrap();
    let x = EE::Scalar::random(&mut OsRng);
    let v = eval_polynomial(&poly, &x);

    let mut pt = EE::TE::new(b"test_ipa");
    let proof =
      IpaProof::<EE>::create(params, &mut pt, &poly, &blind, &x, &v, &commitment).unwrap();

    let mut vt = EE::TE::new(b"test_ipa");
    let guard = proof
      .verify(params, &mut vt, &commitment, &x, &v)
      .unwrap();

    let g = guard.compute_g().unwrap();
    let (msm, acc) = guard.use_g(&g);
    (acc, msm)
  }

  fn test_accumulator_fold_two_with<EE: Engine>()
  where
    EE::GE: DlogGroupExt,
  {
    let k = 4;
    let params = IpaParams::<EE>::new(k);

    let (acc1, msm1) = make_accumulator::<EE>(&params);
    let (acc2, msm2) = make_accumulator::<EE>(&params);

    // Individual checks pass
    assert!(msm1.is_identity().unwrap());
    assert!(msm2.is_identity().unwrap());
    assert!(acc1.decide(&params).unwrap());
    assert!(acc2.decide(&params).unwrap());

    // Fold
    let r = EE::Scalar::random(&mut OsRng);
    let folded = Accumulator::<EE>::fold(&acc1, &acc2, &r);

    // Folded decider passes
    assert!(
      folded.decide(&params).unwrap(),
      "Folded accumulator decide should pass"
    );
  }

  #[test]
  fn test_accumulator_fold_two() {
    test_accumulator_fold_two_with::<PallasHyraxEngine>();
  }

  fn test_accumulator_fold_seven_with<EE: Engine>()
  where
    EE::GE: DlogGroupExt,
  {
    let k = 4;
    let params = IpaParams::<EE>::new(k);

    let mut accumulators = Vec::new();
    for _ in 0..7 {
      let (acc, msm) = make_accumulator::<EE>(&params);
      assert!(msm.is_identity().unwrap());
      accumulators.push(acc);
    }

    let mut transcript = EE::TE::new(b"test_fold_all");
    let folded = Accumulator::<EE>::fold_all(&accumulators, &mut transcript).unwrap();

    assert!(
      folded.decide(&params).unwrap(),
      "fold_all of 7 accumulators should decide correctly"
    );
  }

  #[test]
  fn test_accumulator_fold_seven() {
    test_accumulator_fold_seven_with::<PallasHyraxEngine>();
  }

  fn test_accumulator_fold_bad_g_with<EE: Engine>()
  where
    EE::GE: DlogGroupExt,
  {
    let k = 4;
    let params = IpaParams::<EE>::new(k);

    let (acc1, _) = make_accumulator::<EE>(&params);

    // Create acc2 with a corrupted G
    let (mut acc2, _) = make_accumulator::<EE>(&params);
    acc2.g = acc2.g + EE::GE::generator(); // corrupt

    let r = EE::Scalar::random(&mut OsRng);
    let folded = Accumulator::<EE>::fold(&acc1, &acc2, &r);

    assert!(
      !folded.decide(&params).unwrap(),
      "Folded accumulator with bad G should fail decide"
    );
  }

  #[test]
  fn test_accumulator_fold_bad_g() {
    test_accumulator_fold_bad_g_with::<PallasHyraxEngine>();
  }
}
