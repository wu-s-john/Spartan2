// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
//! Non-ZK, fixed-batch NeutronNova/Spartan over T256's scalar field.
//!
//! Batch variables bind least-significant bit first. Spartan row/column
//! variables bind most-significant bit first, matching Hyrax's MLE convention.
//! Uses the folding and sumcheck kernels shared with `neutronnova_zk`.
//! This module supplies public round messages and native verification, with no ZK verifier circuit.

use crate::{
  CommitmentKey, VerifierKey,
  bellpepper::{r1cs::add_constraint, shape_cs::ShapeCS, solver::SatisfyingAssignment},
  errors::SpartanError,
  neutronnova_zk::{NeutronNovaNIFS, small_matvec_cache},
  polys::{
    eq::EqPolynomial,
    multilinear::{MultilinearPolynomial, SparsePolynomial},
    power::PowPolynomial,
  },
  provider::T256HyraxEngine,
  r1cs::{R1CSInstance, R1CSWitness, SparseMatrix, SplitR1CSShape},
  sumcheck::SumcheckProof,
  traits::{
    Engine,
    pcs::{FoldingEngineTrait, PCSEngineTrait},
    transcript::{TranscriptEngineTrait, TranscriptReprTrait},
  },
};
use bellpepper_core::{Circuit, ConstraintSystem};
use bincode::Options;
use ff::Field;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha3::{Digest, Keccak256};
use std::{borrow::Cow, time::Instant};

/// T256/Hyrax engine used by this non-ZK implementation.
pub type E = T256HyraxEngine;
/// Constraint and sumcheck field: the P-256 coordinate field.
pub type Scalar = <E as Engine>::Scalar;
type F = Scalar;
type Pcs = <E as Engine>::PCS;
type Transcript = <E as Engine>::TE;
type Products = (Vec<F>, Vec<F>, Vec<F>);
/// Protocol result with structured Spartan errors.
pub type Result<T> = std::result::Result<T, SpartanError>;

pub(crate) fn error(reason: impl Into<String>) -> SpartanError {
  SpartanError::ProofVerifyError {
    reason: reason.into(),
  }
}

struct Bytes<'a>(&'a [u8]);
impl TranscriptReprTrait<<E as Engine>::GE> for Bytes<'_> {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    let mut out = (self.0.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(self.0);
    out
  }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
/// Original and padded circuit dimensions.
pub struct CircuitSize {
  /// Original number of constraints.
  pub constraints: usize,
  /// Original number of witness field elements.
  pub variables: usize,
  /// Number of public field elements, excluding the unit coordinate.
  pub public_values: usize,
  /// Shared padded outer-sumcheck domain size.
  pub padded_constraints: usize,
  /// Shared padded committed-witness domain size.
  pub padded_variables: usize,
}

/// A single setup for K equal-shaped steps and one different core circuit.
pub struct ProverKey {
  shapes: [SplitR1CSShape<E>; 2],
  ck: CommitmentKey<E>,
  digest: [u8; 32],
  steps: usize,
  /// Step and core dimensions, in that order.
  pub sizes: [CircuitSize; 2],
}

/// Public shapes and PCS parameters needed for native verification.
pub struct VerificationKey {
  shapes: [SplitR1CSShape<E>; 2],
  pcs: VerifierKey<E>,
  digest: [u8; 32],
  steps: usize,
  width: usize,
}

struct Assignment {
  values: Vec<F>,
  public: Vec<F>,
  small: bool,
}

/// Synthesized witnesses, before any commitments or matrix products.
pub struct Witness {
  key_digest: [u8; 32],
  steps: Vec<Assignment>,
  core: Assignment,
}

/// Committed instances and the retained prover witnesses/blinds.
pub struct Committed {
  key_digest: [u8; 32],
  steps: Vec<R1CSInstance<E>>,
  core: R1CSInstance<E>,
  witnesses: Vec<R1CSWitness<E>>,
  core_witness: R1CSWitness<E>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
/// Public sumcheck messages and one direct opening, including all instances.
pub struct Proof {
  pub(crate) steps: Vec<R1CSInstance<E>>,
  pub(crate) core: R1CSInstance<E>,
  folding: Vec<[F; 4]>,
  folded_target: F,
  outer: Vec<[[F; 4]; 2]>,
  outer_claims: [[F; 3]; 2],
  inner: Vec<[[F; 3]; 2]>,
  evaluations: [F; 2],
  opening: Vec<F>,
  opening_blind: F,
}

#[derive(Default, Clone, Debug, Serialize)]
/// Non-overlapping prover phase times in milliseconds.
pub struct Phases {
  /// Construction of the step/core matrix-vector products.
  pub matrix_ms: f64,
  /// NeutronNova sumcheck and witness/instance folding.
  pub folding_ms: f64,
  /// Paired outer reduction and terminal-claim batching.
  pub outer_ms: f64,
  /// Matrix binding, paired inner reduction, and evaluation batching.
  pub inner_ms: f64,
  /// Combined witness construction and direct Hyrax opening.
  pub opening_ms: f64,
}

impl Proof {
  /// Encode the entire proof, including commitments and public inputs.
  pub fn to_bytes(&self) -> Result<Vec<u8>> {
    bincode::DefaultOptions::new()
      .with_fixint_encoding()
      .serialize(self)
      .map_err(|e| error(e.to_string()))
  }
  /// Decode a length-limited proof, rejecting trailing bytes.
  pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
    bincode::DefaultOptions::new()
      .with_fixint_encoding()
      .with_limit(bytes.len() as u64)
      .reject_trailing_bytes()
      .deserialize(bytes)
      .map_err(|e| error(e.to_string()))
  }
  /// Authenticated public values for each ordered step after verification.
  pub fn step_public_values(&self) -> impl Iterator<Item = &[F]> {
    self.steps.iter().map(|u| u.X.as_slice())
  }
  /// Authenticated core public values after verification.
  pub fn core_public_values(&self) -> &[F] {
    &self.core.X
  }
}

fn shape<C: Circuit<F>>(circuit: C) -> Result<SplitR1CSShape<E>> {
  let mut cs = ShapeCS::<E>::new();
  circuit
    .synthesize(&mut cs)
    .map_err(|e| error(e.to_string()))?;
  let (mut a, mut b, mut c) = (
    SparseMatrix::empty(),
    SparseMatrix::empty(),
    SparseMatrix::empty(),
  );
  let mut rows = 0;
  for (la, lb, lc) in &cs.constraints {
    add_constraint(
      &mut (&mut a, &mut b, &mut c, &mut rows),
      cs.num_aux(),
      la,
      lb,
      lc,
    );
  }
  for matrix in [&mut a, &mut b, &mut c] {
    matrix.cols = cs.num_aux() + cs.num_inputs();
  }
  SplitR1CSShape::new(rows, 0, 0, cs.num_aux(), cs.num_inputs() - 1, 0, a, b, c)
}

/// Synthesize public circuit shapes and prepare one compatible Hyrax key.
pub fn setup<C1: Circuit<F>, C2: Circuit<F>>(
  step: C1,
  core: C2,
  steps: usize,
  width: usize,
) -> Result<(ProverKey, VerificationKey)> {
  if !steps.is_power_of_two() || !width.is_power_of_two() {
    return Err(error(
      "step count and commitment width must be positive powers of two",
    ));
  }
  let (mut step, mut core) = (shape(step)?, shape(core)?);
  SplitR1CSShape::equalize(&mut step, &mut core);
  let shapes = [step, core];
  let variables = shapes[0].num_rest;
  if width > variables {
    return Err(error("commitment width exceeds padded witness"));
  }
  let (ck, pcs) = Pcs::setup(b"non-zk-mc/ck/v1", variables, width);
  Pcs::precompute_ck(&ck);
  let serialized =
    bincode::serialize(&(&shapes, &pcs, steps, width)).map_err(|e| error(e.to_string()))?;
  let digest: [u8; 32] = Keccak256::digest(serialized).into();
  for s in &shapes {
    s.precompute();
  }
  let sizes = std::array::from_fn(|i| CircuitSize {
    constraints: shapes[i].num_cons_unpadded,
    variables: shapes[i].num_rest_unpadded,
    public_values: shapes[i].num_public,
    padded_constraints: shapes[i].num_cons,
    padded_variables: shapes[i].num_rest,
  });
  let vk = VerificationKey {
    shapes: shapes.clone(),
    pcs,
    digest,
    steps,
    width,
  };
  Ok((
    ProverKey {
      shapes,
      ck,
      digest,
      steps,
      sizes,
    },
    vk,
  ))
}

fn assignment<C: Circuit<F>>(c: C, s: &SplitR1CSShape<E>) -> Result<Assignment> {
  let mut cs = SatisfyingAssignment::<E>::new();
  c.synthesize(&mut cs).map_err(|e| error(e.to_string()))?;
  if cs.aux_assignment.len() != s.num_rest_unpadded || cs.input_assignment.len() != s.num_public + 1
  {
    return Err(error("circuit witness shape changed after setup"));
  }
  let small = cs
    .aux_assignment
    .iter()
    .all(|v| *v == F::ZERO || *v == F::ONE);
  cs.aux_assignment.resize(s.num_rest, F::ZERO);
  Ok(Assignment {
    values: cs.aux_assignment,
    public: cs.input_assignment[1..].to_vec(),
    small,
  })
}

/// Synthesize fresh witnesses; no commitments or matrix products are computed.
pub fn generate_witness<C1: Circuit<F> + Send, C2: Circuit<F>>(
  pk: &ProverKey,
  steps: Vec<C1>,
  core: C2,
) -> Result<Witness> {
  if steps.len() != pk.steps {
    return Err(error("wrong step count"));
  }
  let steps = steps
    .into_par_iter()
    .map(|c| assignment(c, &pk.shapes[0]))
    .collect::<Result<Vec<_>>>()?;
  Ok(Witness {
    key_digest: pk.digest,
    steps,
    core: assignment(core, &pk.shapes[1])?,
  })
}

/// Consume witnesses and commit each instance using fresh blinds.
pub fn commit(pk: &ProverKey, witness: Witness) -> Result<Committed> {
  if witness.key_digest != pk.digest {
    return Err(error("witness belongs to a different setup"));
  }
  let commit_one = |a: Assignment| -> Result<_> {
    let small = a.small;
    let blind = Pcs::blind(&pk.ck, a.values.len());
    let comm = Pcs::commit(&pk.ck, &a.values, &blind, small)?;
    Ok((
      R1CSInstance {
        comm_W: comm,
        X: a.public,
      },
      R1CSWitness {
        W: a.values,
        r_W: blind,
        is_small: small,
      },
    ))
  };
  let pairs = witness
    .steps
    .into_par_iter()
    .map(commit_one)
    .collect::<Result<Vec<_>>>()?;
  let (steps, witnesses) = pairs.into_iter().unzip();
  let (core, core_witness) = commit_one(witness.core)?;
  Ok(Committed {
    key_digest: pk.digest,
    steps,
    core,
    witnesses,
    core_witness,
  })
}

fn transcript(
  digest: &[u8; 32],
  binding: &[u8],
  steps: &[R1CSInstance<E>],
  core: &R1CSInstance<E>,
) -> Transcript {
  let mut t = Transcript::new(b"non-zk-mc/v2");
  t.absorb(b"vk", &Bytes(digest));
  t.absorb(b"statement", &Bytes(binding));
  for u in steps {
    t.absorb(b"step", u);
  }
  t.absorb(b"core", core);
  t
}

fn products(s: &SplitR1CSShape<E>, u: &R1CSInstance<E>, w: &R1CSWitness<E>) -> Result<Products> {
  let mut z = w.W.clone();
  z.push(F::ONE);
  z.extend_from_slice(&u.X);
  s.multiply_vec(&z)
}

fn eval<const N: usize>(p: &[F; N], x: F) -> F {
  p.iter().rev().fold(F::ZERO, |v, c| v * x + c)
}

fn round<const N: usize>(t: &mut Transcript, p: &[F; N], claim: F) -> Result<()> {
  if p[0] + eval(p, F::ONE) != claim {
    return Err(error("sumcheck round does not match its incoming claim"));
  }
  t.absorb(b"polynomial", &p.as_slice());
  Ok(())
}

/// Prove the committed batch with public folding/Spartan messages and one opening.
pub fn prove(pk: &ProverKey, binding: &[u8], committed: &Committed) -> Result<(Proof, Phases)> {
  if committed.key_digest != pk.digest {
    return Err(error("commitment belongs to a different setup"));
  }
  let mut phases = Phases::default();
  let mut t = transcript(&pk.digest, binding, &committed.steps, &committed.core);
  let tau = t.squeeze(b"tau")?;
  let rounds_b = pk.steps.ilog2() as usize;
  let rhos = (0..rounds_b)
    .map(|_| t.squeeze(b"rho"))
    .collect::<Result<Vec<_>>>()?;
  let m = pk.shapes[0].num_cons;
  let n = pk.shapes[0].num_rest;
  let rounds_x = m.ilog2() as usize;
  let left = 1 << rounds_x.div_ceil(2);
  let right = 1 << (rounds_x / 2);
  let started = Instant::now();
  let (layers, core) = rayon::join(
    || {
      committed
        .steps
        .par_iter()
        .zip(&committed.witnesses)
        .map(|(u, w)| products(&pk.shapes[0], u, w))
        .collect::<Result<Vec<_>>>()
    },
    || products(&pk.shapes[1], &committed.core, &committed.core_witness),
  );
  let mut layers = layers?;
  let (az_core, bz_core, cz_core) = core?;
  // Build the same small-value caches used by the ZK prover, inside the measured
  // protocol phase: these matrix products depend on this proof's fresh witness.
  let cache = (rounds_b > 0).then(|| small_matvec_cache(&layers));
  phases.matrix_ms = started.elapsed().as_secs_f64() * 1000.;

  let started = Instant::now();
  let mut folding = Vec::new();
  let (mut claim, mut prefix) = (F::ZERO, F::ONE);
  let (mut eq, az_step, bz_step, cz_step, folded_w, folded_u, folded_target) =
    if let Some((small, large_positions)) = cache {
      let folded = NeutronNovaNIFS::<E>::prove_with_rounds(
        &pk.shapes[0],
        &pk.ck,
        committed.steps.par_iter().cloned().collect(),
        committed.witnesses.par_iter().cloned().collect(),
        Some(layers),
        Some(small),
        &large_positions,
        tau,
        &rhos,
        |j, p| {
          round(&mut t, &p, claim)?;
          let r = t.squeeze(b"fold")?;
          claim = eval(&p, r);
          prefix *= (F::ONE - rhos[j]) * (F::ONE - r) + rhos[j] * r;
          folding.push(p);
          Ok(r)
        },
      )?;
      if prefix != folded.eq_rho || claim != prefix * folded.target {
        return Err(error("invalid folding residual"));
      }
      (
        folded.eq,
        folded.az,
        folded.bz,
        folded.cz,
        Cow::Owned(folded.witness),
        Cow::Owned(folded.instance),
        folded.target,
      )
    } else {
      let (a, b, c) = layers.pop().ok_or_else(|| error("empty step batch"))?;
      (
        PowPolynomial::split_evals(tau, rounds_x, left, right),
        a,
        b,
        c,
        Cow::Borrowed(&committed.witnesses[0]),
        Cow::Borrowed(&committed.steps[0]),
        F::ZERO,
      )
    };
  t.absorb(b"folded-target", &folded_target);
  phases.folding_ms = if rounds_b == 0 {
    0.
  } else {
    started.elapsed().as_secs_f64() * 1000.
  };

  let started = Instant::now();
  let eq_right = eq.split_off(left);
  let mut pow_left = MultilinearPolynomial::new(eq);
  let pow_right = MultilinearPolynomial::new(eq_right);
  let mut az_step = MultilinearPolynomial::new(az_step);
  let mut bz_step = MultilinearPolynomial::new(bz_step);
  let mut cz_step = MultilinearPolynomial::new(cz_step);
  let mut az_core = MultilinearPolynomial::new(az_core);
  let mut bz_core = MultilinearPolynomial::new(bz_core);
  let mut cz_core = MultilinearPolynomial::new(cz_core);
  let mut claims = [folded_target, F::ZERO];
  let mut outer = Vec::new();
  let rx = SumcheckProof::<E>::prove_cubic_with_additive_term_batched_with_rounds(
    &[folded_target, F::ZERO],
    rounds_x,
    &mut pow_left,
    &pow_right,
    &mut az_step,
    &mut az_core,
    &mut bz_step,
    &mut bz_core,
    &mut cz_step,
    &mut cz_core,
    |_, p| {
      for j in 0..2 {
        round(&mut t, &p[j], claims[j])?;
      }
      let r = t.squeeze(b"outer")?;
      claims = p.map(|q| eval(&q, r));
      outer.push(p);
      Ok(r)
    },
  )?;
  let outer_claims = [
    [az_step[0], bz_step[0], cz_step[0]],
    [az_core[0], bz_core[0], cz_core[0]],
  ];
  let weight = PowPolynomial::new(&tau, rx.len()).evaluate(&rx)?;
  for (j, q) in outer_claims.iter().enumerate() {
    if claims[j] != weight * (q[0] * q[1] - q[2]) {
      return Err(error("invalid outer terminal"));
    }
    t.absorb(b"outer-claims", &q.as_slice());
  }
  let alpha = t.squeeze(b"matrix-batch")?;
  phases.outer_ms = started.elapsed().as_secs_f64() * 1000.;

  let started = Instant::now();
  let eq_rx = EqPolynomial::evals_from_points(&rx);
  let [mut matrix_step, mut matrix_core] = std::array::from_fn::<_, 2, _>(|j| {
    MultilinearPolynomial::new(
      pk.shapes[j]
        .bind_and_prepare_poly_ABC_full(&eq_rx, &alpha)
        .0,
    )
  });
  let us = [&*folded_u, &committed.core];
  let ws = [&*folded_w, &committed.core_witness];
  let [mut z_step, mut z_core] = std::array::from_fn::<_, 2, _>(|j| {
    let mut v = ws[j].W.clone();
    v.push(F::ONE);
    v.extend_from_slice(&us[j].X);
    v.resize(2 * n, F::ZERO);
    MultilinearPolynomial::new(v)
  });
  claims = outer_claims.map(|q| q[0] + alpha * q[1] + alpha.square() * q[2]);
  let mut inner = Vec::new();
  let initial_inner_claims = claims;
  let (ry, terminals) = SumcheckProof::<E>::prove_quad_batched_with_rounds(
    &initial_inner_claims,
    n.ilog2() as usize + 1,
    &mut matrix_step,
    &mut matrix_core,
    &mut z_step,
    &mut z_core,
    |_, p| {
      for j in 0..2 {
        round(&mut t, &p[j], claims[j])?;
      }
      let r = t.squeeze(b"inner")?;
      claims = p.map(|q| eval(&q, r));
      inner.push(p);
      Ok(r)
    },
  )?;
  for j in 0..2 {
    if claims[j] != terminals[j] * terminals[j + 2] {
      return Err(error("invalid inner terminal"));
    }
  }
  let eq_w = EqPolynomial::evals_from_points(&ry[1..]);
  let evaluations = ws.map(|w| w.W.par_iter().zip(&eq_w).map(|(a, b)| *a * b).sum());
  t.absorb(b"witness-evaluations", &evaluations.as_slice());
  let eta = t.squeeze(b"opening-batch")?;
  phases.inner_ms = started.elapsed().as_secs_f64() * 1000.;

  let started = Instant::now();
  let witness = folded_w
    .W
    .par_iter()
    .zip(&committed.core_witness.W)
    .map(|(a, b)| *a + eta * b)
    .collect::<Vec<_>>();
  let blind = <Pcs as FoldingEngineTrait<E>>::fold_blinds(
    &[folded_w.r_W.clone(), committed.core_witness.r_W.clone()],
    &[F::ONE, eta],
  )?;
  let (opening, opening_blind) = Pcs::prove_direct(&pk.ck, &witness, &blind, &ry[1..])?;
  phases.opening_ms = started.elapsed().as_secs_f64() * 1000.;
  Ok((
    Proof {
      steps: committed.steps.clone(),
      core: committed.core.clone(),
      folding,
      folded_target,
      outer,
      outer_claims,
      inner,
      evaluations,
      opening,
      opening_blind,
    },
    phases,
  ))
}

/// Verify dimensions, folding, both Spartan branches, and the final opening.
pub fn verify(vk: &VerificationKey, binding: &[u8], proof: &Proof) -> Result<()> {
  let m = vk.shapes[0].num_cons;
  let n = vk.shapes[0].num_rest;
  let rounds_b = vk.steps.ilog2() as usize;
  if proof.steps.len() != vk.steps
    || proof.folding.len() != rounds_b
    || proof.outer.len() != m.ilog2() as usize
    || proof.inner.len() != n.ilog2() as usize + 1
    || proof.opening.len() != vk.width
  {
    return Err(error("invalid proof dimensions"));
  }
  for (u, s) in proof
    .steps
    .iter()
    .map(|u| (u, &vk.shapes[0]))
    .chain(std::iter::once((&proof.core, &vk.shapes[1])))
  {
    if u.X.len() != s.num_public {
      return Err(error("invalid public input dimensions"));
    }
    Pcs::check_commitment(&u.comm_W, n, vk.width)?;
  }
  let mut t = transcript(&vk.digest, binding, &proof.steps, &proof.core);
  let tau = t.squeeze(b"tau")?;
  let rhos = (0..rounds_b)
    .map(|_| t.squeeze(b"rho"))
    .collect::<Result<Vec<_>>>()?;
  let (mut claim, mut prefix) = (F::ZERO, F::ONE);
  let mut rb = Vec::new();
  for (p, rho) in proof.folding.iter().zip(rhos) {
    round(&mut t, p, claim)?;
    let r = t.squeeze(b"fold")?;
    claim = eval(p, r);
    prefix *= (F::ONE - rho) * (F::ONE - r) + rho * r;
    rb.push(r);
  }
  if claim != prefix * proof.folded_target {
    return Err(error("invalid folding terminal"));
  }
  t.absorb(b"folded-target", &proof.folded_target);
  let folded = if rounds_b == 0 {
    Cow::Borrowed(&proof.steps[0])
  } else {
    Cow::Owned(R1CSInstance::fold_multiple(&rb, &proof.steps)?)
  };
  let mut claims = [proof.folded_target, F::ZERO];
  let mut rx = Vec::new();
  for p in &proof.outer {
    for j in 0..2 {
      round(&mut t, &p[j], claims[j])?;
    }
    let r = t.squeeze(b"outer")?;
    claims = p.map(|q| eval(&q, r));
    rx.push(r);
  }
  let weight = PowPolynomial::new(&tau, rx.len()).evaluate(&rx)?;
  for (q, claim) in proof.outer_claims.iter().zip(claims) {
    if claim != weight * (q[0] * q[1] - q[2]) {
      return Err(error("invalid outer terminal"));
    }
    t.absorb(b"outer-claims", &q.as_slice());
  }
  let alpha = t.squeeze(b"matrix-batch")?;
  claims = proof
    .outer_claims
    .map(|q| q[0] + alpha * q[1] + alpha.square() * q[2]);
  let mut ry = Vec::new();
  for p in &proof.inner {
    for j in 0..2 {
      round(&mut t, &p[j], claims[j])?;
    }
    let r = t.squeeze(b"inner")?;
    claims = p.map(|q| eval(&q, r));
    ry.push(r);
  }
  let eq_rx = EqPolynomial::evals_from_points(&rx);
  let eq_ry = EqPolynomial::evals_from_points(&ry);
  for (j, u) in [&*folded, &proof.core].iter().enumerate() {
    let (a, b, c) = vk.shapes[j].evaluate_with_tables_fast(&eq_rx, &eq_ry);
    let public = std::iter::once(F::ONE).chain(u.X.iter().copied()).collect();
    let public = SparsePolynomial::new(n.ilog2() as usize, public).evaluate(&ry[1..]);
    let z = (F::ONE - ry[0]) * proof.evaluations[j] + ry[0] * public;
    if claims[j] != (a + alpha * b + alpha.square() * c) * z {
      return Err(error("invalid inner terminal"));
    }
  }
  t.absorb(b"witness-evaluations", &proof.evaluations.as_slice());
  let eta = t.squeeze(b"opening-batch")?;
  let comm = <Pcs as FoldingEngineTrait<E>>::fold_commitments(
    &[folded.comm_W.clone(), proof.core.comm_W.clone()],
    &[F::ONE, eta],
  )?;
  let value = Pcs::verify_direct(
    &vk.pcs,
    &comm,
    &proof.opening,
    &proof.opening_blind,
    &ry[1..],
  )?;
  if value != proof.evaluations[0] + eta * proof.evaluations[1] {
    return Err(error("opening evaluation mismatch"));
  }
  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;
  use bellpepper_core::SynthesisError;

  #[derive(Clone)]
  struct Multiply {
    a: F,
    b: F,
    c: F,
    rows: usize,
  }
  impl Circuit<F> for Multiply {
    fn synthesize<CS: ConstraintSystem<F>>(
      self,
      cs: &mut CS,
    ) -> std::result::Result<(), SynthesisError> {
      let a = cs.alloc(|| "a", || Ok(self.a))?;
      let b = cs.alloc(|| "b", || Ok(self.b))?;
      let c = cs.alloc_input(|| "c", || Ok(self.c))?;
      for row in 0..self.rows {
        cs.enforce(
          || format!("a*b=c {row}"),
          |lc| lc + a,
          |lc| lc + b,
          |lc| lc + c,
        );
      }
      Ok(())
    }
  }
  fn circuit(i: u64) -> Multiply {
    Multiply {
      a: F::from(i),
      b: F::from(i + 1),
      c: F::from(i * (i + 1)),
      rows: 3,
    }
  }

  #[test]
  fn folds_batches_and_rejects_altered_messages() {
    for count in [1, 2, 4, 8] {
      let (pk, vk) = setup(circuit(1), circuit(2), count, 2048).unwrap();
      for seed in [3, 19] {
        let steps = (0..count).map(|i| circuit(seed + i as u64)).collect();
        let w = generate_witness(&pk, steps, circuit(seed + 100)).unwrap();
        let committed = commit(&pk, w).unwrap();
        let (proof, _) = prove(&pk, b"statement", &committed).unwrap();
        verify(&vk, b"statement", &proof).unwrap();
        assert_eq!(proof.folding.len(), count.ilog2() as usize);
        assert_eq!(proof.opening.len(), 2048);
        if count > 1 {
          assert_ne!(proof.folded_target, F::ZERO);
        }
        let encoded = proof.to_bytes().unwrap();
        let decoded = Proof::from_bytes(&encoded).unwrap();
        verify(&vk, b"statement", &decoded).unwrap();
        assert!(verify(&vk, b"different statement", &proof).is_err());
        let mut bad = proof.clone();
        bad.evaluations[0] += F::ONE;
        assert!(verify(&vk, b"statement", &bad).is_err());
        let mut bad = proof.clone();
        bad.opening[0] += F::ONE;
        assert!(verify(&vk, b"statement", &bad).is_err());
        let mut bad = proof.clone();
        bad.inner[0][0][0] += F::ONE;
        assert!(verify(&vk, b"statement", &bad).is_err());
        let mut bad = proof.clone();
        bad.outer[0][0][0] += F::ONE;
        assert!(verify(&vk, b"statement", &bad).is_err());
        let mut bad = proof.clone();
        bad.outer_claims[0][0] += F::ONE;
        assert!(verify(&vk, b"statement", &bad).is_err());
        if count > 1 {
          let mut bad = proof.clone();
          bad.folding[0][0] += F::ONE;
          assert!(verify(&vk, b"statement", &bad).is_err());
        }
        let mut bad = proof.clone();
        bad.folded_target += F::ONE;
        assert!(verify(&vk, b"statement", &bad).is_err());
        let mut bad = proof.clone();
        bad.steps[0].X[0] += F::ONE;
        assert!(verify(&vk, b"statement", &bad).is_err());
        assert!(Proof::from_bytes(&encoded[..encoded.len() - 1]).is_err());
        let mut trailing = encoded.clone();
        trailing.push(0);
        assert!(Proof::from_bytes(&trailing).is_err());
      }
    }
  }

  #[test]
  fn unsatisfied_step_and_core_cannot_prove() {
    for rows in [1, 3] {
      let circuit = |i| Multiply { rows, ..circuit(i) };
      let (pk, _) = setup(circuit(1), circuit(2), 2, 2048).unwrap();
      let mut invalid = circuit(3);
      invalid.c += F::ONE;
      for (steps, core) in [
        (vec![invalid.clone(), circuit(4)], circuit(5)),
        (vec![circuit(3), circuit(4)], invalid),
      ] {
        let w = generate_witness(&pk, steps, core).unwrap();
        let committed = commit(&pk, w).unwrap();
        assert!(prove(&pk, b"statement", &committed).is_err());
      }
    }
  }
}
