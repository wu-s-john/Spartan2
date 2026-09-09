//! Standard SHA-256 chains followed by native P-256 verification.

mod p256_circuit;
#[cfg(test)]
mod tests;

use crate::neutronnova::{self, Scalar, error};
use bellpepper::gadgets::{sha256::sha256_compression_function, uint32::UInt32};
use bellpepper_core::{Circuit, ConstraintSystem, SynthesisError};
use p256::{
  AffinePoint, ProjectivePoint, PublicKey, Scalar as SignatureScalar,
  elliptic_curve::{PrimeField, bigint::U256, ops::Reduce, sec1::ToEncodedPoint},
};
use serde::{Deserialize, Serialize};

pub use crate::neutronnova::{CircuitSize, Committed, Phases, Proof, Result, Witness};

const IV: [u32; 8] = [
  0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// External statement. Message bytes and internal SHA states are not expected
/// verifier inputs. Proof-carried auxiliary inputs need not be hidden.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Statement {
  /// Base-two logarithm of the total padded compression count.
  pub log_compressions: u8,
  /// Canonical big-endian public-key x-coordinate.
  pub qx: [u8; 32],
  /// Canonical big-endian public-key y-coordinate.
  pub qy: [u8; 32],
  /// Canonical big-endian ECDSA r scalar.
  pub r: [u8; 32],
  /// Canonical big-endian ECDSA s scalar; high-s is accepted.
  pub s: [u8; 32],
}

impl Statement {
  /// Canonical external statement encoding.
  pub fn to_bytes(&self) -> Vec<u8> {
    let mut out = vec![self.log_compressions];
    for v in [&self.qx, &self.qy, &self.r, &self.s] {
      out.extend_from_slice(v);
    }
    out
  }
}

#[derive(Clone)]
struct ShaChunk {
  input: Option<[u32; 8]>,
  blocks: Vec<Option<[u32; 16]>>,
  output: Option<[u32; 8]>,
}

impl Circuit<Scalar> for ShaChunk {
  fn synthesize<CS: ConstraintSystem<Scalar>>(
    self,
    cs: &mut CS,
  ) -> std::result::Result<(), SynthesisError> {
    let mut state = Vec::new();
    for i in 0..8 {
      let value = self.input.map(|h| h[i]);
      let word = UInt32::alloc(cs.namespace(|| format!("input {i}")), value)?;
      p256_circuit::publicize_word(cs.namespace(|| format!("public input {i}")), &word, value)?;
      state.push(word);
    }
    for (j, block) in self.blocks.iter().enumerate() {
      let mut bits = Vec::with_capacity(512);
      for i in 0..16 {
        let value = block.map(|b| b[i]);
        let word = UInt32::alloc(cs.namespace(|| format!("block {j} word {i}")), value)?;
        p256_circuit::publicize_word(
          cs.namespace(|| format!("public block {j} word {i}")),
          &word,
          value,
        )?;
        bits.extend(word.into_bits_be());
      }
      state =
        sha256_compression_function(cs.namespace(|| format!("compression {j}")), &bits, &state)?;
    }
    for (i, word) in state.iter().enumerate() {
      p256_circuit::publicize_word(
        cs.namespace(|| format!("public output {i}")),
        word,
        self.output.map(|h| h[i]),
      )?;
    }
    Ok(())
  }
}

#[derive(Clone)]
struct SignatureCircuit {
  statement: Option<Statement>,
  digest: Option<[u8; 32]>,
  r_point: Option<([u8; 32], [u8; 32])>,
}

impl Circuit<Scalar> for SignatureCircuit {
  fn synthesize<CS: ConstraintSystem<Scalar>>(
    self,
    cs: &mut CS,
  ) -> std::result::Result<(), SynthesisError> {
    let inputs = [
      self.digest,
      self.statement.as_ref().map(|s| s.qx),
      self.statement.as_ref().map(|s| s.qy),
      self.statement.as_ref().map(|s| s.r),
      self.statement.as_ref().map(|s| s.s),
    ];
    p256_circuit::constrain(cs, inputs, self.r_point)
  }
}

fn words<const N: usize>(bytes: &[u8]) -> [u32; N] {
  std::array::from_fn(|i| u32::from_be_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap()))
}

fn scalar_words(bytes: &[u8; 32]) -> Vec<Scalar> {
  words::<8>(bytes)
    .map(|w| Scalar::from(u64::from(w)))
    .to_vec()
}

fn padding(n: usize) -> [u8; 64] {
  let mut p = [0u8; 64];
  p[0] = 0x80;
  p[56..].copy_from_slice(&(((n - 1) * 512) as u64).to_be_bytes());
  p
}

fn signature_hint(statement: &Statement, digest: &[u8; 32]) -> Result<([u8; 32], [u8; 32])> {
  let mut key = [0u8; 65];
  key[0] = 4;
  key[1..33].copy_from_slice(&statement.qx);
  key[33..].copy_from_slice(&statement.qy);
  let q = PublicKey::from_sec1_bytes(&key).map_err(|e| error(e.to_string()))?;
  let r = Option::<SignatureScalar>::from(SignatureScalar::from_repr(statement.r.into()))
    .ok_or_else(|| error("noncanonical r"))?;
  let s = Option::<SignatureScalar>::from(SignatureScalar::from_repr(statement.s.into()))
    .ok_or_else(|| error("noncanonical s"))?;
  let inv =
    Option::<SignatureScalar>::from(s.invert()).ok_or_else(|| error("zero signature scalar"))?;
  if r == SignatureScalar::ZERO {
    return Err(error("zero signature scalar"));
  }
  let z = <SignatureScalar as Reduce<U256>>::reduce_bytes(&(*digest).into());
  let rp = AffinePoint::from(
    (ProjectivePoint::GENERATOR * z + ProjectivePoint::from(*q.as_affine()) * r) * inv,
  );
  if bool::from(rp.is_identity()) {
    return Err(error("signature yields infinity"));
  }
  let point = rp.to_encoded_point(false);
  Ok((
    point.x().unwrap().as_slice().try_into().unwrap(),
    point.y().unwrap().as_slice().try_into().unwrap(),
  ))
}

/// Public preprocessing for one (r,c) shape. Reusable across messages and keys.
pub struct Prepared {
  r: usize,
  c: usize,
  pk: neutronnova::ProverKey,
  vk: neutronnova::VerificationKey,
}

impl Prepared {
  /// Prepare 2^c SHA chunks of 2^r compressions and one P-256 core.
  pub fn setup(r: usize, c: usize) -> Result<Self> {
    let i = r.checked_add(c).ok_or_else(|| error("exponent overflow"))?;
    if !(3..=16).contains(&i) {
      return Err(error("expected 3 <= r+c <= 16"));
    }
    let step = ShaChunk {
      input: None,
      blocks: vec![None; 1 << r],
      output: None,
    };
    let core = SignatureCircuit {
      statement: None,
      digest: None,
      r_point: None,
    };
    let (pk, vk) = neutronnova::setup(step, core, 1 << c, 2048)?;
    Ok(Self { r, c, pk, vk })
  }
  /// Original and padded sizes for the SHA and P-256 circuits.
  pub fn sizes(&self) -> &[CircuitSize; 2] {
    &self.pk.sizes
  }
  /// Total number of compressions, including the final padding block.
  pub fn compressions(&self) -> usize {
    1 << (self.r + self.c)
  }
  /// Required message length before the mandatory padding block.
  pub fn message_bytes(&self) -> usize {
    64 * (self.compressions() - 1)
  }
  fn check_statement(&self, statement: &Statement) -> Result<()> {
    if statement.log_compressions as usize != self.r + self.c {
      return Err(error("statement exponent differs from setup"));
    }
    Ok(())
  }
  /// Generate native chaining states, signature hints, and circuit witnesses.
  pub fn generate_witness(&self, statement: &Statement, message: &[u8]) -> Result<Witness> {
    self.check_statement(statement)?;
    if message.len() != self.message_bytes() {
      return Err(error("wrong message length"));
    }
    let mut bytes = message.to_vec();
    bytes.extend_from_slice(&padding(self.compressions()));
    let mut state = IV;
    let mut steps = Vec::with_capacity(1 << self.c);
    for chunk in bytes.chunks_exact(64 * (1 << self.r)) {
      let input = state;
      let mut blocks = Vec::with_capacity(1 << self.r);
      for b in chunk.chunks_exact(64) {
        let block: [u8; 64] = b.try_into().unwrap();
        blocks.push(Some(words::<16>(&block)));
        sha2::compress256(&mut state, &[block.into()]);
      }
      steps.push(ShaChunk {
        input: Some(input),
        blocks,
        output: Some(state),
      });
    }
    let digest: [u8; 32] = state
      .into_iter()
      .flat_map(u32::to_be_bytes)
      .collect::<Vec<_>>()
      .try_into()
      .unwrap();
    let r_point = signature_hint(statement, &digest)?;
    let core = SignatureCircuit {
      statement: Some(statement.clone()),
      digest: Some(digest),
      r_point: Some(r_point),
    };
    neutronnova::generate_witness(&self.pk, steps, core)
  }
  /// Commit the fresh witnesses before any protocol challenges.
  pub fn commit(&self, witness: Witness) -> Result<Committed> {
    neutronnova::commit(&self.pk, witness)
  }
  /// Prove the committed relation, returning separate phase measurements.
  pub fn prove(&self, statement: &Statement, committed: &Committed) -> Result<(Proof, Phases)> {
    self.check_statement(statement)?;
    neutronnova::prove(&self.pk, &statement.to_bytes(), committed)
  }
  /// Verify the proof and all public statement, chain, padding, and digest links.
  pub fn verify(&self, statement: &Statement, proof: &Proof) -> Result<()> {
    self.check_statement(statement)?;
    // These comparisons are part of verification, never fixture-only checks.
    let core = proof.core_public_values();
    if core.len() != 40 {
      return Err(error("wrong core public input length"));
    }
    let expected = [statement.qx, statement.qy, statement.r, statement.s]
      .iter()
      .flat_map(scalar_words)
      .collect::<Vec<_>>();
    if core[8..] != expected {
      return Err(error("key or signature mismatch"));
    }
    let mut previous = IV.map(|w| Scalar::from(u64::from(w))).to_vec();
    let steps = proof.step_public_values().collect::<Vec<_>>();
    if steps.len() != 1 << self.c {
      return Err(error("wrong SHA instance count"));
    }
    let block_words = 16 * (1 << self.r);
    for (i, step) in steps.iter().enumerate() {
      if step.len() != 16 + block_words || step[..8] != previous {
        return Err(error("broken SHA chain"));
      }
      if i + 1 == steps.len() {
        let pad = words::<16>(&padding(self.compressions())).map(|v| Scalar::from(u64::from(v)));
        if step[8 + block_words - 16..8 + block_words] != pad {
          return Err(error("wrong SHA padding"));
        }
      }
      previous = step[8 + block_words..].to_vec();
    }
    if core[..8] != previous {
      return Err(error("SHA/ECDSA digest mismatch"));
    }
    neutronnova::verify(&self.vk, &statement.to_bytes(), proof)
  }
}
