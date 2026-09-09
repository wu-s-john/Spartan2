use super::*;
use bellpepper_core::test_cs::TestConstraintSystem;
use ff::Field;
use p256::ecdsa::{
  Signature, SigningKey,
  signature::{Signer, Verifier, hazmat::PrehashSigner},
};
use sha2::{Digest, Sha256};

fn fixture(i: u8, seed: u8) -> (Statement, Vec<u8>) {
  let message = (0..64 * ((1usize << i) - 1))
    .map(|j| (j as u8).wrapping_add(seed))
    .collect::<Vec<_>>();
  let key = SigningKey::from_bytes((&[seed + 1; 32]).into()).unwrap();
  let signature: Signature = key.sign(&message);
  key.verifying_key().verify(&message, &signature).unwrap();
  (statement(i, &key, &signature), message)
}

fn statement(i: u8, key: &SigningKey, sig: &Signature) -> Statement {
  let q = key.verifying_key().to_encoded_point(false);
  let (r, s) = sig.split_bytes();
  Statement {
    log_compressions: i,
    qx: q.x().unwrap().as_slice().try_into().unwrap(),
    qy: q.y().unwrap().as_slice().try_into().unwrap(),
    r: r.into(),
    s: s.into(),
  }
}

fn satisfied(circuit: SignatureCircuit) -> bool {
  let mut cs = TestConstraintSystem::<Scalar>::new();
  circuit.synthesize(&mut cs).unwrap();
  cs.is_satisfied()
}

#[test]
fn p256_circuit_accepts_digest_boundaries_and_high_s() {
  let key = SigningKey::from_bytes((&[17u8; 32]).into()).unwrap();
  let n = num_bigint::BigUint::parse_bytes(p256_circuit::ORDER.as_bytes(), 16).unwrap();
  let p = num_bigint::BigUint::parse_bytes(
    b"ffffffff00000001000000000000000000000000ffffffffffffffffffffffff",
    16,
  )
  .unwrap();
  let mut digests = vec![[0u8; 32], [255u8; 32]];
  for v in [&n - 1u8, n.clone(), &p - 1u8, p] {
    let b = v.to_bytes_be();
    let mut digest = [0u8; 32];
    digest[32 - b.len()..].copy_from_slice(&b);
    digests.push(digest);
  }
  for digest in digests {
    let signature: Signature = key.sign_prehash(&digest).unwrap();
    let mut s = statement(3, &key, &signature);
    let hint = signature_hint(&s, &digest).unwrap();
    assert!(satisfied(SignatureCircuit {
      statement: Some(s.clone()),
      digest: Some(digest),
      r_point: Some(hint)
    }));
    let alternate = &n - num_bigint::BigUint::from_bytes_be(&s.s);
    let b = alternate.to_bytes_be();
    s.s = [0u8; 32];
    s.s[32 - b.len()..].copy_from_slice(&b);
    let hint = signature_hint(&s, &digest).unwrap();
    assert!(satisfied(SignatureCircuit {
      statement: Some(s),
      digest: Some(digest),
      r_point: Some(hint)
    }));
  }
}

#[test]
fn p256_constraints_reject_invalid_values_with_allocated_hints() {
  let (statement, message) = fixture(3, 4);
  let digest: [u8; 32] = Sha256::digest(&message).into();
  let hint = signature_hint(&statement, &digest).unwrap();
  let valid = SignatureCircuit {
    statement: Some(statement),
    digest: Some(digest),
    r_point: Some(hint),
  };
  for variant in 0..6 {
    let mut bad = valid.clone();
    let s = bad.statement.as_mut().unwrap();
    match variant {
      0 => s.r = [0; 32],
      1 => s.s = [0; 32],
      2 => s.r = [255; 32],
      3 => s.qy[31] ^= 1,
      4 => bad.digest.as_mut().unwrap()[0] ^= 1,
      5 => bad.r_point.as_mut().unwrap().1[31] ^= 1,
      _ => unreachable!(),
    }
    assert!(!satisfied(bad), "accepted variant {variant}");
  }
}

#[test]
fn sha_chain_and_signature_proofs_cover_chunkings_and_tampering() {
  for (r, c) in [(3, 0), (2, 1), (0, 3)] {
    let prepared = Prepared::setup(r, c).unwrap();
    let (statement, message) = fixture(3, 5);
    let witness = prepared.generate_witness(&statement, &message).unwrap();
    let committed = prepared.commit(witness).unwrap();
    let (proof, _) = prepared.prove(&statement, &committed).unwrap();
    let bytes = proof.to_bytes().unwrap();
    let decoded = Proof::from_bytes(&bytes).unwrap();
    prepared.verify(&statement, &decoded).unwrap();
    let mut bad = statement.clone();
    bad.s[0] ^= 1;
    assert!(prepared.verify(&bad, &proof).is_err());
    let mut bad = proof.clone();
    bad.steps[0].X[0] += Scalar::ONE;
    assert!(prepared.verify(&statement, &bad).is_err());
    let mut bad = proof.clone();
    bad.core.X[0] += Scalar::ONE;
    assert!(prepared.verify(&statement, &bad).is_err());
    let mut bad = proof.clone();
    let last = bad.steps.last_mut().unwrap();
    let len = last.X.len();
    last.X[len - 9] += Scalar::ONE;
    assert!(prepared.verify(&statement, &bad).is_err());
    if c > 0 {
      let mut bad = proof.clone();
      bad.steps[1].X[0] += Scalar::ONE;
      assert!(prepared.verify(&statement, &bad).is_err());
    }
    assert!(
      prepared
        .generate_witness(&statement, &message[..message.len() - 1])
        .is_err()
    );
    // Reuse exactly the same setup with another message, key and signature.
    let (statement, message) = fixture(3, 11);
    let witness = prepared.generate_witness(&statement, &message).unwrap();
    let committed = prepared.commit(witness).unwrap();
    let (proof, _) = prepared.prove(&statement, &committed).unwrap();
    prepared.verify(&statement, &proof).unwrap();
  }
}

#[test]
fn invalid_signature_cannot_be_proved() {
  let prepared = Prepared::setup(1, 2).unwrap();
  let (mut statement, message) = fixture(3, 7);
  statement.s[31] ^= 1;
  let witness = prepared.generate_witness(&statement, &message).unwrap();
  let committed = prepared.commit(witness).unwrap();
  assert!(prepared.prove(&statement, &committed).is_err());
}
