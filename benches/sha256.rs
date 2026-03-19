// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT

//! Criterion benchmark for SHA-256 Spartan proving pipeline.
//!
//! Usage:
//!   cargo bench --bench sha256                       # default: 128 bytes
//!   cargo bench --bench sha256 -- --bytes 64         # custom size
//!   cargo bench --bench sha256 -- --bytes 2048       # large

use criterion::{Criterion, criterion_group, criterion_main};
use spartan2::{
  provider::Bn254Engine,
  sha256_circuits::SmallSha256Circuit,
  small_field::{DelayedReduction, SmallValueField},
  spartan::SpartanSNARK,
  traits::{Engine, snark::R1CSSNARKTrait},
};
use std::sync::OnceLock;

/// Read message size from `BENCH_BYTES` env var, defaulting to 128.
///
/// Usage:
///   BENCH_BYTES=128  cargo bench --bench sha256
///   BENCH_BYTES=2048 cargo bench --bench sha256
fn parse_bytes_arg() -> usize {
  std::env::var("BENCH_BYTES")
    .ok()
    .and_then(|v| v.parse::<usize>().ok())
    .unwrap_or(128)
}

static BYTES: OnceLock<usize> = OnceLock::new();

fn get_bytes() -> usize {
  *BYTES.get_or_init(parse_bytes_arg)
}

/// Benchmark setup + prove + verify for a given engine and small/large mode.
fn bench_prove<E: Engine>(c: &mut Criterion, mode_name: &str, is_small: bool)
where
  E::Scalar: SmallValueField<i64>
    + DelayedReduction<i64>
    + DelayedReduction<i128>
    + DelayedReduction<E::Scalar>,
{
  let msg_len = get_bytes();
  let circuit = SmallSha256Circuit::<E::Scalar>::new(vec![0u8; msg_len], true);

  // Setup once (not timed by criterion)
  let (pk, vk) = SpartanSNARK::<E>::setup(circuit.clone()).expect("setup failed");

  let group_name = format!("sha256_{msg_len}B/{mode_name}");
  let mut group = c.benchmark_group(&group_name);
  group.sample_size(10);

  group.bench_function("prove", |b| {
    b.iter(|| {
      let prep = SpartanSNARK::<E>::prep_prove(&pk, circuit.clone(), is_small)
        .expect("prep_prove failed");
      let proof = SpartanSNARK::<E>::prove(&pk, circuit.clone(), &prep, is_small)
        .expect("prove failed");
      proof
    });
  });

  group.bench_function("verify", |b| {
    let prep = SpartanSNARK::<E>::prep_prove(&pk, circuit.clone(), is_small)
      .expect("prep_prove failed");
    let proof = SpartanSNARK::<E>::prove(&pk, circuit.clone(), &prep, is_small)
      .expect("prove failed");
    b.iter(|| {
      proof.verify(&vk).expect("verify failed");
    });
  });

  group.finish();
}

fn sha256_bench(c: &mut Criterion) {
  bench_prove::<Bn254Engine>(c, "small", true);
  bench_prove::<Bn254Engine>(c, "large", false);
}

criterion_group!(benches, sha256_bench);
criterion_main!(benches);
