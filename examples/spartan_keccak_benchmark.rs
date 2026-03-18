// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! examples/spartan_keccak_benchmark.rs
//!
//! Spartan SNARK benchmark for Keccak-256 comparing:
//! - Small-value path (i8 shape, bool witness)
//! - Field-element path (standard)
//!
//! Run with: `RUST_LOG=info cargo run --release --no-default-features --example spartan_keccak_benchmark`

#[cfg(feature = "jem")]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

use spartan2::{
  keccak_circuits::KeccakChainCircuit,
  spartan::SpartanSNARK,
  timing::{SPARTAN_PHASES, TimingLayer, clear_timings, print_table, snapshot_timings},
  traits::{Engine, snark::R1CSSNARKTrait},
};
use std::time::Instant;
use tracing::{info, info_span};
use tracing_subscriber::{EnvFilter, Layer as _, layer::SubscriberExt, util::SubscriberInitExt};

type E = spartan2::provider::Bn254Engine;
type F = <E as Engine>::Scalar;

fn main() {
  let (timing_layer, timing_data, constraints_data) = TimingLayer::new();

  tracing_subscriber::registry()
    .with(timing_layer)
    .with(
      tracing_subscriber::fmt::layer()
        .with_target(false)
        .with_ansi(true)
        .with_writer(std::io::stderr)
        .with_filter(EnvFilter::from_default_env()),
    )
    .init();

  let args: Vec<String> = std::env::args().collect();
  let input_bytes: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(64);
  let chain_length: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);

  let input: Vec<u8> = (0..input_bytes).map(|i| i as u8).collect();

  let circuit = KeccakChainCircuit::<F>::new(input, chain_length);

  let _root = info_span!("keccak_bench", input_bytes, chain_length).entered();
  info!("===== Keccak-256 Spartan SNARK Benchmark (input_bytes={input_bytes}, chain_length={chain_length}) =====");

  // ─── FIELD-ELEMENT PATH (baseline) ───

  info!("--- Field-element path ---");
  clear_timings(&timing_data);

  let t0 = Instant::now();
  let (pk, vk) = SpartanSNARK::<E>::setup(circuit.clone()).expect("setup failed");
  let setup_ms = t0.elapsed().as_millis();
  info!(elapsed_ms = setup_ms, "setup");

  let t0 = Instant::now();
  let prep = SpartanSNARK::<E>::prep_prove(&pk, circuit.clone(), false).expect("prep failed");
  let field_prep_ms = t0.elapsed().as_millis();
  info!(elapsed_ms = field_prep_ms, "prep_prove (field)");

  let t0 = Instant::now();
  let proof = SpartanSNARK::<E>::prove(&pk, circuit.clone(), &prep, false).expect("prove failed");
  let field_prove_ms = t0.elapsed().as_millis();
  info!(elapsed_ms = field_prove_ms, "prove (field)");

  let large_timings = snapshot_timings(&timing_data, SPARTAN_PHASES);

  let t0 = Instant::now();
  proof.verify(&vk).expect("verify failed");
  let verify_ms = t0.elapsed().as_millis();
  info!(elapsed_ms = verify_ms, "verify");

  // ─── SMALL-VALUE PATH (i8 shape, bool witness) ───

  info!("--- Small-value path (i8 shape, bool witness) ---");
  clear_timings(&timing_data);

  let t0 = Instant::now();
  let pk_small = SpartanSNARK::<E>::setup_small::<i8, _>(&circuit, &vk).expect("setup_small failed");
  let setup_small_ms = t0.elapsed().as_millis();
  info!(elapsed_ms = setup_small_ms, "setup_small");

  let t0 = Instant::now();
  let prep_small = SpartanSNARK::<E>::prep_prove_small::<_, i8, bool>(&pk_small, &circuit)
    .expect("prep_prove_small failed");
  let small_prep_ms = t0.elapsed().as_millis();
  info!(elapsed_ms = small_prep_ms, "prep_prove_small");

  let t0 = Instant::now();
  let proof_small = SpartanSNARK::<E>::prove_small_value(&pk_small, circuit.clone(), &prep_small)
    .expect("prove_small_value failed");
  let small_prove_ms = t0.elapsed().as_millis();
  info!(elapsed_ms = small_prove_ms, "prove_small_value");

  let small_timings = snapshot_timings(&timing_data, SPARTAN_PHASES);

  let t0 = Instant::now();
  proof_small.verify(&vk).expect("verify failed");
  let verify_small_ms = t0.elapsed().as_millis();
  info!(elapsed_ms = verify_small_ms, "verify (small)");

  // ─── Results ───

  let constraints = constraints_data.lock().unwrap().take();
  let header = match constraints {
    Some(c) => format!(
      "===== Keccak input_bytes={input_bytes}, chain_length={chain_length}, constraints={c} ====="
    ),
    None => format!(
      "===== Keccak input_bytes={input_bytes}, chain_length={chain_length} ====="
    ),
  };
  print_table(&header, SPARTAN_PHASES, &small_timings, &large_timings);

  let total_field = field_prep_ms + field_prove_ms;
  let total_small = small_prep_ms + small_prove_ms;
  let speedup = if total_small > 0 {
    total_field as f64 / total_small as f64
  } else {
    f64::INFINITY
  };

  println!();
  println!("Total proving (field):       {total_field} ms");
  println!("Total proving (small-value): {total_small} ms");
  println!("Speedup:                     {speedup:.2}x");
}
