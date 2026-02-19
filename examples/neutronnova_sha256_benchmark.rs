// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! examples/neutronnova_sha256_benchmark.rs
//! Benchmark NeutronNova NIFS folding of SHA-256 hash chains comparing:
//! - Small-value NIFS sumcheck
//! - Large-value (vanilla) NIFS sumcheck
//!
//! Run with: `RUST_LOG=info cargo run --release --example neutronnova_sha256_benchmark`

#[cfg(feature = "jem")]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

use clap::{Parser, ValueEnum};
use spartan2::{
  bellpepper::{
    r1cs::{MultiRoundSpartanShape, MultiRoundSpartanWitness, SpartanWitness},
    shape_cs::ShapeCS,
    solver::SatisfyingAssignment,
  },
  cli::FieldChoice,
  math::Math,
  neutronnova_zk::{
    NeutronNovaNIFS, NeutronNovaPrepZkSNARK, NeutronNovaProverKey, NeutronNovaVerifierKey,
    NeutronNovaZkSNARK,
  },
  provider::{Bn254Engine, PallasHyraxEngine, VestaHyraxEngine},
  r1cs::{R1CSInstance, R1CSWitness, SplitR1CSShape},
  sha256_circuits::{NativeSmallSha256ChainCircuit, SmallSha256ChainCircuit},
  small_field::{DelayedReduction, SmallValueField, WideMul},
  small_r1cs::{Accumulator, WideningMul, Witness},
  timing::{
    NEUTRONNOVA_PHASES, NEUTRONNOVA_ZK_PROVE_PHASES, TimingData, TimingLayer, clear_timings,
    normalize_parallel_timings, print_table, snapshot_timings,
  },
  traits::{Engine, circuit::SpartanCircuit, pcs::FoldingEngineTrait, transcript::TranscriptEngineTrait},
  zk::NeutronNovaVerifierCircuit,
};
use std::{collections::HashMap, time::Instant};
use tracing::{info, info_span};
use tracing_subscriber::{EnvFilter, Layer as _, layer::SubscriberExt, util::SubscriberInitExt};

/// Benchmark mode
#[derive(ValueEnum, Clone, Default, Debug)]
enum BenchMode {
  /// Benchmark NIFS folding only
  #[default]
  Nifs,
  /// Benchmark full NeutronNovaZkSNARK::prove
  ZkProve,
  /// Benchmark native i32 arithmetic path (SmallCS + prove_native)
  Native,
}

#[derive(Parser)]
#[command(about = "NeutronNova benchmark: small vs large sumcheck")]
struct Args {
  #[arg(long, default_value = "4")]
  instances: usize,
  #[arg(long, default_value = "4")]
  chain_length: usize,
  #[arg(long, value_enum, default_value = "bn254-fr")]
  field: FieldChoice,
  #[arg(long, value_enum, default_value = "nifs")]
  mode: BenchMode,
}

/// Generate step circuits with distinct inputs
fn make_circuits<F: ff::PrimeField + ff::PrimeFieldBits>(
  num_instances: usize,
  chain_length: usize,
) -> Vec<SmallSha256ChainCircuit<F>> {
  (0..num_instances)
    .map(|i| {
      let mut input = [0u8; 32];
      input[0] = i as u8;
      input[1] = (i >> 8) as u8;
      SmallSha256ChainCircuit::new(input, chain_length)
    })
    .collect()
}

/// Generate instances and witnesses from circuits using the NeutronNova pipeline.
fn generate_instances_and_witnesses<E, C>(
  pk: &NeutronNovaProverKey<E>,
  prep: &NeutronNovaPrepZkSNARK<E>,
  step_circuits: &[C],
  is_small: bool,
) -> (Vec<R1CSInstance<E>>, Vec<R1CSWitness<E>>)
where
  E: Engine,
  E::PCS: FoldingEngineTrait<E>,
  C: SpartanCircuit<E>,
{
  let mut ps_step: Vec<_> = prep.ps_step.clone();

  let (instances, witnesses): (Vec<_>, Vec<_>) = ps_step
    .iter_mut()
    .zip(step_circuits.iter().enumerate())
    .map(|(pre_state, (i, circuit))| {
      let mut transcript = E::TE::new(b"neutronnova_prove");
      transcript.absorb(b"vk", &pk.vk_digest);
      transcript.absorb(
        b"num_circuits",
        &E::Scalar::from(step_circuits.len() as u64),
      );
      transcript.absorb(b"circuit_index", &E::Scalar::from(i as u64));

      let public_values = circuit.public_values().expect("public_values failed");
      transcript.absorb(b"public_values", &public_values.as_slice());

      SatisfyingAssignment::r1cs_instance_and_witness(
        pre_state,
        &pk.S_step,
        &pk.ck,
        circuit,
        is_small,
        &mut transcript,
      )
      .expect("r1cs_instance_and_witness failed")
    })
    .unzip();

  let instances_regular: Vec<_> = instances
    .iter()
    .map(|u| u.to_regular_instance().expect("to_regular_instance failed"))
    .collect();

  (instances_regular, witnesses)
}

/// Run NeutronNovaNIFS::prove with fresh vc/vc_state/transcript.
fn nifs_prove_single<E: Engine>(
  pk: &NeutronNovaProverKey<E>,
  instances: &[R1CSInstance<E>],
  witnesses: &[R1CSWitness<E>],
  is_small: bool,
) where
  E::PCS: FoldingEngineTrait<E>,
  E::Scalar: SmallValueField<i64>
    + DelayedReduction<i64>
    + DelayedReduction<i128>
    + DelayedReduction<E::Scalar>,
{
  let n_padded = instances.len().next_power_of_two();
  let num_vars = pk.S_step.num_shared + pk.S_step.num_precommitted + pk.S_step.num_rest;
  let num_rounds_b = n_padded.log_2();
  let num_rounds_x = pk.S_step.num_cons.log_2();
  let num_rounds_y = num_vars.log_2() + 1;

  let mut vc = NeutronNovaVerifierCircuit::<E>::default(num_rounds_b, num_rounds_x, num_rounds_y);
  let mut vc_state =
    SatisfyingAssignment::<E>::initialize_multiround_witness(&pk.vc_shape).expect("init vc_state");

  let mut transcript = E::TE::new(b"neutronnova_prove");
  transcript.absorb(b"vk", &pk.vk_digest);

  NeutronNovaNIFS::<E>::prove(
    &pk.S_step,
    instances,
    witnesses,
    &mut vc,
    &mut vc_state,
    &pk.vc_shape,
    &pk.vc_ck,
    &mut transcript,
    is_small,
  )
  .expect("NeutronNovaNIFS::prove failed");
}

/// Verify a SNARK and print result.
fn verify_snark<E: Engine>(
  pk: &NeutronNovaProverKey<E>,
  vk: &NeutronNovaVerifierKey<E>,
  circuits: &[SmallSha256ChainCircuit<E::Scalar>],
  core_circuit: &SmallSha256ChainCircuit<E::Scalar>,
  num_instances: usize,
  is_small: bool,
) where
  E::PCS: FoldingEngineTrait<E>,
  E::Scalar: SmallValueField<i64>
    + DelayedReduction<i64>
    + DelayedReduction<i128>
    + DelayedReduction<E::Scalar>,
{
  let mode = if is_small { "small-value" } else { "large-value" };
  let snark = if is_small {
    let prep = NeutronNovaZkSNARK::<E>::prep_prove_small(pk, circuits, core_circuit).expect("prep_prove_small");
    NeutronNovaZkSNARK::<E>::prove_small(pk, circuits, core_circuit, &prep).expect("prove_small")
  } else {
    let prep = NeutronNovaZkSNARK::<E>::prep_prove(pk, circuits, core_circuit).expect("prep_prove");
    NeutronNovaZkSNARK::<E>::prove(pk, circuits, core_circuit, &prep).expect("prove")
  };
  let res = snark.verify(vk, num_instances);
  assert!(res.is_ok(), "Verification failed: {:?}", res.err());
  eprintln!("  verified: yes ({})", mode);
}

/// Parallel spans that need to be normalized by dividing by parallelism factor.
/// These spans are called once per circuit and run in parallel.
const PARALLEL_SPANS: &[&str] = &["precom_syn", "commit_pre"];

fn benchmark_nifs_prove<E: Engine>(
  num_instances: usize,
  chain_length: usize,
  timing_data: &TimingData,
) where
  E::PCS: FoldingEngineTrait<E>,
  E::Scalar: SmallValueField<i64>
    + DelayedReduction<i64>
    + DelayedReduction<i128>
    + DelayedReduction<E::Scalar>,
{
  let num_cores = rayon::current_num_threads();
  // +1 for core circuit which also runs precommitted_witness
  let parallel_divisor = (num_instances + 1).min(num_cores);

  let circuits = make_circuits::<E::Scalar>(num_instances, chain_length);
  let core_circuit = circuits[0].clone();

  eprintln!(
    "Setting up NeutronNova for {} instances, chain_length={}, cores={}...",
    num_instances, chain_length, num_cores
  );
  let t0 = Instant::now();
  let (pk, vk) =
    NeutronNovaZkSNARK::<E>::setup(&circuits[0], &core_circuit, num_instances).expect("setup");
  let setup_ms = t0.elapsed().as_millis();
  eprintln!("Setup done in {} ms", setup_ms);

  let mut small_timings = HashMap::new();
  let mut large_timings = HashMap::new();

  for is_small in [true, false] {
    let mode = if is_small { "small" } else { "large" };
    let _mode_span = info_span!("mode", mode).entered();

    clear_timings(timing_data);

    let t_total = Instant::now();

    // Witness generation: prep_prove and synthesize instances (using appropriate method based on is_small)
    let (instances, witnesses) = if is_small {
      let prep = NeutronNovaZkSNARK::<E>::prep_prove_small(&pk, &circuits, &core_circuit)
        .expect("prep_prove_small");
      generate_instances_and_witnesses(&pk, &prep, &circuits, is_small)
    } else {
      let prep = NeutronNovaZkSNARK::<E>::prep_prove(&pk, &circuits, &core_circuit)
        .expect("prep_prove");
      generate_instances_and_witnesses(&pk, &prep, &circuits, is_small)
    };

    // NIFS prove
    nifs_prove_single(&pk, &instances, &witnesses, is_small);

    let total_ms = t_total.elapsed().as_millis();
    info!(elapsed_ms = total_ms as u64, "end_to_end_total");

    let mut timings = snapshot_timings(timing_data, NEUTRONNOVA_PHASES);
    // Normalize parallel spans to approximate wall-clock time
    normalize_parallel_timings(&mut timings, PARALLEL_SPANS, parallel_divisor);
    if is_small {
      small_timings = timings;
    } else {
      large_timings = timings;
    }
  }

  // Verify using the full ZkSNARK pipeline
  verify_snark(&pk, &vk, &circuits, &core_circuit, num_instances, true);
  verify_snark(&pk, &vk, &circuits, &core_circuit, num_instances, false);

  let header = format!(
    "===== NeutronNova NIFS: instances={}, chain_length={}, constraints={}, cores={} =====",
    num_instances, chain_length, pk.S_step.num_cons, num_cores,
  );
  print_table(&header, NEUTRONNOVA_PHASES, &small_timings, &large_timings);
}

fn benchmark_zk_prove<E: Engine>(
  num_instances: usize,
  chain_length: usize,
  timing_data: &TimingData,
) where
  E::PCS: FoldingEngineTrait<E>,
  E::Scalar: SmallValueField<i64>
    + DelayedReduction<i64>
    + DelayedReduction<i128>
    + DelayedReduction<E::Scalar>,
{
  let num_cores = rayon::current_num_threads();
  // +1 for core circuit which also runs precommitted_witness
  let parallel_divisor = (num_instances + 1).min(num_cores);

  let circuits = make_circuits::<E::Scalar>(num_instances, chain_length);
  let core_circuit = circuits[0].clone();

  eprintln!(
    "Setting up NeutronNova for {} instances, chain_length={}, cores={}...",
    num_instances, chain_length, num_cores
  );
  let t0 = Instant::now();
  let (pk, vk) =
    NeutronNovaZkSNARK::<E>::setup(&circuits[0], &core_circuit, num_instances).expect("setup");
  let setup_ms = t0.elapsed().as_millis();
  eprintln!("Setup done in {} ms", setup_ms);

  let mut small_timings = HashMap::new();
  let mut large_timings = HashMap::new();

  for is_small in [true, false] {
    let mode = if is_small { "small" } else { "large" };
    let _mode_span = info_span!("mode", mode).entered();

    clear_timings(timing_data);

    let t_total = Instant::now();

    // Full ZK prove (using appropriate method based on is_small)
    let snark = if is_small {
      let prep = NeutronNovaZkSNARK::<E>::prep_prove_small(&pk, &circuits, &core_circuit)
        .expect("prep_prove_small");
      NeutronNovaZkSNARK::<E>::prove_small(&pk, &circuits, &core_circuit, &prep)
        .expect("prove_small")
    } else {
      let prep = NeutronNovaZkSNARK::<E>::prep_prove(&pk, &circuits, &core_circuit)
        .expect("prep_prove");
      NeutronNovaZkSNARK::<E>::prove(&pk, &circuits, &core_circuit, &prep)
        .expect("prove")
    };

    let total_ms = t_total.elapsed().as_millis();
    info!(elapsed_ms = total_ms as u64, "end_to_end_total");

    let mut timings = snapshot_timings(timing_data, NEUTRONNOVA_ZK_PROVE_PHASES);
    // Normalize parallel spans to approximate wall-clock time
    normalize_parallel_timings(&mut timings, PARALLEL_SPANS, parallel_divisor);
    if is_small {
      small_timings = timings;
    } else {
      large_timings = timings;
    }

    // Verify
    let res = snark.verify(&vk, num_instances);
    assert!(res.is_ok(), "Verification failed: {:?}", res.err());
    eprintln!("  verified: yes ({})", mode);
  }

  let header = format!(
    "===== NeutronNova ZkProve: instances={}, chain_length={}, constraints={}, cores={} =====",
    num_instances, chain_length, pk.S_step.num_cons, num_cores,
  );
  print_table(
    &header,
    NEUTRONNOVA_ZK_PROVE_PHASES,
    &small_timings,
    &large_timings,
  );
}

/// Generate native SHA-256 circuits (using SmallCS + small_gadgets)
fn make_native_circuits(
  num_instances: usize,
  chain_length: usize,
) -> Vec<NativeSmallSha256ChainCircuit> {
  (0..num_instances)
    .map(|i| {
      let mut input = [0u8; 32];
      input[0] = i as u8;
      input[1] = (i >> 8) as u8;
      NativeSmallSha256ChainCircuit::new(input, chain_length)
    })
    .collect()
}

/// Benchmark native i32 arithmetic path using SmallCS + prove_native.
///
/// This path uses:
/// - NativeSmallSha256ChainCircuit (SmallCS<i32, i32> + small_gadgets/sha256)
/// - SplitR1CSShape<E, i32> (small coefficients)
/// - R1CSWitness<E, i32> / R1CSInstance<E, i32> (small values)
/// - NeutronNovaNIFS::prove_native with pure i32 × i32 → i64 arithmetic
fn benchmark_native_prove<E: Engine>(
  num_instances: usize,
  chain_length: usize,
  _timing_data: &TimingData,
) where
  E::PCS: FoldingEngineTrait<E>,
  E::Scalar: SmallValueField<i32>
    + SmallValueField<i64>
    + DelayedReduction<i32>
    + DelayedReduction<i64>
    + DelayedReduction<i128>
    + DelayedReduction<E::Scalar>,
  i32: WideningMul<i32, i64>,
  i64: Witness + WideMul + Accumulator,
  <i64 as WideMul>::Product: Copy + Ord + num_traits::Signed
    + std::ops::Div<Output = <i64 as WideMul>::Product>
    + std::ops::Mul<Output = <i64 as WideMul>::Product>
    + num_traits::One
    + From<i32>,
{
  let num_cores = rayon::current_num_threads();

  eprintln!(
    "Setting up Native NeutronNova for {} instances, chain_length={}, cores={}...",
    num_instances, chain_length, num_cores
  );

  // Create native circuits
  let circuits = make_native_circuits(num_instances, chain_length);

  // Get shape from first circuit
  let t0 = Instant::now();
  let shape: SplitR1CSShape<E, i32> = circuits[0].to_shape();

  // Create commitment key from shape (need field version for CK setup)
  let shape_field: SplitR1CSShape<E> = SplitR1CSShape::new_simple(
    shape.num_cons_unpadded,
    shape.num_rest_unpadded,
    shape.num_public,
    shape.A.map_coeffs(|c| <E as Engine>::Scalar::from(c as u64)),
    shape.B.map_coeffs(|c| <E as Engine>::Scalar::from(c as u64)),
    shape.C.map_coeffs(|c| <E as Engine>::Scalar::from(c as u64)),
  );
  let (ck, _vk_ee) = SplitR1CSShape::commitment_key(&[&shape_field]).expect("commitment_key");

  // Set up VC infrastructure
  let n_padded = num_instances.next_power_of_two();
  let num_vars = shape.num_shared + shape.num_precommitted + shape.num_rest;
  let num_rounds_b = n_padded.log_2();
  let num_rounds_x = shape.num_cons.log_2();
  let num_rounds_y = num_vars.log_2() + 1;

  let vc_template = NeutronNovaVerifierCircuit::<E>::default(num_rounds_b, num_rounds_x, num_rounds_y);
  let (vc_shape, vc_ck, _vk_mr) =
    <ShapeCS<E> as MultiRoundSpartanShape<E>>::multiround_r1cs_shape(&vc_template).expect("vc shape");

  let setup_ms = t0.elapsed().as_millis();
  eprintln!("Setup done in {} ms (constraints: {})", setup_ms, shape.num_cons);

  // Generate witnesses and instances (native i32)
  let t_witness = Instant::now();
  let (witnesses, instances): (Vec<R1CSWitness<E, i32>>, Vec<R1CSInstance<E, i32>>) = circuits
    .iter()
    .map(|c| c.to_witness_and_instance(&ck, shape.num_rest).expect("to_witness_and_instance"))
    .unzip();
  let witness_gen_ms = t_witness.elapsed().as_millis();
  eprintln!("Witness generation done in {} ms", witness_gen_ms);

  // Run prove_native
  let t_prove = Instant::now();

  let mut vc = NeutronNovaVerifierCircuit::<E>::default(num_rounds_b, num_rounds_x, num_rounds_y);
  let mut vc_state =
    SatisfyingAssignment::<E>::initialize_multiround_witness(&vc_shape).expect("init vc_state");
  let mut transcript = <E as Engine>::TE::new(b"neutronnova_native_benchmark");

  let result = NeutronNovaNIFS::<E>::prove_native::<i32, i64, i32>(
    &shape,
    &instances,
    &witnesses,
    &mut vc,
    &mut vc_state,
    &vc_shape,
    &vc_ck,
    &mut transcript,
  );

  let prove_ms = t_prove.elapsed().as_millis();

  match &result {
    Ok(_) => eprintln!("prove_native succeeded in {} ms", prove_ms),
    Err(e) => eprintln!("prove_native failed: {:?}", e),
  }

  // Print summary
  eprintln!("\n===== Native Path Summary =====");
  eprintln!("  Instances:    {}", num_instances);
  eprintln!("  Chain length: {}", chain_length);
  eprintln!("  Constraints:  {}", shape.num_cons);
  eprintln!("  Cores:        {}", num_cores);
  eprintln!("  Setup:        {} ms", setup_ms);
  eprintln!("  Witness gen:  {} ms", witness_gen_ms);
  eprintln!("  prove_native: {} ms", prove_ms);
  eprintln!("  Total:        {} ms", setup_ms + witness_gen_ms + prove_ms);
}

fn main() {
  let args = Args::parse();

  let (timing_layer, timing_data, _constraints_data) = TimingLayer::new();

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

  match (args.field, args.mode) {
    (FieldChoice::Bn254Fr, BenchMode::Nifs) => {
      benchmark_nifs_prove::<Bn254Engine>(args.instances, args.chain_length, &timing_data)
    }
    (FieldChoice::Bn254Fr, BenchMode::ZkProve) => {
      benchmark_zk_prove::<Bn254Engine>(args.instances, args.chain_length, &timing_data)
    }
    (FieldChoice::Bn254Fr, BenchMode::Native) => {
      benchmark_native_prove::<Bn254Engine>(args.instances, args.chain_length, &timing_data)
    }
    (FieldChoice::PallasFq, BenchMode::Nifs) => {
      benchmark_nifs_prove::<PallasHyraxEngine>(args.instances, args.chain_length, &timing_data)
    }
    (FieldChoice::PallasFq, BenchMode::ZkProve) => {
      benchmark_zk_prove::<PallasHyraxEngine>(args.instances, args.chain_length, &timing_data)
    }
    (FieldChoice::PallasFq, BenchMode::Native) => {
      benchmark_native_prove::<PallasHyraxEngine>(args.instances, args.chain_length, &timing_data)
    }
    (FieldChoice::VestaFp, BenchMode::Nifs) => {
      benchmark_nifs_prove::<VestaHyraxEngine>(args.instances, args.chain_length, &timing_data)
    }
    (FieldChoice::VestaFp, BenchMode::ZkProve) => {
      benchmark_zk_prove::<VestaHyraxEngine>(args.instances, args.chain_length, &timing_data)
    }
    (FieldChoice::VestaFp, BenchMode::Native) => {
      benchmark_native_prove::<VestaHyraxEngine>(args.instances, args.chain_length, &timing_data)
    }
  }
}
