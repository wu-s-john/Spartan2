// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT

//! examples/neutronnova_bench.rs
//! Benchmark NeutronNova SHA-256 chain proving with timing table output.
//!
//! Run with: `RUST_LOG=info cargo run --release --example neutronnova_bench`

#[cfg(feature = "jem")]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

use bellpepper::gadgets::sha256::sha256;
use bellpepper_core::{
  ConstraintSystem, SynthesisError,
  boolean::{AllocatedBit, Boolean},
  num::AllocatedNum,
};
use clap::Parser;
use ff::{Field, PrimeField, PrimeFieldBits};
use sha2::{Digest, Sha256};
use spartan2::{
  neutronnova_zk::NeutronNovaZkSNARK,
  provider::Bn254Engine,
  timing::{NEUTRONNOVA_ZK_PROVE_PHASES, TimingLayer, clear_timings, print_single_table, snapshot_timings},
  traits::{Engine, circuit::SpartanCircuit},
};
use std::{marker::PhantomData, time::Instant};
use tracing::info;
use tracing_subscriber::{EnvFilter, Layer as _, layer::SubscriberExt, util::SubscriberInitExt};

type E = Bn254Engine;

#[derive(Parser)]
#[command(about = "Benchmark NeutronNova SHA-256 chain proving")]
struct Args {
  /// Number of instances to fold
  #[arg(short = 'n', long, default_value = "8")]
  instances: usize,

  /// Chain length (number of SHA256 iterations per instance)
  #[arg(short = 'c', long, default_value = "8")]
  chain: usize,
}

/// SHA256 chain circuit: computes H(H(H(...H(msg)))) for `chain_length` iterations.
#[derive(Clone, Debug)]
struct Sha256ChainCircuit<Scalar: PrimeField> {
  input: [u8; 32],
  chain_length: usize,
  _p: PhantomData<Scalar>,
}

impl<Scalar: PrimeField + PrimeFieldBits> Sha256ChainCircuit<Scalar> {
  fn new(input: [u8; 32], chain_length: usize) -> Self {
    Self {
      input,
      chain_length,
      _p: PhantomData,
    }
  }

  /// Compute the final hash after `chain_length` iterations.
  fn compute_final_hash(&self) -> [u8; 32] {
    let mut current = self.input;
    for _ in 0..self.chain_length {
      let mut hasher = Sha256::new();
      hasher.update(current);
      current = hasher.finalize().into();
    }
    current
  }
}

impl<E: Engine> SpartanCircuit<E> for Sha256ChainCircuit<E::Scalar> {
  fn public_values(&self) -> Result<Vec<<E as Engine>::Scalar>, SynthesisError> {
    // Public output is the final hash as bits
    let final_hash = self.compute_final_hash();
    let hash_scalars: Vec<<E as Engine>::Scalar> = final_hash
      .iter()
      .flat_map(|&byte| {
        (0..8).rev().map(move |i| {
          if (byte >> i) & 1 == 1 {
            E::Scalar::ONE
          } else {
            E::Scalar::ZERO
          }
        })
      })
      .collect();
    Ok(hash_scalars)
  }

  fn shared<CS: ConstraintSystem<E::Scalar>>(
    &self,
    _: &mut CS,
  ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
    Ok(vec![])
  }

  fn precommitted<CS: ConstraintSystem<E::Scalar>>(
    &self,
    cs: &mut CS,
    _: &[AllocatedNum<E::Scalar>],
  ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
    // Start with input bits
    let input_bit_values: Vec<_> = self
      .input
      .iter()
      .flat_map(|&byte| (0..8).rev().map(move |i| (byte >> i) & 1 == 1))
      .collect();

    let mut current_bits: Vec<Boolean> = input_bit_values
      .into_iter()
      .enumerate()
      .map(|(i, bit)| {
        AllocatedBit::alloc(cs.namespace(|| format!("input bit {i}")), Some(bit))
          .map(Boolean::from)
      })
      .collect::<Result<Vec<_>, _>>()?;

    // Chain SHA256 computations
    for round in 0..self.chain_length {
      current_bits = sha256(cs.namespace(|| format!("sha256 round {round}")), &current_bits)?;
    }

    // Verify against expected final hash
    let expected_hash = self.compute_final_hash();
    let mut expected_bits = expected_hash
      .iter()
      .flat_map(|&byte| (0..8).rev().map(move |i| (byte >> i) & 1 == 1));

    for b in &current_bits {
      match b {
        Boolean::Is(bit) => assert_eq!(expected_bits.next().unwrap(), bit.get_value().unwrap()),
        Boolean::Not(bit) => assert_ne!(expected_bits.next().unwrap(), bit.get_value().unwrap()),
        Boolean::Constant(_) => unreachable!(),
      }
    }

    // Expose final hash as public input
    for (i, bit) in current_bits.iter().enumerate() {
      let n = AllocatedNum::alloc_input(cs.namespace(|| format!("output bit {i}")), || {
        Ok(
          if bit.get_value().ok_or(SynthesisError::AssignmentMissing)? {
            E::Scalar::ONE
          } else {
            E::Scalar::ZERO
          },
        )
      })?;

      cs.enforce(
        || format!("bit == num {i}"),
        |_| bit.lc(CS::one(), E::Scalar::ONE),
        |lc| lc + CS::one(),
        |lc| lc + n.get_variable(),
      );
    }

    Ok(vec![])
  }

  fn num_challenges(&self) -> usize {
    0
  }

  fn synthesize<CS: ConstraintSystem<E::Scalar>>(
    &self,
    _: &mut CS,
    _: &[AllocatedNum<E::Scalar>],
    _: &[AllocatedNum<E::Scalar>],
    _: Option<&[E::Scalar]>,
  ) -> Result<(), SynthesisError> {
    Ok(())
  }
}

fn main() {
  let args = Args::parse();
  let num_instances = args.instances;
  let chain_length = args.chain;

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

  eprintln!(
    "\n======= NeutronNova SHA256 Chain: instances={}, chain={} =======",
    num_instances, chain_length
  );

  // Create circuits with different inputs
  let step_circuits: Vec<_> = (0..num_instances)
    .map(|i| {
      let mut input = [0u8; 32];
      input[0] = i as u8;
      Sha256ChainCircuit::<<E as Engine>::Scalar>::new(input, chain_length)
    })
    .collect();

  let core_circuit = step_circuits[0].clone();

  // SETUP
  let t0 = Instant::now();
  let (pk, vk) =
    NeutronNovaZkSNARK::<E>::setup(&step_circuits[0], &core_circuit, num_instances)
      .expect("setup failed");
  let setup_ms = t0.elapsed().as_millis();
  info!(elapsed_ms = setup_ms as u64, "setup");

  // Clear timing data before prove phases
  clear_timings(&timing_data);

  // PREP_PROVE (is_small = false)
  let t0 = Instant::now();
  let prep_snark =
    NeutronNovaZkSNARK::<E>::prep_prove(&pk, &step_circuits, &core_circuit, false)
      .expect("prep_prove failed");
  let prep_ms = t0.elapsed().as_millis();
  info!(elapsed_ms = prep_ms as u64, "prep_prove");

  // PROVE (is_small = false)
  let t0 = Instant::now();
  let proof =
    NeutronNovaZkSNARK::<E>::prove(&pk, &step_circuits, &core_circuit, &prep_snark, false)
      .expect("prove failed");
  let prove_ms = t0.elapsed().as_millis();
  info!(elapsed_ms = prove_ms as u64, "prove");

  // VERIFY
  let t0 = Instant::now();
  proof.verify(&vk, num_instances).expect("verify failed");
  let verify_ms = t0.elapsed().as_millis();
  info!(elapsed_ms = verify_ms as u64, "verify");

  // Snapshot and print timing table
  let timings = snapshot_timings(&timing_data, NEUTRONNOVA_ZK_PROVE_PHASES);
  let header = format!(
    "===== NeutronNova: instances={}, chain={}, setup={}ms, prep={}ms, prove={}ms, verify={}ms =====",
    num_instances, chain_length, setup_ms, prep_ms, prove_ms, verify_ms
  );
  print_single_table(&header, NEUTRONNOVA_ZK_PROVE_PHASES, &timings);
}
