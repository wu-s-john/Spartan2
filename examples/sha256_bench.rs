// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT

//! examples/sha256_bench.rs
//! Benchmark Spartan SHA-256 proving with timing table output.
//!
//! Run with:
//!   RUST_LOG=info cargo run --release --example sha256_bench
//!   RUST_LOG=info cargo run --release --example sha256_bench -- --msg-size 2048

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
  provider::Bn254Engine,
  spartan::SpartanSNARK,
  timing::{SPARTAN_PHASES, TimingLayer, clear_timings, print_single_table, snapshot_timings},
  traits::{Engine, circuit::SpartanCircuit, snark::R1CSSNARKTrait},
};
use std::{marker::PhantomData, time::Instant};
use tracing::info;
use tracing_subscriber::{EnvFilter, Layer as _, layer::SubscriberExt, util::SubscriberInitExt};

type E = Bn254Engine;

#[derive(Parser)]
#[command(about = "Benchmark Spartan SHA-256 proving")]
struct Args {
  /// Message size in bytes
  #[arg(short = 'm', long, default_value = "1024")]
  msg_size: usize,
}

#[derive(Clone, Debug)]
struct Sha256Circuit<Scalar: PrimeField> {
  preimage: Vec<u8>,
  _p: PhantomData<Scalar>,
}

impl<Scalar: PrimeField + PrimeFieldBits> Sha256Circuit<Scalar> {
  fn new(preimage: Vec<u8>) -> Self {
    Self {
      preimage,
      _p: PhantomData,
    }
  }
}

impl<E: Engine> SpartanCircuit<E> for Sha256Circuit<E::Scalar> {
  fn public_values(&self) -> Result<Vec<<E as Engine>::Scalar>, SynthesisError> {
    let mut hasher = Sha256::new();
    hasher.update(&self.preimage);
    let hash = hasher.finalize();
    let hash_scalars: Vec<<E as Engine>::Scalar> = hash
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
    let bit_values: Vec<_> = self
      .preimage
      .clone()
      .into_iter()
      .flat_map(|byte| (0..8).rev().map(move |i| (byte >> i) & 1 == 1))
      .map(Some)
      .collect();

    let preimage_bits = bit_values
      .into_iter()
      .enumerate()
      .map(|(i, b)| AllocatedBit::alloc(cs.namespace(|| format!("preimage bit {i}")), b))
      .map(|b| b.map(Boolean::from))
      .collect::<Result<Vec<_>, _>>()?;

    let hash_bits = sha256(cs.namespace(|| "sha256"), &preimage_bits)?;

    // Sanity-check against Rust SHA-256
    let mut hasher = Sha256::new();
    hasher.update(&self.preimage);
    let expected = hasher.finalize();
    let mut expected_bits = expected
      .iter()
      .flat_map(|&byte| (0..8).rev().map(move |i| (byte >> i) & 1 == 1));

    for b in &hash_bits {
      match b {
        Boolean::Is(bit) => assert_eq!(expected_bits.next().unwrap(), bit.get_value().unwrap()),
        Boolean::Not(bit) => assert_ne!(expected_bits.next().unwrap(), bit.get_value().unwrap()),
        Boolean::Constant(_) => unreachable!(),
      }
    }

    for (i, bit) in hash_bits.iter().enumerate() {
      let n = AllocatedNum::alloc_input(cs.namespace(|| format!("public num {i}")), || {
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

  let msg_size = args.msg_size;
  let circuit = Sha256Circuit::<<E as Engine>::Scalar>::new(vec![0u8; msg_size]);

    eprintln!("\n======= SHA256: msg={}B =======", msg_size);

    // SETUP
    let t0 = Instant::now();
    let (pk, vk) = SpartanSNARK::<E>::setup(circuit.clone()).expect("setup failed");
    let setup_ms = t0.elapsed().as_millis();
    info!(elapsed_ms = setup_ms as u64, "setup");

    // Clear timing data before prove phases
    clear_timings(&timing_data);

    // PREP_PROVE (is_small = false)
    let t0 = Instant::now();
    let prep_snark =
      SpartanSNARK::<E>::prep_prove(&pk, circuit.clone(), false).expect("prep_prove failed");
    let prep_ms = t0.elapsed().as_millis();
    info!(elapsed_ms = prep_ms as u64, "prep_prove");

    // PROVE (is_small = false)
    let t0 = Instant::now();
    let proof =
      SpartanSNARK::<E>::prove(&pk, circuit.clone(), &prep_snark, false).expect("prove failed");
    let prove_ms = t0.elapsed().as_millis();
    info!(elapsed_ms = prove_ms as u64, "prove");

    // VERIFY
    let t0 = Instant::now();
    proof.verify(&vk).expect("verify failed");
    let verify_ms = t0.elapsed().as_millis();
    info!(elapsed_ms = verify_ms as u64, "verify");

    // Snapshot and print timing table
    let timings = snapshot_timings(&timing_data, SPARTAN_PHASES);
    let header = format!(
      "===== SHA256: msg={}B, setup={}ms, prep={}ms, prove={}ms, verify={}ms =====",
      msg_size, setup_ms, prep_ms, prove_ms, verify_ms
    );
    print_single_table(&header, SPARTAN_PHASES, &timings);
}
