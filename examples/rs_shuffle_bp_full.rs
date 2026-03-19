//! Full RS Shuffle with Re-encryption (Bellpepper + Spartan)
//!
//! N=52 cards, LEVELS=6, with parallel re-encryption.
//! Uses the Pallas/Vesta cycle:
//!   - Spartan engine: PallasHyraxEngine (proof system)
//!   - EC engine: VestaHyraxEngine (in-circuit EC ops on Vesta curve)
//!   - Circuit field: pallas::Scalar = vesta::Base
//!
//! Run with:
//!   cargo run --release --example rs_shuffle_bp_full

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use ff::Field;
use std::time::Instant;
use tracing_subscriber::prelude::*;

use spartan2::{
  gadgets::ecc::AllocatedPoint,
  provider::{pasta::pallas, PallasHyraxEngine, VestaHyraxEngine},
  rs_shuffle_bp::{
    data_structures::{
      ElGamalCiphertext, ElGamalCiphertextVar, PermutationWitnessTrace,
      PermutationWitnessTraceVar,
    },
    encryption::{
      native_reencrypt_parallel, reencrypt_deck_bp,
      NativeReencryptionData,
    },
    native::run_rs_shuffle_permutation,
    permutation::{check_grand_product, IndexPositionPair},
  },
  spartan::SpartanSNARK,
  traits::{
    circuit::SpartanCircuit,
    snark::R1CSSNARKTrait,
    Engine, Group,
  },
};

const N: usize = 52;
const LEVELS: usize = 6;

/// Circuit field = pallas::Scalar = vesta::Base
type Scalar = pallas::Scalar;

/// EC engine for in-circuit operations (Vesta curve, coords in pallas::Scalar)
type ECEngine = VestaHyraxEngine;

// ============================================================================
// Helper: find a valid curve point
// ============================================================================

fn find_point_on_vesta(start_x: Scalar) -> (Scalar, Scalar) {
  let (a, b, _, _) = <ECEngine as Engine>::GE::group_params();
  let mut x = start_x;
  loop {
    let rhs = x.cube() + a * x + b;
    if let Some(y) = Option::from(rhs.sqrt()) {
      return (x, y);
    }
    x += Scalar::ONE;
  }
}

// ============================================================================
// Circuit definition
// ============================================================================

#[derive(Clone)]
struct RSShuffleReencryptCircuit {
  witness_trace: PermutationWitnessTrace<N, LEVELS>,

  /// Input ciphertexts (before shuffle+re-encrypt)
  input_ciphertexts: [ElGamalCiphertext<ECEngine>; N],
  /// Randomization scalars
  randomizations: [Scalar; N],
  /// Permutation (maps input position -> output position)
  permutation: [usize; N],

  /// Public key coords
  pk_coords: (Scalar, Scalar),
  /// Generator coords
  gen_coords: (Scalar, Scalar),

  /// Pre-computed native re-encryption data (for future parallel witness path)
  _native_reencrypt_data: NativeReencryptionData<ECEngine, N>,
}

impl SpartanCircuit<PallasHyraxEngine> for RSShuffleReencryptCircuit {
  fn public_values(&self) -> Result<Vec<Scalar>, SynthesisError> {
    Ok(vec![])
  }

  fn shared<CS: ConstraintSystem<Scalar>>(
    &self,
    _cs: &mut CS,
  ) -> Result<Vec<AllocatedNum<Scalar>>, SynthesisError> {
    Ok(vec![])
  }

  fn precommitted<CS: ConstraintSystem<Scalar>>(
    &self,
    cs: &mut CS,
    _shared: &[AllocatedNum<Scalar>],
  ) -> Result<Vec<AllocatedNum<Scalar>>, SynthesisError> {
    // Allocate witness trace as precommitted
    let _witness_var =
      PermutationWitnessTraceVar::<Scalar, N, LEVELS>::alloc(cs.namespace(|| "witness"), &self.witness_trace)?;
    Ok(vec![])
  }

  fn num_challenges(&self) -> usize {
    // 2 for permutation grand product + 5 for indexed ciphertext grand product
    2
  }

  fn synthesize<CS: ConstraintSystem<Scalar>>(
    &self,
    cs: &mut CS,
    _shared: &[AllocatedNum<Scalar>],
    _precommitted: &[AllocatedNum<Scalar>],
    challenges: Option<&[Scalar]>,
  ) -> Result<(), SynthesisError> {
    let (alpha_val, beta_val) = match challenges {
      Some(c) => (c[0], c[1]),
      None => (Scalar::from(0u64), Scalar::from(0u64)),
    };

    // Allocate challenges as public inputs
    let alpha_var =
      AllocatedNum::alloc_input(cs.namespace(|| "challenge_alpha"), || Ok(alpha_val))?;
    let beta_var =
      AllocatedNum::alloc_input(cs.namespace(|| "challenge_beta"), || Ok(beta_val))?;

    // Re-allocate witness trace for synthesis ("rest" variables)
    let witness_var = PermutationWitnessTraceVar::<Scalar, N, LEVELS>::alloc(
      cs.namespace(|| "witness_synth"),
      &self.witness_trace,
    )?;

    // =========================================================================
    // Part 1: Permutation checks (grand product for each level)
    // =========================================================================
    for level in 0..LEVELS {
      let unsorted_pairs: Vec<IndexPositionPair<Scalar>> = witness_var.uns_levels[level]
        .iter()
        .map(|u| IndexPositionPair::new(u.idx.clone(), u.next_pos.clone()))
        .collect();

      let mut sorted_pairs: Vec<IndexPositionPair<Scalar>> = Vec::with_capacity(N);
      for i in 0..N {
        let pos =
          AllocatedNum::alloc(cs.namespace(|| format!("sorted_pos_{}_{}", level, i)), || {
            Ok(Scalar::from(i as u64))
          })?;
        sorted_pairs.push(IndexPositionPair::new(
          witness_var.sorted_levels[level][i].idx.clone(),
          pos,
        ));
      }

      check_grand_product::<Scalar, _, _, 2>(
        cs.namespace(|| format!("grand_product_level_{}", level)),
        &unsorted_pairs,
        &sorted_pairs,
        &[alpha_var.clone(), beta_var.clone()],
      )?;
    }

    // =========================================================================
    // Part 2: Re-encryption
    // =========================================================================

    // Allocate input ciphertexts (shuffled order)
    let mut shuffled_deck_vars = Vec::with_capacity(N);
    for i in 0..N {
      // Apply permutation: shuffled_deck[i] = input_ciphertexts[permutation[i]]
      let src = self.permutation[i];
      let ct_var = ElGamalCiphertextVar::<ECEngine>::alloc(
        cs.namespace(|| format!("shuffled_ct_{}", i)),
        &self.input_ciphertexts[src],
      )?;
      shuffled_deck_vars.push(ct_var);
    }
    let shuffled_deck: [ElGamalCiphertextVar<ECEngine>; N] =
      shuffled_deck_vars.try_into().ok().unwrap();

    // Allocate randomization scalars
    let mut rand_vars = Vec::with_capacity(N);
    for i in 0..N {
      let r_var = AllocatedNum::alloc(cs.namespace(|| format!("rand_{}", i)), || {
        Ok(self.randomizations[i])
      })?;
      rand_vars.push(r_var);
    }
    let rand_arr: [AllocatedNum<Scalar>; N] = rand_vars.try_into().ok().unwrap();

    // Allocate public key
    let pk_var = AllocatedPoint::<ECEngine>::alloc(
      cs.namespace(|| "pk"),
      Some((self.pk_coords.0, self.pk_coords.1, false)),
    )?;

    // Re-encrypt the deck
    let _output_deck = reencrypt_deck_bp::<ECEngine, _, N>(
      cs,
      &shuffled_deck,
      &rand_arr,
      &pk_var,
      &self._native_reencrypt_data,
      self.gen_coords,
    )?;

    Ok(())
  }
}

// ============================================================================
// Main
// ============================================================================

fn main() {
  println!("================================================================");
  println!("  RS Shuffle + Re-encryption (Bellpepper/Spartan)");
  println!("  N = {} cards, LEVELS = {}", N, LEVELS);
  println!("  Curve: Pallas/Vesta cycle");
  println!("================================================================\n");

  // =========================================================================
  // Step 1: Native shuffle
  // =========================================================================
  println!("--- Native Shuffle ---");
  let shuffle_start = Instant::now();

  let seed = Scalar::from(42u64);
  let input: [usize; N] = std::array::from_fn(|i| i);
  let trace = run_rs_shuffle_permutation::<Scalar, _, N, LEVELS>(seed, &input);

  let shuffle_time = shuffle_start.elapsed();
  println!("  Shuffle time: {:?}", shuffle_time);
  println!("  Output[0..8]: {:?}", &trace.permuted_output[..8]);

  // Extract permutation
  let permutation = trace.extract_permutation_array();

  // =========================================================================
  // Step 2: Generate ciphertexts and re-encrypt
  // =========================================================================
  println!("\n--- Native Re-encryption ---");

  let gen_coords = find_point_on_vesta(Scalar::ONE);
  let pk_coords = find_point_on_vesta(Scalar::from(100u64));

  // Create N distinct input ciphertexts (valid Vesta curve points)
  let mut input_cts = Vec::with_capacity(N);
  for i in 0..N {
    let (c1_x, c1_y) = find_point_on_vesta(Scalar::from((i * 2 + 1) as u64));
    let (c2_x, c2_y) = find_point_on_vesta(Scalar::from((i * 2 + 200) as u64));
    input_cts.push(ElGamalCiphertext::<ECEngine>::new(c1_x, c1_y, c2_x, c2_y));
  }
  let input_ct_arr: [ElGamalCiphertext<ECEngine>; N] = input_cts.try_into().ok().unwrap();

  // Apply permutation to get shuffled ciphertexts
  let mut shuffled_cts: Vec<ElGamalCiphertext<ECEngine>> = Vec::with_capacity(N);
  for i in 0..N {
    shuffled_cts.push(input_ct_arr[permutation[i]].clone());
  }
  let shuffled_ct_arr: [ElGamalCiphertext<ECEngine>; N] =
    shuffled_cts.try_into().ok().unwrap();

  // Randomization scalars
  let mut randomizations = Vec::with_capacity(N);
  for i in 0..N {
    randomizations.push(Scalar::from((i + 10) as u64));
  }
  let rand_arr: [Scalar; N] = randomizations.try_into().ok().unwrap();

  // Parallel native re-encryption
  let reencrypt_start = Instant::now();
  let native_data = native_reencrypt_parallel::<ECEngine, N>(
    &shuffled_ct_arr,
    &rand_arr,
    pk_coords,
    gen_coords,
  );
  let reencrypt_time = reencrypt_start.elapsed();
  println!("  Re-encryption time: {:?} (parallel)", reencrypt_time);

  // =========================================================================
  // Step 3: Build circuit
  // =========================================================================
  println!("\n--- Circuit Construction ---");

  let circuit = RSShuffleReencryptCircuit {
    witness_trace: trace.witness_trace.clone(),
    input_ciphertexts: input_ct_arr.clone(),
    randomizations: rand_arr,
    permutation,
    pk_coords,
    gen_coords,
    _native_reencrypt_data: native_data,
  };

  // =========================================================================
  // Setup tracing for phase-level timing
  // =========================================================================
  use spartan2::timing::{clear_timings, snapshot_timings, TimingLayer, SPARTAN_PHASES};

  let (timing_layer, timing_data, _constraints) = TimingLayer::new();
  let subscriber = tracing_subscriber::registry().with(timing_layer);
  let _guard = tracing::subscriber::set_default(subscriber);

  // =========================================================================
  // Step 4: Spartan Setup
  // =========================================================================
  println!("\n--- Spartan Setup ---");
  let setup_start = Instant::now();
  let (pk, vk) =
    SpartanSNARK::<PallasHyraxEngine>::setup(circuit.clone()).expect("Setup failed");
  let setup_time = setup_start.elapsed();
  println!("  Setup time: {:?}", setup_time);

  let sizes = pk.sizes();
  println!("  Constraints (unpadded): {}", sizes[0]);
  println!("  Constraints (padded):   {}", sizes[4]);
  println!("  Variables (shared):     {}", sizes[5]);
  println!("  Variables (precommit):  {}", sizes[6]);
  println!("  Variables (rest):       {}", sizes[7]);

  // =========================================================================
  // Step 5: Prep Prove
  // =========================================================================
  println!("\n--- Prep Prove ---");
  clear_timings(&timing_data);
  let prep_start = Instant::now();
  let prep = SpartanSNARK::<PallasHyraxEngine>::prep_prove(&pk, circuit.clone(), false)
    .expect("Prep prove failed");
  let prep_time = prep_start.elapsed();
  let prep_timings = snapshot_timings(&timing_data, SPARTAN_PHASES);
  println!("  Prep time: {:?}", prep_time);

  // =========================================================================
  // Step 6: Prove
  // =========================================================================
  println!("\n--- Prove ---");
  clear_timings(&timing_data);
  let prove_start = Instant::now();
  let snark = SpartanSNARK::<PallasHyraxEngine>::prove(&pk, circuit, &prep, false)
    .expect("Proof generation failed");
  let prove_time = prove_start.elapsed();
  let prove_timings = snapshot_timings(&timing_data, SPARTAN_PHASES);
  println!("  Prove time: {:?}", prove_time);

  // =========================================================================
  // Step 7: Verify
  // =========================================================================
  println!("\n--- Verify ---");
  let verify_start = Instant::now();
  let result = snark.verify(&vk);
  let verify_time = verify_start.elapsed();

  match result {
    Ok(outputs) => {
      println!("  Proof verified successfully!");
      println!("  Verify time: {:?}", verify_time);
      println!("  Public outputs: {} values", outputs.len());
    }
    Err(e) => {
      println!("  Verification FAILED: {:?}", e);
      std::process::exit(1);
    }
  }

  // =========================================================================
  // Timing Breakdown
  // =========================================================================
  println!("\n================================================================");
  println!("           TIMING BREAKDOWN (N={}, LEVELS={})", N, LEVELS);
  println!("================================================================");
  println!("  SETUP (one-time):               {:>10.2?}", setup_time);
  println!("  Constraints:                    {:>10}", sizes[0]);
  println!("----------------------------------------------------------------");
  println!("  PREP:                           {:>10.2?}", prep_time);
  for (name, ms) in &prep_timings {
    if *ms > 0 {
      println!("    {:<32} {:>7}ms", name, ms);
    }
  }
  println!("----------------------------------------------------------------");
  println!("  PROVE:                          {:>10.2?}", prove_time);
  for (name, ms) in &prove_timings {
    if *ms > 0 {
      println!("    {:<32} {:>7}ms", name, ms);
    }
  }
  println!("----------------------------------------------------------------");
  println!("  VERIFY:                         {:>10.2?}", verify_time);
  println!("----------------------------------------------------------------");
  println!("  TOTAL PROVE (prep+prove):       {:>10.2?}", prep_time + prove_time);
  println!("  Native re-encrypt (parallel):   {:>10.2?}", reencrypt_time);
  println!("================================================================");
}
