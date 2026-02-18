//! Simple RS Shuffle Bellpepper Example
//!
//! This example demonstrates the RS shuffle using bellpepper R1CS primitives
//! with Spartan proof generation and verification.
//!
//! Run with:
//!   cargo run --release --example rs_shuffle_bp_simple

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use std::time::Instant;

use spartan2::{
  provider::{pasta::pallas, PallasHyraxEngine},
  rs_shuffle_bp::{
    data_structures::PermutationWitnessTraceVar,
    native::run_rs_shuffle_permutation,
    permutation::{check_grand_product, IndexPositionPair},
  },
  spartan::SpartanSNARK,
  traits::{circuit::SpartanCircuit, snark::R1CSSNARKTrait},
};

/// Small deck size for testing
const N: usize = 8;
/// Number of shuffle levels
const LEVELS: usize = 3;

type Scalar = pallas::Scalar;

/// RS Shuffle Verification Circuit
///
/// This circuit verifies that a permutation witness correctly represents
/// a valid RS shuffle by checking:
/// 1. Grand product equality for multiset membership
#[derive(Clone)]
struct RSShuffleCircuit {
  /// Permutation witness trace
  witness_trace: spartan2::rs_shuffle_bp::data_structures::PermutationWitnessTrace<N, LEVELS>,
}

impl RSShuffleCircuit {
  fn new(
    witness_trace: spartan2::rs_shuffle_bp::data_structures::PermutationWitnessTrace<N, LEVELS>,
  ) -> Self {
    Self { witness_trace }
  }
}

impl SpartanCircuit<PallasHyraxEngine> for RSShuffleCircuit {
  fn public_values(&self) -> Result<Vec<Scalar>, SynthesisError> {
    // No additional public values beyond challenges
    Ok(vec![])
  }

  fn shared<CS: ConstraintSystem<Scalar>>(
    &self,
    _cs: &mut CS,
  ) -> Result<Vec<AllocatedNum<Scalar>>, SynthesisError> {
    // No shared variables
    Ok(vec![])
  }

  fn precommitted<CS: ConstraintSystem<Scalar>>(
    &self,
    cs: &mut CS,
    _shared: &[AllocatedNum<Scalar>],
  ) -> Result<Vec<AllocatedNum<Scalar>>, SynthesisError> {
    // Allocate the witness trace as precommitted variables
    let _witness_var = PermutationWitnessTraceVar::<Scalar, N, LEVELS>::alloc(
      cs.namespace(|| "witness"),
      &self.witness_trace,
    )?;

    Ok(vec![])
  }

  fn num_challenges(&self) -> usize {
    // We need 2 challenges: alpha and beta for the grand product check
    2
  }

  fn synthesize<CS: ConstraintSystem<Scalar>>(
    &self,
    cs: &mut CS,
    _shared: &[AllocatedNum<Scalar>],
    _precommitted: &[AllocatedNum<Scalar>],
    challenges: Option<&[Scalar]>,
  ) -> Result<(), SynthesisError> {
    // Challenges are provided by the Fiat-Shamir transcript
    // During shape generation, they're None - we use placeholder values
    // During witness generation, they contain the actual challenge values
    let (alpha_val, beta_val) = match challenges {
      Some(c) => (c[0], c[1]),
      None => (Scalar::from(0u64), Scalar::from(0u64)), // Placeholder for shape
    };

    // Allocate challenges as public inputs FIRST (required by Spartan protocol)
    let alpha_var = AllocatedNum::alloc_input(cs.namespace(|| "challenge_alpha"), || Ok(alpha_val))?;
    let beta_var = AllocatedNum::alloc_input(cs.namespace(|| "challenge_beta"), || Ok(beta_val))?;

    // Re-allocate the witness trace for synthesis (these become "rest" variables)
    let witness_var = PermutationWitnessTraceVar::<Scalar, N, LEVELS>::alloc(
      cs.namespace(|| "witness_synth"),
      &self.witness_trace,
    )?;

    // For each level, verify the grand product permutation check
    // This ensures that the mapping from unsorted to sorted is a valid permutation
    for level in 0..LEVELS {
      // Create (idx, next_pos) pairs from unsorted rows
      let unsorted_pairs: Vec<IndexPositionPair<Scalar>> = witness_var.uns_levels[level]
        .iter()
        .map(|u| IndexPositionPair::new(u.idx.clone(), u.next_pos.clone()))
        .collect();

      // Create (idx, pos) pairs from sorted rows
      // Position is just the array index for sorted rows
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

      // Grand product check: unsorted pairs should be a permutation of sorted pairs
      check_grand_product::<Scalar, _, _, 2>(
        cs.namespace(|| format!("grand_product_level_{}", level)),
        &unsorted_pairs,
        &sorted_pairs,
        &[alpha_var.clone(), beta_var.clone()],
      )?;
    }

    Ok(())
  }
}

fn main() {
  println!("╔══════════════════════════════════════════════════════════════╗");
  println!("║    RS Shuffle Bellpepper Simple Example                      ║");
  println!("╠══════════════════════════════════════════════════════════════╣");
  println!(
    "║  N = {} cards, LEVELS = {}                                      ║",
    N, LEVELS
  );
  println!("╚══════════════════════════════════════════════════════════════╝\n");

  // Generate random seed and run native shuffle
  let seed = Scalar::from(42u64);
  println!("Running native RS shuffle...");
  let input: [usize; N] = std::array::from_fn(|i| i);
  let trace = run_rs_shuffle_permutation::<Scalar, _, N, LEVELS>(seed, &input);

  println!("Input:  {:?}", input);
  println!("Output: {:?}", trace.permuted_output);
  println!("Permutation is valid: {:?}", {
    let mut sorted = trace.permuted_output.to_vec();
    sorted.sort();
    sorted == input.to_vec()
  });

  // Create circuit
  let circuit = RSShuffleCircuit::new(trace.witness_trace.clone());

  // =========================================================================
  // Setup
  // =========================================================================
  println!("\n--- Setup ---");
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
  // Prep Prove
  // =========================================================================
  println!("\n--- Prep Prove ---");
  let prep_start = Instant::now();
  let prep = SpartanSNARK::<PallasHyraxEngine>::prep_prove(&pk, circuit.clone(), false)
    .expect("Prep prove failed");
  let prep_time = prep_start.elapsed();
  println!("  Prep time: {:?}", prep_time);

  // =========================================================================
  // Prove
  // =========================================================================
  println!("\n--- Prove ---");
  let prove_start = Instant::now();
  let snark = SpartanSNARK::<PallasHyraxEngine>::prove(&pk, circuit, &prep, false)
    .expect("Proof generation failed");
  let prove_time = prove_start.elapsed();
  println!("  Prove time: {:?}", prove_time);

  // =========================================================================
  // Verify
  // =========================================================================
  println!("\n--- Verify ---");
  let verify_start = Instant::now();
  let result = snark.verify(&vk);
  let verify_time = verify_start.elapsed();

  match result {
    Ok(public_outputs) => {
      println!("  ✓ Proof verified successfully!");
      println!("  Verify time: {:?}", verify_time);
      println!("  Public outputs: {} values", public_outputs.len());
    }
    Err(e) => {
      println!("  ✗ Verification failed: {:?}", e);
      std::process::exit(1);
    }
  }

  // =========================================================================
  // Summary
  // =========================================================================
  println!("\n╔══════════════════════════════════════════════════════════════╗");
  println!("║                        SUMMARY                               ║");
  println!("╠══════════════════════════════════════════════════════════════╣");
  println!(
    "║  Constraints:    {:>10}                               ║",
    sizes[0]
  );
  println!(
    "║  Setup time:     {:>10.2?}                             ║",
    setup_time
  );
  println!(
    "║  Prove time:     {:>10.2?}                             ║",
    prep_time + prove_time
  );
  println!(
    "║  Verify time:    {:>10.2?}                             ║",
    verify_time
  );
  println!("╚══════════════════════════════════════════════════════════════╝");
}
