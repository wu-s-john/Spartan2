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
//!   cargo run --release --example rs_shuffle_bp_full -- --zk

use bellpepper_core::{ConstraintSystem, SynthesisError, num::AllocatedNum};
use clap::Parser;
use ff::{Field, PrimeField};
use std::time::{Duration, Instant};
use tracing_subscriber::prelude::*;

use spartan2::{
  gadgets::ecc::AllocatedPointNonInfinity,
  provider::{PallasHyraxEngine, VestaHyraxEngine, pasta::pallas},
  rs_shuffle_bp::{
    data_structures::{ElGamalCiphertext, ElGamalCiphertextVar, PermutationWitnessTrace},
    encryption::{
      NativeReencryptionData, native_reencrypt_parallel, precompute_native_powers,
      reencrypt_deck_bp,
    },
    native::run_rs_shuffle_permutation,
    permutation::{IndexPositionPair, IndexedCiphertext, check_grand_product},
  },
  spartan::SpartanSNARK,
  spartan_zk::SpartanZkSNARK,
  timing::{
    SPARTAN_PHASES, SPARTAN_ZK_PHASES, TimingData, TimingLayer, clear_timings, snapshot_timings,
  },
  traits::{Engine, Group, circuit::SpartanCircuit, snark::R1CSSNARKTrait},
};

#[derive(Parser)]
#[command(name = "rs_shuffle_bp_full")]
struct Cli {
  /// Use ZK Spartan (zero-knowledge) instead of non-ZK
  #[arg(long)]
  zk: bool,
}

const N: usize = 52;
const LEVELS: usize = 6;

/// Circuit field = pallas::Scalar = vesta::Base
type Scalar = pallas::Scalar;

/// EC engine for in-circuit operations (Vesta curve, coords in pallas::Scalar)
type ECEngine = VestaHyraxEngine;

fn alloc_point_public_input<E: Engine, CS: ConstraintSystem<E::Base>>(
  mut cs: CS,
  coords: (E::Base, E::Base),
) -> Result<AllocatedPointNonInfinity<E>, SynthesisError> {
  Ok(AllocatedPointNonInfinity {
    x: AllocatedNum::alloc_input(cs.namespace(|| "x"), || Ok(coords.0))?,
    y: AllocatedNum::alloc_input(cs.namespace(|| "y"), || Ok(coords.1))?,
  })
}

fn alloc_ciphertext_public_input<E: Engine, CS: ConstraintSystem<E::Base>>(
  mut cs: CS,
  ct: &ElGamalCiphertext<E>,
) -> Result<ElGamalCiphertextVar<E>, SynthesisError> {
  Ok(ElGamalCiphertextVar::new(
    alloc_point_public_input(cs.namespace(|| "c1"), (ct.c1_x, ct.c1_y))?,
    alloc_point_public_input(cs.namespace(|| "c2"), (ct.c2_x, ct.c2_y))?,
  ))
}

#[derive(Clone)]
struct MinimalUnsortedRowVar<F: PrimeField> {
  idx: AllocatedNum<F>,
  next_pos: AllocatedNum<F>,
}

#[derive(Clone)]
struct MinimalSortedRowVar<F: PrimeField> {
  idx: AllocatedNum<F>,
}

#[derive(Clone)]
struct MinimalPermutationWitnessTraceVar<F: PrimeField, const N: usize, const LEVELS: usize> {
  uns_levels: [[MinimalUnsortedRowVar<F>; N]; LEVELS],
  sorted_levels: [[MinimalSortedRowVar<F>; N]; LEVELS],
}

impl<F: PrimeField, const N: usize, const LEVELS: usize>
  MinimalPermutationWitnessTraceVar<F, N, LEVELS>
{
  fn alloc<CS: ConstraintSystem<F>>(
    mut cs: CS,
    witness_data: &PermutationWitnessTrace<N, LEVELS>,
  ) -> Result<Self, SynthesisError> {
    let mut uns_levels_vec: Vec<[MinimalUnsortedRowVar<F>; N]> = Vec::with_capacity(LEVELS);
    for level in 0..LEVELS {
      let mut level_rows: Vec<MinimalUnsortedRowVar<F>> = Vec::with_capacity(N);
      for i in 0..N {
        let row = &witness_data.uns_levels[level][i];
        level_rows.push(MinimalUnsortedRowVar {
          idx: AllocatedNum::alloc(cs.namespace(|| format!("uns_idx_{}_{}", level, i)), || {
            Ok(F::from(row.idx as u64))
          })?,
          next_pos: AllocatedNum::alloc(
            cs.namespace(|| format!("uns_next_pos_{}_{}", level, i)),
            || Ok(F::from(row.next_pos as u64)),
          )?,
        });
      }
      uns_levels_vec.push(
        level_rows
          .try_into()
          .map_err(|_| SynthesisError::Unsatisfiable)?,
      );
    }

    let mut sorted_levels_vec: Vec<[MinimalSortedRowVar<F>; N]> = Vec::with_capacity(LEVELS);
    for level in 0..LEVELS {
      let mut level_rows: Vec<MinimalSortedRowVar<F>> = Vec::with_capacity(N);
      for i in 0..N {
        let row = &witness_data.next_levels[level][i];
        level_rows.push(MinimalSortedRowVar {
          idx: AllocatedNum::alloc(
            cs.namespace(|| format!("sorted_idx_{}_{}", level, i)),
            || Ok(F::from(row.idx as u64)),
          )?,
        });
      }
      sorted_levels_vec.push(
        level_rows
          .try_into()
          .map_err(|_| SynthesisError::Unsatisfiable)?,
      );
    }

    Ok(Self {
      uns_levels: uns_levels_vec
        .try_into()
        .map_err(|_| SynthesisError::Unsatisfiable)?,
      sorted_levels: sorted_levels_vec
        .try_into()
        .map_err(|_| SynthesisError::Unsatisfiable)?,
    })
  }
}

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
  /// Natively precomputed generator power table: powers[i] = 2^i · G
  gen_powers_native: Vec<(Scalar, Scalar)>,
  /// Natively precomputed PK power table: powers[i] = 2^i · PK
  pk_powers_native: Vec<(Scalar, Scalar)>,

  /// Pre-computed native re-encryption data (for future parallel witness path)
  _native_reencrypt_data: NativeReencryptionData<ECEngine, N>,
}

impl SpartanCircuit<PallasHyraxEngine> for RSShuffleReencryptCircuit {
  fn public_values(&self) -> Result<Vec<Scalar>, SynthesisError> {
    let mut vals = Vec::with_capacity(4 + 2 * self.pk_powers_native.len() + 4 * N * 2);

    // Generator coords
    vals.push(self.gen_coords.0);
    vals.push(self.gen_coords.1);

    // Public key coords
    vals.push(self.pk_coords.0);
    vals.push(self.pk_coords.1);

    // PK power table coords (public inputs)
    for &(x, y) in &self.pk_powers_native {
      vals.push(x);
      vals.push(y);
    }

    // Input ciphertexts (original order)
    for ct in &self.input_ciphertexts {
      vals.push(ct.c1_x);
      vals.push(ct.c1_y);
      vals.push(ct.c2_x);
      vals.push(ct.c2_y);
    }

    // Output ciphertexts (re-encrypted)
    for ct in &self._native_reencrypt_data.output_ciphertexts {
      vals.push(ct.c1_x);
      vals.push(ct.c1_y);
      vals.push(ct.c2_x);
      vals.push(ct.c2_y);
    }

    Ok(vals)
  }

  fn shared<CS: ConstraintSystem<Scalar>>(
    &self,
    _cs: &mut CS,
  ) -> Result<Vec<AllocatedNum<Scalar>>, SynthesisError> {
    Ok(vec![])
  }

  fn precommitted<CS: ConstraintSystem<Scalar>>(
    &self,
    _cs: &mut CS,
    _shared: &[AllocatedNum<Scalar>],
  ) -> Result<Vec<AllocatedNum<Scalar>>, SynthesisError> {
    Ok(vec![])
  }

  fn num_challenges(&self) -> usize {
    // 2 for permutation grand product + 5 for indexed ciphertext grand product
    7
  }

  fn synthesize<CS: ConstraintSystem<Scalar>>(
    &self,
    cs: &mut CS,
    _shared: &[AllocatedNum<Scalar>],
    _precommitted: &[AllocatedNum<Scalar>],
    challenges: Option<&[Scalar]>,
  ) -> Result<(), SynthesisError> {
    let challenge_vals: [Scalar; 7] = match challenges {
      Some(c) => std::array::from_fn(|i| c[i]),
      None => [Scalar::from(0u64); 7],
    };

    let witness_var = MinimalPermutationWitnessTraceVar::<Scalar, N, LEVELS>::alloc(
      cs.namespace(|| "witness_synth"),
      &self.witness_trace,
    )?;

    // =========================================================================
    // Allocate variables and register public inputs (inputize before challenges)
    // Order must match public_values(): gen, pk, input deck, output deck
    // =========================================================================

    // Allocate generator from native powers[0] — public input
    let _gen_var = alloc_point_public_input::<ECEngine, _>(
      cs.namespace(|| "generator"),
      self.gen_powers_native[0],
    )?;

    // Allocate public key (non-infinity) — public input
    let _pk_var = alloc_point_public_input::<ECEngine, _>(
      cs.namespace(|| "pk"),
      (self.pk_coords.0, self.pk_coords.1),
    )?;

    // Allocate pk power table — public inputs
    let num_bits = Scalar::NUM_BITS as usize;
    let mut pk_powers_vars = Vec::with_capacity(num_bits);
    for i in 0..num_bits {
      let p = alloc_point_public_input::<ECEngine, _>(
        cs.namespace(|| format!("pk_power_{}", i)),
        self.pk_powers_native[i],
      )?;
      pk_powers_vars.push(p);
    }

    // Allocate input ciphertexts (original order, before shuffle) — public inputs
    let mut input_deck_vars = Vec::with_capacity(N);
    for i in 0..N {
      let ct_var = alloc_ciphertext_public_input::<ECEngine, _>(
        cs.namespace(|| format!("input_ct_{}", i)),
        &self.input_ciphertexts[i],
      )?;
      input_deck_vars.push(ct_var);
    }

    // Allocate shuffled ciphertexts (permuted order)
    let mut shuffled_deck_vars = Vec::with_capacity(N);
    for i in 0..N {
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
    let index_vars: [AllocatedNum<Scalar>; N] = (0..N)
      .map(|i| {
        AllocatedNum::alloc(cs.namespace(|| format!("idx_{}", i)), || {
          Ok(Scalar::from(i as u64))
        })
      })
      .collect::<Result<Vec<_>, _>>()?
      .try_into()
      .map_err(|_| SynthesisError::Unsatisfiable)?;

    // Re-encrypt the deck (includes inputize for output coords)
    reencrypt_deck_bp::<ECEngine, _, N>(
      cs,
      &shuffled_deck,
      &rand_arr,
      &pk_powers_vars,
      &self._native_reencrypt_data,
      &self.gen_powers_native,
    )?;

    // =========================================================================
    // Allocate challenges as public inputs (must come after all inputize calls)
    // =========================================================================
    let alpha_var =
      AllocatedNum::alloc_input(cs.namespace(|| "challenge_alpha"), || Ok(challenge_vals[0]))?;
    let beta_var =
      AllocatedNum::alloc_input(cs.namespace(|| "challenge_beta"), || Ok(challenge_vals[1]))?;
    let mut ict_challenges = Vec::with_capacity(5);
    for i in 0..5 {
      let c = AllocatedNum::alloc_input(cs.namespace(|| format!("challenge_ict_{}", i)), || {
        Ok(challenge_vals[2 + i])
      })?;
      ict_challenges.push(c);
    }
    let ict_challenges_arr: [AllocatedNum<Scalar>; 5] = ict_challenges.try_into().unwrap();

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
        sorted_pairs.push(IndexPositionPair::new(
          witness_var.sorted_levels[level][i].idx.clone(),
          index_vars[i].clone(),
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
    // Part 2: Indexed ciphertext grand product
    // =========================================================================
    // Proves that (i, input_ct[i]) multiset-equals (σ(i), shuffled_ct[i])
    // where σ is the permutation proved by Part 1.

    let left: Vec<IndexedCiphertext<Scalar>> = (0..N)
      .map(|i| {
        Ok(IndexedCiphertext::new::<ECEngine>(
          index_vars[i].clone(),
          &input_deck_vars[i],
        ))
      })
      .collect::<Result<Vec<_>, SynthesisError>>()?;

    let right: Vec<IndexedCiphertext<Scalar>> = (0..N)
      .map(|i| {
        Ok(IndexedCiphertext::new::<ECEngine>(
          witness_var.sorted_levels[LEVELS - 1][i].idx.clone(),
          &shuffled_deck[i],
        ))
      })
      .collect::<Result<Vec<_>, SynthesisError>>()?;

    check_grand_product::<_, _, _, 5>(
      cs.namespace(|| "ict_grand_product"),
      &left,
      &right,
      &ict_challenges_arr,
    )?;

    Ok(())
  }
}

// ============================================================================
// Generic Spartan runner (works for both SpartanSNARK and SpartanZkSNARK)
// ============================================================================

fn run_spartan<S: R1CSSNARKTrait<PallasHyraxEngine>>(
  circuit: RSShuffleReencryptCircuit,
  timing_data: &TimingData,
  phases: &[(&str, &'static str)],
  label: &str,
  reencrypt_time: Duration,
) {
  // Setup
  println!("\n--- {} Setup ---", label);
  let setup_start = Instant::now();
  let (pk, vk) = S::setup(circuit.clone()).expect("Setup failed");
  let setup_time = setup_start.elapsed();
  println!("  Setup time: {:?}", setup_time);

  let sizes = S::pk_sizes(&pk);
  println!("  Constraints (unpadded): {}", sizes[0]);
  println!("  Constraints (padded):   {}", sizes[4]);
  println!(
    "  Variables (shared):     {} (unpadded: {})",
    sizes[5], sizes[1]
  );
  println!(
    "  Variables (precommit):  {} (unpadded: {})",
    sizes[6], sizes[2]
  );
  println!(
    "  Variables (rest):       {} (unpadded: {})",
    sizes[7], sizes[3]
  );

  // Prep Prove
  println!("\n--- Prep Prove ---");
  clear_timings(timing_data);
  let prep_start = Instant::now();
  let prep = S::prep_prove(&pk, circuit.clone(), false).expect("Prep prove failed");
  let prep_time = prep_start.elapsed();
  let prep_timings = snapshot_timings(timing_data, phases);
  println!("  Prep time: {:?}", prep_time);

  // Prove
  println!("\n--- Prove ---");
  clear_timings(timing_data);
  let prove_start = Instant::now();
  let snark = S::prove(&pk, circuit, &prep, false).expect("Proof generation failed");
  let prove_time = prove_start.elapsed();
  let prove_timings = snapshot_timings(timing_data, phases);
  println!("  Prove time: {:?}", prove_time);

  // Verify
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

  // Timing Breakdown
  println!("\n================================================================");
  println!(
    "           {} TIMING BREAKDOWN (N={}, LEVELS={})",
    label.to_uppercase(),
    N,
    LEVELS
  );
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
  println!(
    "  TOTAL PROVE (prep+prove):       {:>10.2?}",
    prep_time + prove_time
  );
  println!(
    "  Native re-encrypt (parallel):   {:>10.2?}",
    reencrypt_time
  );
  println!("================================================================");
}

// ============================================================================
// Main
// ============================================================================

fn main() {
  let cli = Cli::parse();
  let mode_label = if cli.zk { "ZK Spartan" } else { "Spartan" };

  println!("================================================================");
  println!("  RS Shuffle + Re-encryption (Bellpepper/{})", mode_label);
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
  let shuffled_ct_arr: [ElGamalCiphertext<ECEngine>; N] = shuffled_cts.try_into().ok().unwrap();

  // Randomization scalars
  let mut randomizations = Vec::with_capacity(N);
  for i in 0..N {
    randomizations.push(Scalar::from((i + 10) as u64));
  }
  let rand_arr: [Scalar; N] = randomizations.try_into().ok().unwrap();

  // Parallel native re-encryption
  let reencrypt_start = Instant::now();
  let native_data =
    native_reencrypt_parallel::<ECEngine, N>(&shuffled_ct_arr, &rand_arr, pk_coords, gen_coords);
  let reencrypt_time = reencrypt_start.elapsed();
  println!("  Re-encryption time: {:?} (parallel)", reencrypt_time);

  // =========================================================================
  // Step 3: Build circuit
  // =========================================================================
  println!("\n--- Circuit Construction ---");

  let num_bits = Scalar::NUM_BITS as usize;
  let gen_powers_native = precompute_native_powers::<ECEngine>(gen_coords, num_bits);
  let pk_powers_native = precompute_native_powers::<ECEngine>(pk_coords, num_bits);

  let circuit = RSShuffleReencryptCircuit {
    witness_trace: trace.witness_trace.clone(),
    input_ciphertexts: input_ct_arr.clone(),
    randomizations: rand_arr,
    permutation,
    pk_coords,
    gen_coords,
    gen_powers_native,
    pk_powers_native,
    _native_reencrypt_data: native_data,
  };

  // =========================================================================
  // Setup tracing for phase-level timing
  // =========================================================================
  let (timing_layer, timing_data, _constraints) = TimingLayer::new();
  let subscriber = tracing_subscriber::registry().with(timing_layer);
  let _guard = tracing::subscriber::set_default(subscriber);

  // =========================================================================
  // Step 4: Run Spartan (ZK or non-ZK)
  // =========================================================================
  if cli.zk {
    run_spartan::<SpartanZkSNARK<PallasHyraxEngine>>(
      circuit,
      &timing_data,
      SPARTAN_ZK_PHASES,
      "ZK Spartan",
      reencrypt_time,
    );
  } else {
    run_spartan::<SpartanSNARK<PallasHyraxEngine>>(
      circuit,
      &timing_data,
      SPARTAN_PHASES,
      "Spartan",
      reencrypt_time,
    );
  }
}
