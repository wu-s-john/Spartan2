//! SHA-256 Spartan: small-value 4×64 vs small-value 5×52
//!
//! Both methods use identical small-value outer sumcheck.
//! Only difference: inner sumcheck uses prove_quad (4×64) vs prove_quad_52 (5×52).
//!
//! Run: `cargo run --release --example sha256_inner_sumcheck_bench`
//! Options: `-- --bytes 2048 --iters 10`

#[cfg(feature = "jem")]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

use spartan2::{
  provider::Bn254Engine,
  sha256_circuits::SmallSha256Circuit,
  spartan::SpartanSNARK,
  timing::{SPARTAN_PHASES, TimingLayer, clear_timings, snapshot_timings},
  traits::{Engine, snark::R1CSSNARKTrait},
};
use std::{collections::HashMap, time::Instant};
use tracing::info_span;
use tracing_subscriber::{EnvFilter, Layer as _, layer::SubscriberExt, util::SubscriberInitExt};

type E = Bn254Engine;
type F = <E as Engine>::Scalar;

fn main() {
  let (timing_layer, timing_data, _) = TimingLayer::new();

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
  let mut msg_len = 1024usize;
  let mut iters = 7usize;
  let mut i = 1;
  while i < args.len() {
    match args[i].as_str() {
      "--bytes" => { msg_len = args[i + 1].parse().unwrap(); i += 2; }
      "--iters" => { iters = args[i + 1].parse().unwrap(); i += 2; }
      _ => { i += 1; }
    }
  }

  let circuit = SmallSha256Circuit::<F>::new(vec![0u8; msg_len], true);

  eprintln!("=== Small-value 4x64 vs 5x52 (msg={}B, {} iters) ===\n", msg_len, iters);

  let t0 = Instant::now();
  let (pk, vk) = SpartanSNARK::<E>::setup(circuit.clone()).expect("setup");
  eprintln!("Setup: {} ms", t0.elapsed().as_millis());

  let t0 = Instant::now();
  let prep = SpartanSNARK::<E>::prep_prove(&pk, circuit.clone(), true).expect("prep");
  eprintln!("Prep:  {} ms\n", t0.elapsed().as_millis());

  // Warmup
  eprintln!("Warming up...");
  for _ in 0..2 {
    let _ = SpartanSNARK::<E>::prove(&pk, circuit.clone(), &prep, true).unwrap();
    let _ = SpartanSNARK::<E>::prove_small_52::<_, 3>(&pk, circuit.clone(), &prep).unwrap();
  }
  eprintln!("Done.\n");

  let methods: Vec<(&str, Box<dyn Fn() -> SpartanSNARK<E>>)> = vec![
    ("small-value 4x64", Box::new(|| {
      SpartanSNARK::<E>::prove(&pk, circuit.clone(), &prep, true).unwrap()
    })),
    ("small-value 5x52", Box::new(|| {
      SpartanSNARK::<E>::prove_small_52::<_, 3>(&pk, circuit.clone(), &prep).unwrap()
    })),
  ];

  let phases: &[&str] = &[
    "mat_vec", "outer_sc", "eval_sparse", "poly_ABC", "inner_sc", "pcs", "prove_total",
  ];

  struct Result {
    name: &'static str,
    medians: HashMap<&'static str, u64>,
    prove_median: u128,
    ok: bool,
  }

  let mut results: Vec<Result> = Vec::new();

  for (name, run) in &methods {
    let mut samples: HashMap<&str, Vec<u64>> = phases.iter().map(|p| (*p, Vec::new())).collect();
    let mut prove_times: Vec<u128> = Vec::new();
    let mut ok = false;

    for iter in 0..iters {
      let _span = info_span!("bench", name = *name, iter).entered();
      clear_timings(&timing_data);

      let t0 = Instant::now();
      let proof = run();
      prove_times.push(t0.elapsed().as_millis());

      let timings = snapshot_timings(&timing_data, SPARTAN_PHASES);
      for p in phases {
        samples.get_mut(p).unwrap().push(timings.get(p).copied().unwrap_or(0));
      }
      if !ok { ok = proof.verify(&vk).is_ok(); }
    }

    let mut medians = HashMap::new();
    for p in phases {
      let s = samples.get_mut(p).unwrap();
      s.sort();
      medians.insert(*p, s[s.len() / 2]);
    }
    prove_times.sort();

    results.push(Result { name, medians, prove_median: prove_times[prove_times.len() / 2], ok });
  }

  // Table
  eprintln!("=== Results (median of {} runs) ===\n", iters);

  eprint!("{:<22}", "Method");
  for p in phases { eprint!("{:>12}", p); }
  eprintln!("{:>10}", "verified");
  eprintln!("{}", "-".repeat(22 + phases.len() * 12 + 10));

  for r in &results {
    eprint!("{:<22}", r.name);
    for p in phases {
      eprint!("{:>10} ms", r.medians.get(p).copied().unwrap_or(0));
    }
    eprintln!("{:>10}", if r.ok { "OK" } else { "FAIL" });
  }

  // Delta row
  let a = &results[0];
  let b = &results[1];
  eprintln!();
  eprint!("{:<22}", "5x52 vs 4x64");
  for p in phases {
    let va = a.medians.get(p).copied().unwrap_or(1).max(1) as f64;
    let vb = b.medians.get(p).copied().unwrap_or(0) as f64;
    let diff_ms = vb - va;
    let pct = (vb / va - 1.0) * 100.0;
    if diff_ms.abs() < 0.5 {
      eprint!("{:>12}", "same");
    } else {
      eprint!("{:>+8.0} ms {:>+.0}%", diff_ms, pct);
    }
  }
  let speedup = a.prove_median as f64 / b.prove_median as f64;
  eprintln!("{:>10}", format!("{:.2}x", speedup));
}
