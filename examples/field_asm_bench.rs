//! Benchmark harness for field arithmetic kernel experiments.
//!
//! Compares 4x64 (current/fused/asm), 5x52 (lazy carry), and 8x32 (NEON)
//! representations across all MAC types and ILP batching levels.
//!
//! Usage:
//!   cargo run --release --example field_asm_bench -- all
//!   cargo run --release --example field_asm_bench -- case ff
//!   cargo run --release --example field_asm_bench -- case fi64
//!   cargo run --release --example field_asm_bench -- case fi128
//!   cargo run --release --example field_asm_bench -- case fi32
//!   cargo run --release --example field_asm_bench -- case ilp
//!   cargo run --release --example field_asm_bench -- case reduce
//!   cargo run --release --example field_asm_bench -- case batch
//!   cargo run --release --example field_asm_bench -- profile <variant>

use clap::{Parser, Subcommand};
use ff::Field;
use halo2curves::bn256::Fr as Bn254Fr;
use spartan2::small_field::{
  aarch64::{mac_4x1_into_asm, mac_4x2_into_asm, mac_4x4_into_asm, mac_4x4_into_fused},
  barrett::{barrett_reduce_6, barrett_reduce_7},
  limbs::{mac, mul_4_by_4},
  limbs32::{
    carry_propagate_32_to_6limb, carry_propagate_32_to_7limb, carry_propagate_32_to_9limb,
    mac_ff_32, mac_ff_neon, mac_fi128_32, mac_fi64_32, to_32,
  },
  limbs52::{
    carry_propagate_52_to_6limb, carry_propagate_52_to_7limb, carry_propagate_52_to_9limb,
    mac_ff_52, mac_ff_52_locals, mac_fi128_52, mac_fi64_52, mul_52, to_52,
  },
  montgomery::{montgomery_reduce_9, MontgomeryLimbs},
};
use std::hint::black_box;
use std::time::{Duration, Instant};

#[derive(Parser)]
#[command(name = "field_asm_bench")]
struct Cli {
  #[command(subcommand)]
  command: Commands,
}

#[derive(Subcommand)]
enum Commands {
  /// Run all benchmark cases
  All,
  /// Run a specific benchmark case
  Case { name: String },
  /// Run a single variant in a long loop (for profiling)
  Profile { variant: String },
}

// ============================================================================
// Timing utilities
// ============================================================================

fn median_ns(trials: &[Duration]) -> f64 {
  let mut ns: Vec<f64> = trials.iter().map(|d| d.as_nanos() as f64).collect();
  ns.sort_by(|a, b| a.partial_cmp(b).unwrap());
  ns[ns.len() / 2]
}

fn bench<F: FnMut()>(mut f: F, warmup: usize, trials: usize) -> Duration {
  for _ in 0..warmup {
    f();
  }
  let mut times = Vec::with_capacity(trials);
  for _ in 0..trials {
    let start = Instant::now();
    f();
    times.push(start.elapsed());
  }
  let med_ns = median_ns(&times);
  Duration::from_nanos(med_ns as u64)
}

// ============================================================================
// Test data generation
// ============================================================================

fn gen_field_pairs(n: usize) -> Vec<([u64; 4], [u64; 4])> {
  let mut rng = rand_core::OsRng;
  (0..n)
    .map(|_| {
      let a = Bn254Fr::random(&mut rng);
      let b = Bn254Fr::random(&mut rng);
      (*a.to_limbs(), *b.to_limbs())
    })
    .collect()
}

fn gen_field_i64_pairs(n: usize) -> Vec<([u64; 4], u64)> {
  use rand_core::RngCore;
  let mut rng = rand_core::OsRng;
  (0..n)
    .map(|_| {
      let a = Bn254Fr::random(&mut rng);
      let b = rng.next_u64();
      (*a.to_limbs(), b)
    })
    .collect()
}

fn gen_field_i128_pairs(n: usize) -> Vec<([u64; 4], u64, u64)> {
  use rand_core::RngCore;
  let mut rng = rand_core::OsRng;
  (0..n)
    .map(|_| {
      let a = Bn254Fr::random(&mut rng);
      let b_lo = rng.next_u64();
      let b_hi = rng.next_u64();
      (*a.to_limbs(), b_lo, b_hi)
    })
    .collect()
}

fn gen_field_i32_pairs(n: usize) -> Vec<([u64; 4], u32)> {
  use rand_core::RngCore;
  let mut rng = rand_core::OsRng;
  (0..n)
    .map(|_| {
      let a = Bn254Fr::random(&mut rng);
      let b = rng.next_u32();
      (*a.to_limbs(), b)
    })
    .collect()
}

// ============================================================================
// Case 1: Field×Field dot product
// ============================================================================

fn bench_ff(sizes: &[usize]) {
  println!("\n=== Case 1: Field×Field Dot Product ===");
  println!(
    "{:<10} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}",
    "N", "naive_mul", "4x64_cur", "4x64_fused", "4x64_asm", "5x52_lazy", "8x32_neon"
  );
  println!("{}", "-".repeat(94));

  for &n in sizes {
    let data = gen_field_pairs(n);

    // Naive: full field multiply + field add per iteration
    let field_data: Vec<_> = data
      .iter()
      .map(|(a, b)| {
        (Bn254Fr::from_limbs(*a), Bn254Fr::from_limbs(*b))
      })
      .collect();
    let t_naive = bench(
      || {
        let mut acc = Bn254Fr::ZERO;
        for (a, b) in &field_data {
          acc += a * b;
        }
        black_box(&acc);
      },
      3,
      10,
    );

    // 4x64 current: mul_4_by_4 + add to 9-limb acc
    let t_cur = bench(
      || {
        let mut acc = [0u64; 9];
        for (a, b) in &data {
          let product = mul_4_by_4(a, b);
          let mut carry = 0u128;
          for i in 0..8 {
            let sum = (acc[i] as u128) + (product[i] as u128) + carry;
            acc[i] = sum as u64;
            carry = sum >> 64;
          }
          acc[8] = acc[8].wrapping_add(carry as u64);
        }
        black_box(&acc);
      },
      3,
      10,
    );

    // 4x64 fused
    let t_fused = bench(
      || {
        let mut acc = [0u64; 9];
        for (a, b) in &data {
          mac_4x4_into_fused(&mut acc, a, b);
        }
        black_box(&acc);
      },
      3,
      10,
    );

    // 4x64 asm
    let t_asm = bench(
      || {
        let mut acc = [0u64; 9];
        for (a, b) in &data {
          mac_4x4_into_asm(&mut acc, a, b);
        }
        black_box(&acc);
      },
      3,
      10,
    );

    // 5x52 lazy
    let data_52: Vec<_> = data.iter().map(|(a, b)| (to_52(a), to_52(b))).collect();
    let t_52 = bench(
      || {
        let mut cols = [0u128; 9];
        for (a, b) in &data_52 {
          mac_ff_52(&mut cols, a, b);
        }
        let _ = carry_propagate_52_to_9limb(black_box(&cols));
      },
      3,
      10,
    );

    // 8x32 NEON
    let data_32: Vec<_> = data.iter().map(|(a, b)| (to_32(a), to_32(b))).collect();
    let t_neon = bench(
      || {
        let mut cols = [0u128; 15];
        for (a, b) in &data_32 {
          mac_ff_neon(&mut cols, a, b);
        }
        let _ = carry_propagate_32_to_9limb(black_box(&cols));
      },
      3,
      10,
    );

    let ns_naive = t_naive.as_nanos() as f64 / n as f64;
    let ns_cur = t_cur.as_nanos() as f64 / n as f64;
    let ns_fused = t_fused.as_nanos() as f64 / n as f64;
    let ns_asm = t_asm.as_nanos() as f64 / n as f64;
    let ns_52 = t_52.as_nanos() as f64 / n as f64;
    let ns_neon = t_neon.as_nanos() as f64 / n as f64;

    println!(
      "{:<10} {:>10.2}ns {:>10.2}ns {:>10.2}ns {:>10.2}ns {:>10.2}ns {:>10.2}ns",
      n, ns_naive, ns_cur, ns_fused, ns_asm, ns_52, ns_neon
    );
  }
}

// ============================================================================
// Case 2: Field×i64 dot product
// ============================================================================

fn bench_fi64(sizes: &[usize]) {
  println!("\n=== Case 2: Field×i64 Dot Product ===");
  println!(
    "{:<10} {:>12} {:>12} {:>12} {:>12} {:>12}",
    "N", "naive_mul", "4x64_cur", "4x64_asm", "5x52_lazy", "8x32_scalar"
  );
  println!("{}", "-".repeat(70));

  for &n in sizes {
    let data = gen_field_i64_pairs(n);

    // Naive: full field multiply per iteration (convert i64→field, then field×field)
    let field_data: Vec<_> = data
      .iter()
      .map(|(a, b)| {
        let fa = Bn254Fr::from_limbs(*a);
        // Convert u64 scalar to field element
        let fb = Bn254Fr::from_limbs([*b, 0, 0, 0]);
        (fa, fb)
      })
      .collect();
    let t_naive = bench(
      || {
        let mut acc = Bn254Fr::ZERO;
        for (a, b) in &field_data {
          acc += a * b;
        }
        black_box(&acc);
      },
      3,
      10,
    );

    // 4x64 current: mac() chain
    let t_cur = bench(
      || {
        let mut acc = [0u64; 6];
        for (a, b) in &data {
          let (r0, c) = mac(acc[0], a[0], *b, 0);
          let (r1, c) = mac(acc[1], a[1], *b, c);
          let (r2, c) = mac(acc[2], a[2], *b, c);
          let (r3, c) = mac(acc[3], a[3], *b, c);
          let (r4, of) = acc[4].overflowing_add(c);
          acc[0] = r0;
          acc[1] = r1;
          acc[2] = r2;
          acc[3] = r3;
          acc[4] = r4;
          acc[5] = acc[5].wrapping_add(of as u64);
        }
        black_box(&acc);
      },
      3,
      10,
    );

    // 4x64 asm
    let t_asm = bench(
      || {
        let mut acc = [0u64; 6];
        for (a, b) in &data {
          mac_4x1_into_asm(&mut acc, a, *b);
        }
        black_box(&acc);
      },
      3,
      10,
    );

    // 5x52 lazy
    let data_52: Vec<_> = data.iter().map(|(a, b)| (to_52(a), *b)).collect();
    let t_52 = bench(
      || {
        let mut cols = [0u128; 5];
        for (a, b) in &data_52 {
          mac_fi64_52(&mut cols, a, *b);
        }
        let _ = carry_propagate_52_to_6limb(black_box(&cols));
      },
      3,
      10,
    );

    // 8x32 scalar
    let data_32: Vec<_> = data
      .iter()
      .map(|(a, b)| (to_32(a), *b as u32, (*b >> 32) as u32))
      .collect();
    let t_32 = bench(
      || {
        let mut cols = [0u128; 9];
        for (a, b_lo, b_hi) in &data_32 {
          mac_fi64_32(&mut cols, a, *b_lo, *b_hi);
        }
        let _ = carry_propagate_32_to_6limb(black_box(&cols));
      },
      3,
      10,
    );

    let ns_naive = t_naive.as_nanos() as f64 / n as f64;
    let ns_cur = t_cur.as_nanos() as f64 / n as f64;
    let ns_asm = t_asm.as_nanos() as f64 / n as f64;
    let ns_52 = t_52.as_nanos() as f64 / n as f64;
    let ns_32 = t_32.as_nanos() as f64 / n as f64;

    println!(
      "{:<10} {:>10.2}ns {:>10.2}ns {:>10.2}ns {:>10.2}ns {:>10.2}ns",
      n, ns_naive, ns_cur, ns_asm, ns_52, ns_32
    );
  }
}

// ============================================================================
// Case 3: Field×i128 dot product
// ============================================================================

fn bench_fi128(sizes: &[usize]) {
  println!("\n=== Case 3: Field×i128 Dot Product ===");
  println!(
    "{:<10} {:>12} {:>12} {:>12} {:>12}",
    "N", "4x64_cur", "4x64_asm", "5x52_lazy", "8x32_scalar"
  );
  println!("{}", "-".repeat(58));

  for &n in sizes {
    let data = gen_field_i128_pairs(n);

    // 4x64 current: 2-pass mac chain
    let t_cur = bench(
      || {
        let mut acc = [0u64; 7];
        for (a, b_lo, b_hi) in &data {
          // Pass 1: multiply by b_lo at offset 0
          let (r0, c) = mac(acc[0], a[0], *b_lo, 0);
          let (r1, c) = mac(acc[1], a[1], *b_lo, c);
          let (r2, c) = mac(acc[2], a[2], *b_lo, c);
          let (r3, c) = mac(acc[3], a[3], *b_lo, c);
          let (r4, of1) = acc[4].overflowing_add(c);
          let c1 = of1 as u64;
          acc[0] = r0;

          // Pass 2: multiply by b_hi at offset 1
          let (r1, c) = mac(r1, a[0], *b_hi, 0);
          let (r2, c) = mac(r2, a[1], *b_hi, c);
          let (r3, c) = mac(r3, a[2], *b_hi, c);
          let (r4, c) = mac(r4, a[3], *b_hi, c);
          let (r5, c) = mac(acc[5], c1, 1, c);
          acc[1] = r1;
          acc[2] = r2;
          acc[3] = r3;
          acc[4] = r4;
          acc[5] = r5;
          acc[6] = acc[6].wrapping_add(c);
        }
        black_box(&acc);
      },
      3,
      10,
    );

    // 4x64 asm
    let t_asm = bench(
      || {
        let mut acc = [0u64; 7];
        for (a, b_lo, b_hi) in &data {
          mac_4x2_into_asm(&mut acc, a, *b_lo, *b_hi);
        }
        black_box(&acc);
      },
      3,
      10,
    );

    // 5x52 lazy
    let data_52: Vec<_> = data.iter().map(|(a, bl, bh)| (to_52(a), *bl, *bh)).collect();
    let t_52 = bench(
      || {
        let mut cols = [0u128; 7];
        for (a, bl, bh) in &data_52 {
          // For 5x52, we need b split in 52-bit limbs
          // But the plan says b_lo/b_hi are 64-bit halves of i128
          // We'll use them directly as two "limb" values
          mac_fi128_52(&mut cols, a, *bl, *bh);
        }
        let _ = carry_propagate_52_to_7limb(black_box(&cols));
      },
      3,
      10,
    );

    // 8x32 scalar
    let data_32: Vec<_> = data
      .iter()
      .map(|(a, bl, bh)| {
        (
          to_32(a),
          [*bl as u32, (*bl >> 32) as u32, *bh as u32, (*bh >> 32) as u32],
        )
      })
      .collect();
    let t_32 = bench(
      || {
        let mut cols = [0u128; 12];
        for (a, b) in &data_32 {
          mac_fi128_32(&mut cols, a, b);
        }
        let _ = carry_propagate_32_to_7limb(black_box(&cols));
      },
      3,
      10,
    );

    let ns_cur = t_cur.as_nanos() as f64 / n as f64;
    let ns_asm = t_asm.as_nanos() as f64 / n as f64;
    let ns_52 = t_52.as_nanos() as f64 / n as f64;
    let ns_32 = t_32.as_nanos() as f64 / n as f64;

    println!(
      "{:<10} {:>10.2}ns {:>10.2}ns {:>10.2}ns {:>10.2}ns",
      n, ns_cur, ns_asm, ns_52, ns_32
    );
  }
}

// ============================================================================
// Case 4: Field×i32 dot product
// ============================================================================

fn bench_fi32(sizes: &[usize]) {
  println!("\n=== Case 4: Field×i32 Dot Product ===");
  println!(
    "{:<10} {:>12} {:>12} {:>12}",
    "N", "4x64_cur", "5x52_lazy", "8x32_scalar"
  );
  println!("{}", "-".repeat(46));

  for &n in sizes {
    let data = gen_field_i32_pairs(n);

    // 4x64 current: extends i32→i64, same mac chain
    let t_cur = bench(
      || {
        let mut acc = [0u64; 6];
        for (a, b) in &data {
          let b64 = *b as u64;
          let (r0, c) = mac(acc[0], a[0], b64, 0);
          let (r1, c) = mac(acc[1], a[1], b64, c);
          let (r2, c) = mac(acc[2], a[2], b64, c);
          let (r3, c) = mac(acc[3], a[3], b64, c);
          let (r4, of) = acc[4].overflowing_add(c);
          acc[0] = r0;
          acc[1] = r1;
          acc[2] = r2;
          acc[3] = r3;
          acc[4] = r4;
          acc[5] = acc[5].wrapping_add(of as u64);
        }
        black_box(&acc);
      },
      3,
      10,
    );

    // 5x52 lazy
    let data_52: Vec<_> = data.iter().map(|(a, b)| (to_52(a), *b as u64)).collect();
    let t_52 = bench(
      || {
        let mut cols = [0u128; 5];
        for (a, b) in &data_52 {
          mac_fi64_52(&mut cols, a, *b);
        }
        let _ = carry_propagate_52_to_6limb(black_box(&cols));
      },
      3,
      10,
    );

    // 8x32 scalar
    let data_32: Vec<_> = data.iter().map(|(a, b)| (to_32(a), *b)).collect();
    let t_32 = bench(
      || {
        let mut cols = [0u128; 9];
        for (a, b) in &data_32 {
          mac_fi64_32(&mut cols, a, *b, 0);
        }
        let _ = carry_propagate_32_to_6limb(black_box(&cols));
      },
      3,
      10,
    );

    let ns_cur = t_cur.as_nanos() as f64 / n as f64;
    let ns_52 = t_52.as_nanos() as f64 / n as f64;
    let ns_32 = t_32.as_nanos() as f64 / n as f64;

    println!(
      "{:<10} {:>10.2}ns {:>10.2}ns {:>10.2}ns",
      n, ns_cur, ns_52, ns_32
    );
  }
}

// ============================================================================
// Case 4b: 5x52 micro-optimizations
// ============================================================================

fn bench_52_micro(sizes: &[usize]) {
  println!("\n=== Case 4b: 5×52 Micro-Optimizations (Field×Field) ===");
  println!(
    "{:<10} {:>12} {:>12} {:>12} {:>12}",
    "N", "52_array", "52_locals", "52_K2", "52_unroll2"
  );
  println!("{}", "-".repeat(58));

  // Also test fi64 K=2 (fits in registers: 2×10 + 6 + 1 = 27 regs)
  {
    println!("\n  --- Field×i64 batching test ---");
    println!(
      "  {:<10} {:>12} {:>12} {:>12}",
      "N", "fi64_K1", "fi64_K2", "fi64_K3"
    );
    for &n in sizes {
      let data = gen_field_i64_pairs(n);
      let data_52: Vec<_> = data.iter().map(|(a, b)| (to_52(a), *b)).collect();

      // K=1
      let t_k1 = bench(
        || {
          let mut cols = [0u128; 5];
          for (a, b) in &data_52 {
            mac_fi64_52(&mut cols, a, *b);
          }
          let _ = carry_propagate_52_to_6limb(black_box(&cols));
        },
        3,
        10,
      );

      // K=2
      let t_k2 = bench(
        || {
          let mut cols0 = [0u128; 5];
          let mut cols1 = [0u128; 5];
          let half = data_52.len() / 2;
          for i in 0..half {
            let (a0, b0) = &data_52[2 * i];
            let (a1, b1) = &data_52[2 * i + 1];
            mac_fi64_52(&mut cols0, a0, *b0);
            mac_fi64_52(&mut cols1, a1, *b1);
          }
          let _ = carry_propagate_52_to_6limb(black_box(&cols0));
          let _ = carry_propagate_52_to_6limb(black_box(&cols1));
        },
        3,
        10,
      );

      // K=3
      let t_k3 = bench(
        || {
          let mut cols0 = [0u128; 5];
          let mut cols1 = [0u128; 5];
          let mut cols2 = [0u128; 5];
          let third = data_52.len() / 3;
          for i in 0..third {
            let (a0, b0) = &data_52[3 * i];
            let (a1, b1) = &data_52[3 * i + 1];
            let (a2, b2) = &data_52[3 * i + 2];
            mac_fi64_52(&mut cols0, a0, *b0);
            mac_fi64_52(&mut cols1, a1, *b1);
            mac_fi64_52(&mut cols2, a2, *b2);
          }
          let _ = carry_propagate_52_to_6limb(black_box(&cols0));
          let _ = carry_propagate_52_to_6limb(black_box(&cols1));
          let _ = carry_propagate_52_to_6limb(black_box(&cols2));
        },
        3,
        10,
      );

      let ns = |t: Duration| t.as_nanos() as f64 / n as f64;
      println!(
        "  {:<10} {:>10.2}ns {:>10.2}ns {:>10.2}ns",
        n,
        ns(t_k1),
        ns(t_k2),
        ns(t_k3)
      );
    }
    println!();
  }

  for &n in sizes {
    let data = gen_field_pairs(n);
    let data_52: Vec<_> = data.iter().map(|(a, b)| (to_52(a), to_52(b))).collect();

    // Baseline: mac_ff_52 with array
    let t_array = bench(
      || {
        let mut cols = [0u128; 9];
        for (a, b) in &data_52 {
          mac_ff_52(&mut cols, a, b);
        }
        let _ = carry_propagate_52_to_9limb(black_box(&cols));
      },
      3,
      10,
    );

    // Micro-opt 1: explicit locals
    let t_locals = bench(
      || {
        let (mut c0, mut c1, mut c2, mut c3, mut c4, mut c5, mut c6, mut c7, mut c8) =
          (0u128, 0u128, 0u128, 0u128, 0u128, 0u128, 0u128, 0u128, 0u128);
        for (a, b) in &data_52 {
          mac_ff_52_locals(&mut c0, &mut c1, &mut c2, &mut c3, &mut c4, &mut c5, &mut c6, &mut c7, &mut c8, a, b);
        }
        let cols = black_box([c0, c1, c2, c3, c4, c5, c6, c7, c8]);
        let _ = carry_propagate_52_to_9limb(&cols);
      },
      3,
      10,
    );

    // Micro-opt 2: K=2 ILP (two independent accumulator sets)
    let t_k2 = bench(
      || {
        let mut cols0 = [0u128; 9];
        let mut cols1 = [0u128; 9];
        let half = data_52.len() / 2;
        for i in 0..half {
          let (a0, b0) = &data_52[2 * i];
          let (a1, b1) = &data_52[2 * i + 1];
          mac_ff_52(&mut cols0, a0, b0);
          mac_ff_52(&mut cols1, a1, b1);
        }
        let _ = carry_propagate_52_to_9limb(black_box(&cols0));
        let _ = carry_propagate_52_to_9limb(black_box(&cols1));
      },
      3,
      10,
    );

    // Micro-opt 3: unroll by 2 (process 2 elements per iteration, single accumulator)
    let t_unroll2 = bench(
      || {
        let mut cols = [0u128; 9];
        let len = data_52.len();
        let mut i = 0;
        while i + 1 < len {
          let (a0, b0) = &data_52[i];
          let (a1, b1) = &data_52[i + 1];
          mac_ff_52(&mut cols, a0, b0);
          mac_ff_52(&mut cols, a1, b1);
          i += 2;
        }
        if i < len {
          let (a, b) = &data_52[i];
          mac_ff_52(&mut cols, a, b);
        }
        let _ = carry_propagate_52_to_9limb(black_box(&cols));
      },
      3,
      10,
    );

    let ns = |t: Duration| t.as_nanos() as f64 / n as f64;
    println!(
      "{:<10} {:>10.2}ns {:>10.2}ns {:>10.2}ns {:>10.2}ns",
      n,
      ns(t_array),
      ns(t_locals),
      ns(t_k2),
      ns(t_unroll2)
    );
  }
}

// ============================================================================
// Case 5: ILP Batching (K=1,2,4)
// ============================================================================

fn bench_ilp(sizes: &[usize]) {
  println!("\n=== Case 5: ILP Batching (Field×Field) ===");
  println!(
    "{:<10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
    "N", "cur_K1", "cur_K2", "cur_K4", "fused_K1", "fused_K2", "fused_K4"
  );
  println!("{}", "-".repeat(72));

  for &n in sizes {
    let data = gen_field_pairs(n);

    // K=1 current
    let t_k1 = bench(
      || {
        let mut acc = [0u64; 9];
        for (a, b) in &data {
          let product = mul_4_by_4(a, b);
          let mut carry = 0u128;
          for i in 0..8 {
            let sum = (acc[i] as u128) + (product[i] as u128) + carry;
            acc[i] = sum as u64;
            carry = sum >> 64;
          }
          acc[8] = acc[8].wrapping_add(carry as u64);
        }
        black_box(&acc);
      },
      3,
      10,
    );

    // K=2 current (2 independent accumulators)
    let t_k2 = bench(
      || {
        let mut acc0 = [0u64; 9];
        let mut acc1 = [0u64; 9];
        let half = data.len() / 2;
        for i in 0..half {
          let (a0, b0) = &data[2 * i];
          let (a1, b1) = &data[2 * i + 1];

          let p0 = mul_4_by_4(a0, b0);
          let p1 = mul_4_by_4(a1, b1);

          let mut carry0 = 0u128;
          let mut carry1 = 0u128;
          for j in 0..8 {
            let sum0 = (acc0[j] as u128) + (p0[j] as u128) + carry0;
            acc0[j] = sum0 as u64;
            carry0 = sum0 >> 64;
            let sum1 = (acc1[j] as u128) + (p1[j] as u128) + carry1;
            acc1[j] = sum1 as u64;
            carry1 = sum1 >> 64;
          }
          acc0[8] = acc0[8].wrapping_add(carry0 as u64);
          acc1[8] = acc1[8].wrapping_add(carry1 as u64);
        }
        black_box(&acc0);
        black_box(&acc1);
      },
      3,
      10,
    );

    // K=4 current
    let t_k4 = bench(
      || {
        let mut accs = [[0u64; 9]; 4];
        let quarter = data.len() / 4;
        for i in 0..quarter {
          for k in 0..4 {
            let (a, b) = &data[4 * i + k];
            let product = mul_4_by_4(a, b);
            let mut carry = 0u128;
            for j in 0..8 {
              let sum = (accs[k][j] as u128) + (product[j] as u128) + carry;
              accs[k][j] = sum as u64;
              carry = sum >> 64;
            }
            accs[k][8] = accs[k][8].wrapping_add(carry as u64);
          }
        }
        black_box(&accs);
      },
      3,
      10,
    );

    // K=1 fused
    let t_fk1 = bench(
      || {
        let mut acc = [0u64; 9];
        for (a, b) in &data {
          mac_4x4_into_fused(&mut acc, a, b);
        }
        black_box(&acc);
      },
      3,
      10,
    );

    // K=2 fused
    let t_fk2 = bench(
      || {
        let mut acc0 = [0u64; 9];
        let mut acc1 = [0u64; 9];
        let half = data.len() / 2;
        for i in 0..half {
          let (a0, b0) = &data[2 * i];
          let (a1, b1) = &data[2 * i + 1];
          mac_4x4_into_fused(&mut acc0, a0, b0);
          mac_4x4_into_fused(&mut acc1, a1, b1);
        }
        black_box(&acc0);
        black_box(&acc1);
      },
      3,
      10,
    );

    // K=4 fused
    let t_fk4 = bench(
      || {
        let mut accs = [[0u64; 9]; 4];
        let quarter = data.len() / 4;
        for i in 0..quarter {
          for k in 0..4 {
            let (a, b) = &data[4 * i + k];
            mac_4x4_into_fused(&mut accs[k], a, b);
          }
        }
        black_box(&accs);
      },
      3,
      10,
    );

    let ns = |t: Duration| t.as_nanos() as f64 / n as f64;

    println!(
      "{:<10} {:>8.2}ns {:>8.2}ns {:>8.2}ns {:>8.2}ns {:>8.2}ns {:>8.2}ns",
      n,
      ns(t_k1),
      ns(t_k2),
      ns(t_k4),
      ns(t_fk1),
      ns(t_fk2),
      ns(t_fk4)
    );
  }
}

// ============================================================================
// Case 6: Reduction operations (standalone)
// ============================================================================

fn bench_reduce() {
  let n = 1_000_000;
  println!("\n=== Case 6: Reduction Operations (N={n}) ===");

  // Generate random 9-limb accumulators
  use rand_core::RngCore;
  let mut rng = rand_core::OsRng;

  let acc9s: Vec<[u64; 9]> = (0..n)
    .map(|_| {
      // Generate realistic 9-limb accumulator (sum of ~100 products)
      let mut acc = [0u64; 9];
      for _ in 0..100 {
        let a = [rng.next_u64(), rng.next_u64(), rng.next_u64(), rng.next_u64() >> 2];
        let b = [rng.next_u64(), rng.next_u64(), rng.next_u64(), rng.next_u64() >> 2];
        let product = mul_4_by_4(&a, &b);
        let mut carry = 0u128;
        for i in 0..8 {
          let sum = (acc[i] as u128) + (product[i] as u128) + carry;
          acc[i] = sum as u64;
          carry = sum >> 64;
        }
        acc[8] = acc[8].wrapping_add(carry as u64);
      }
      acc
    })
    .collect();

  let acc6s: Vec<[u64; 6]> = (0..n)
    .map(|_| {
      [
        rng.next_u64(),
        rng.next_u64(),
        rng.next_u64(),
        rng.next_u64() >> 2,
        rng.next_u64() >> 48,
        0,
      ]
    })
    .collect();

  let acc7s: Vec<[u64; 7]> = (0..n)
    .map(|_| {
      [
        rng.next_u64(),
        rng.next_u64(),
        rng.next_u64(),
        rng.next_u64() >> 2,
        rng.next_u64() >> 48,
        0,
        0,
      ]
    })
    .collect();

  // Montgomery reduce 9
  let t_mont9 = bench(
    || {
      for acc in &acc9s {
        black_box(montgomery_reduce_9::<Bn254Fr>(acc));
      }
    },
    3,
    10,
  );

  // Barrett reduce 6
  let t_bar6 = bench(
    || {
      for acc in &acc6s {
        black_box(barrett_reduce_6::<Bn254Fr>(acc));
      }
    },
    3,
    10,
  );

  // Barrett reduce 7
  let t_bar7 = bench(
    || {
      for acc in &acc7s {
        black_box(barrett_reduce_7::<Bn254Fr>(acc));
      }
    },
    3,
    10,
  );

  // 5x52 carry prop + montgomery 9
  let cols9s: Vec<[u128; 9]> = acc9s
    .iter()
    .map(|acc| {
      let mut cols = [0u128; 9];
      for (i, &v) in acc.iter().enumerate() {
        cols[i] = v as u128;
      }
      cols
    })
    .collect();

  let t_52_mont9 = bench(
    || {
      for cols in &cols9s {
        let limbs = carry_propagate_52_to_9limb(cols);
        black_box(montgomery_reduce_9::<Bn254Fr>(&limbs));
      }
    },
    3,
    10,
  );

  // 8x32 carry prop + montgomery 9
  let cols16s: Vec<[u128; 15]> = acc9s
    .iter()
    .map(|acc| {
      let mut cols = [0u128; 15];
      for (i, &v) in acc.iter().enumerate().take(8) {
        // Split 64-bit into two 32-bit columns
        cols[2 * i] = (v & 0xFFFF_FFFF) as u128;
        if 2 * i + 1 < 15 {
          cols[2 * i + 1] = (v >> 32) as u128;
        }
      }
      cols
    })
    .collect();

  let t_32_mont9 = bench(
    || {
      for cols in &cols16s {
        let limbs = carry_propagate_32_to_9limb(cols);
        black_box(montgomery_reduce_9::<Bn254Fr>(&limbs));
      }
    },
    3,
    10,
  );

  let ns = |t: Duration| t.as_nanos() as f64 / n as f64;

  println!(
    "  montgomery_reduce_9:            {:>8.2} ns/op",
    ns(t_mont9)
  );
  println!(
    "  barrett_reduce_6:               {:>8.2} ns/op",
    ns(t_bar6)
  );
  println!(
    "  barrett_reduce_7:               {:>8.2} ns/op",
    ns(t_bar7)
  );
  println!(
    "  52_carry_prop + mont_reduce_9:  {:>8.2} ns/op",
    ns(t_52_mont9)
  );
  println!(
    "  32_carry_prop + mont_reduce_9:  {:>8.2} ns/op",
    ns(t_32_mont9)
  );
}

// ============================================================================
// Case 7: Standalone Field Operations (mul, add)
// ============================================================================

fn bench_standalone_ops() {
  let n = 1_000_000;
  println!("\n=== Case 7: Standalone Field Operations (N={n}) ===");
  println!("  (Shows that 5×52 is slower for individual ops — advantage is only in dot products)\n");

  let data = gen_field_pairs(n);

  // 1. Baseline: halo2curves field multiply
  let field_data: Vec<_> = data
    .iter()
    .map(|(a, b)| (Bn254Fr::from_limbs(*a), Bn254Fr::from_limbs(*b)))
    .collect();

  let t_mul_baseline = bench(
    || {
      for (a, b) in &field_data {
        black_box(a * b);
      }
    },
    3,
    10,
  );

  // 2. 5×52 field multiply (to_52 → mac_ff_52 → carry_prop → mont_reduce)
  let t_mul_52 = bench(
    || {
      for (a, b) in &data {
        black_box(mul_52::<Bn254Fr>(a, b));
      }
    },
    3,
    10,
  );

  // 3. Baseline: halo2curves field add
  let t_add_baseline = bench(
    || {
      for (a, b) in &field_data {
        black_box(a + b);
      }
    },
    3,
    10,
  );

  let ns = |t: Duration| t.as_nanos() as f64 / n as f64;

  println!(
    "  field_mul (halo2curves):   {:>8.2} ns/op",
    ns(t_mul_baseline)
  );
  println!(
    "  field_mul (5×52 pipeline): {:>8.2} ns/op  ({:.2}× vs baseline)",
    ns(t_mul_52),
    ns(t_mul_baseline) / ns(t_mul_52)
  );
  println!(
    "  field_add (halo2curves):   {:>8.2} ns/op",
    ns(t_add_baseline)
  );
  println!();
  println!("  Note: 5×52 field_mul is slower because conversion + carry_prop + REDC overhead");
  println!("  is per-element. In dot products, N MACs share one reduction pass.");
}

// ============================================================================
// Correctness verification
// ============================================================================

fn verify_correctness() {
  println!("\n=== Correctness Verification ===");

  use rand_core::RngCore;
  let mut rng = rand_core::OsRng;
  let n = 1000;

  // Verify Field×Field
  {
    let mut acc_ref = [0u64; 9];
    let mut acc_fused = [0u64; 9];
    let mut acc_asm = [0u64; 9];
    let mut cols_52 = [0u128; 9];
    let mut cols_32 = [0u128; 15];

    for _ in 0..n {
      let a = [rng.next_u64(), rng.next_u64(), rng.next_u64(), rng.next_u64() >> 2];
      let b = [rng.next_u64(), rng.next_u64(), rng.next_u64(), rng.next_u64() >> 2];

      // Reference
      let product = mul_4_by_4(&a, &b);
      let mut carry = 0u128;
      for i in 0..8 {
        let sum = (acc_ref[i] as u128) + (product[i] as u128) + carry;
        acc_ref[i] = sum as u64;
        carry = sum >> 64;
      }
      acc_ref[8] = acc_ref[8].wrapping_add(carry as u64);

      // Fused
      mac_4x4_into_fused(&mut acc_fused, &a, &b);

      // Asm
      mac_4x4_into_asm(&mut acc_asm, &a, &b);

      // 5x52
      let a52 = to_52(&a);
      let b52 = to_52(&b);
      mac_ff_52(&mut cols_52, &a52, &b52);

      // 8x32
      let a32 = to_32(&a);
      let b32 = to_32(&b);
      mac_ff_32(&mut cols_32, &a32, &b32);
    }

    // Reduce all and compare
    let r_ref = montgomery_reduce_9::<Bn254Fr>(&acc_ref);
    let r_fused = montgomery_reduce_9::<Bn254Fr>(&acc_fused);
    let r_asm = montgomery_reduce_9::<Bn254Fr>(&acc_asm);
    let limbs_52 = carry_propagate_52_to_9limb(&cols_52);
    let r_52 = montgomery_reduce_9::<Bn254Fr>(&limbs_52);
    let limbs_32 = carry_propagate_32_to_9limb(&cols_32);
    let r_32 = montgomery_reduce_9::<Bn254Fr>(&limbs_32);

    assert_eq!(r_ref, r_fused, "Fused mismatch!");
    assert_eq!(r_ref, r_asm, "Asm mismatch!");
    assert_eq!(r_ref, r_52, "5x52 mismatch!");
    assert_eq!(r_ref, r_32, "8x32 mismatch!");
    println!("  Field×Field: ALL PASS ({n} products)");
  }

  // Verify Field×i64
  {
    let mut acc_ref = [0u64; 6];
    let mut acc_asm = [0u64; 6];
    let mut cols_52 = [0u128; 5];

    for _ in 0..n {
      let a = [rng.next_u64(), rng.next_u64(), rng.next_u64(), rng.next_u64() >> 2];
      let b = rng.next_u64();

      // Reference
      let (r0, c) = mac(acc_ref[0], a[0], b, 0);
      let (r1, c) = mac(acc_ref[1], a[1], b, c);
      let (r2, c) = mac(acc_ref[2], a[2], b, c);
      let (r3, c) = mac(acc_ref[3], a[3], b, c);
      let (r4, of) = acc_ref[4].overflowing_add(c);
      acc_ref[0] = r0;
      acc_ref[1] = r1;
      acc_ref[2] = r2;
      acc_ref[3] = r3;
      acc_ref[4] = r4;
      acc_ref[5] = acc_ref[5].wrapping_add(of as u64);

      // Asm
      mac_4x1_into_asm(&mut acc_asm, &a, b);

      // 5x52
      let a52 = to_52(&a);
      mac_fi64_52(&mut cols_52, &a52, b);
    }

    let r_ref = barrett_reduce_6::<Bn254Fr>(&acc_ref);
    let r_asm = barrett_reduce_6::<Bn254Fr>(&acc_asm);
    let limbs_52 = carry_propagate_52_to_6limb(&cols_52);
    let r_52 = barrett_reduce_6::<Bn254Fr>(&limbs_52);

    assert_eq!(r_ref, r_asm, "Fi64 asm mismatch!");
    assert_eq!(r_ref, r_52, "Fi64 5x52 mismatch!");
    println!("  Field×i64:   ALL PASS ({n} products)");
  }

  println!("  All correctness checks passed!");
}

// ============================================================================
// Case 8: Batched MAC Benchmark (K=1,2,3,4,8)
// ============================================================================

fn bench_batching(sizes: &[usize]) {
  println!("\n=== Batched MAC Benchmark ===");

  // --- Field×Field ---
  println!("\n--- Field×Field ---");
  println!(
    "{:<10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
    "N", "naive", "5x52_K1", "5x52_K2", "5x52_K3", "5x52_K4", "5x52_K8",
    "4x64_K1", "4x64_K2", "4x64_K3", "4x64_K4", "4x64_K8"
  );
  println!("{}", "-".repeat(130));

  for &n in sizes {
    let data = gen_field_pairs(n);
    let data_52: Vec<_> = data.iter().map(|(a, b)| (to_52(a), to_52(b))).collect();
    let field_data: Vec<_> = data
      .iter()
      .map(|(a, b)| (Bn254Fr::from_limbs(*a), Bn254Fr::from_limbs(*b)))
      .collect();

    // Naive
    let t_naive = bench(
      || {
        let mut acc = Bn254Fr::ZERO;
        for (a, b) in &field_data {
          acc += a * b;
        }
        black_box(&acc);
      },
      3, 10,
    );

    // 5x52 K=1..8
    let ks: [usize; 5] = [1, 2, 3, 4, 8];
    let mut t_52 = [Duration::ZERO; 5];
    for (ki, &k) in ks.iter().enumerate() {
      t_52[ki] = bench(
        || {
          let mut cols = [[0u128; 9]; 8];
          let chunk = data_52.len() / k;
          for i in 0..chunk {
            for j in 0..k {
              let (a, b) = &data_52[k * i + j];
              mac_ff_52(&mut cols[j], a, b);
            }
          }
          for j in 0..k {
            let _ = carry_propagate_52_to_9limb(black_box(&cols[j]));
          }
        },
        3, 10,
      );
    }

    // 4x64 fused K=1..8
    let mut t_64 = [Duration::ZERO; 5];
    for (ki, &k) in ks.iter().enumerate() {
      t_64[ki] = bench(
        || {
          let mut accs = [[0u64; 9]; 8];
          let chunk = data.len() / k;
          for i in 0..chunk {
            for j in 0..k {
              let (a, b) = &data[k * i + j];
              mac_4x4_into_fused(&mut accs[j], a, b);
            }
          }
          black_box(&accs);
        },
        3, 10,
      );
    }

    let ns = |t: Duration| t.as_nanos() as f64 / n as f64;
    println!(
      "{:<10} {:>8.2}ns {:>8.2}ns {:>8.2}ns {:>8.2}ns {:>8.2}ns {:>8.2}ns {:>8.2}ns {:>8.2}ns {:>8.2}ns {:>8.2}ns {:>8.2}ns",
      n, ns(t_naive),
      ns(t_52[0]), ns(t_52[1]), ns(t_52[2]), ns(t_52[3]), ns(t_52[4]),
      ns(t_64[0]), ns(t_64[1]), ns(t_64[2]), ns(t_64[3]), ns(t_64[4]),
    );
  }

  // --- Field×i64 ---
  println!("\n--- Field×i64 ---");
  println!(
    "{:<10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
    "N", "naive", "5x52_K1", "5x52_K2", "5x52_K3", "5x52_K4", "5x52_K8",
    "4x64_K1", "4x64_K2", "4x64_K3", "4x64_K4", "4x64_K8"
  );
  println!("{}", "-".repeat(130));

  for &n in sizes {
    let data = gen_field_i64_pairs(n);
    let data_52: Vec<_> = data.iter().map(|(a, b)| (to_52(a), *b)).collect();
    let field_data: Vec<_> = data
      .iter()
      .map(|(a, b)| {
        (Bn254Fr::from_limbs(*a), Bn254Fr::from_limbs([*b, 0, 0, 0]))
      })
      .collect();

    // Naive
    let t_naive = bench(
      || {
        let mut acc = Bn254Fr::ZERO;
        for (a, b) in &field_data {
          acc += a * b;
        }
        black_box(&acc);
      },
      3, 10,
    );

    // 5x52 K=1..8
    let ks: [usize; 5] = [1, 2, 3, 4, 8];
    let mut t_52 = [Duration::ZERO; 5];
    for (ki, &k) in ks.iter().enumerate() {
      t_52[ki] = bench(
        || {
          let mut cols = [[0u128; 5]; 8];
          let chunk = data_52.len() / k;
          for i in 0..chunk {
            for j in 0..k {
              let (a, b) = &data_52[k * i + j];
              mac_fi64_52(&mut cols[j], a, *b);
            }
          }
          for j in 0..k {
            let _ = carry_propagate_52_to_6limb(black_box(&cols[j]));
          }
        },
        3, 10,
      );
    }

    // 4x64 mac K=1..8
    let mut t_64 = [Duration::ZERO; 5];
    for (ki, &k) in ks.iter().enumerate() {
      t_64[ki] = bench(
        || {
          let mut accs = [[0u64; 6]; 8];
          let chunk = data.len() / k;
          for i in 0..chunk {
            for j in 0..k {
              let (a, b) = &data[k * i + j];
              let acc = &mut accs[j];
              let (r0, c) = mac(acc[0], a[0], *b, 0);
              let (r1, c) = mac(acc[1], a[1], *b, c);
              let (r2, c) = mac(acc[2], a[2], *b, c);
              let (r3, c) = mac(acc[3], a[3], *b, c);
              let (r4, of) = acc[4].overflowing_add(c);
              acc[0] = r0; acc[1] = r1; acc[2] = r2; acc[3] = r3;
              acc[4] = r4; acc[5] = acc[5].wrapping_add(of as u64);
            }
          }
          black_box(&accs);
        },
        3, 10,
      );
    }

    let ns = |t: Duration| t.as_nanos() as f64 / n as f64;
    println!(
      "{:<10} {:>8.2}ns {:>8.2}ns {:>8.2}ns {:>8.2}ns {:>8.2}ns {:>8.2}ns {:>8.2}ns {:>8.2}ns {:>8.2}ns {:>8.2}ns {:>8.2}ns",
      n, ns(t_naive),
      ns(t_52[0]), ns(t_52[1]), ns(t_52[2]), ns(t_52[3]), ns(t_52[4]),
      ns(t_64[0]), ns(t_64[1]), ns(t_64[2]), ns(t_64[3]), ns(t_64[4]),
    );
  }

  // --- Field×i128 ---
  println!("\n--- Field×i128 ---");
  println!(
    "{:<10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
    "N", "naive", "5x52_K1", "5x52_K2", "5x52_K3", "5x52_K4", "5x52_K8",
    "4x64_K1", "4x64_K2", "4x64_K3", "4x64_K4", "4x64_K8"
  );
  println!("{}", "-".repeat(130));

  for &n in sizes {
    let data = gen_field_i128_pairs(n);
    let data_52: Vec<_> = data.iter().map(|(a, bl, bh)| (to_52(a), *bl, *bh)).collect();
    let field_data: Vec<_> = data
      .iter()
      .map(|(a, b_lo, b_hi)| {
        (Bn254Fr::from_limbs(*a), Bn254Fr::from_limbs([*b_lo, *b_hi, 0, 0]))
      })
      .collect();

    // Naive
    let t_naive = bench(
      || {
        let mut acc = Bn254Fr::ZERO;
        for (a, b) in &field_data {
          acc += a * b;
        }
        black_box(&acc);
      },
      3, 10,
    );

    // 5x52 K=1..8
    let ks: [usize; 5] = [1, 2, 3, 4, 8];
    let mut t_52 = [Duration::ZERO; 5];
    for (ki, &k) in ks.iter().enumerate() {
      t_52[ki] = bench(
        || {
          let mut cols = [[0u128; 7]; 8];
          let chunk = data_52.len() / k;
          for i in 0..chunk {
            for j in 0..k {
              let (a, bl, bh) = &data_52[k * i + j];
              mac_fi128_52(&mut cols[j], a, *bl, *bh);
            }
          }
          for j in 0..k {
            let _ = carry_propagate_52_to_7limb(black_box(&cols[j]));
          }
        },
        3, 10,
      );
    }

    // 4x64 2-pass mac K=1..8
    let mut t_64 = [Duration::ZERO; 5];
    for (ki, &k) in ks.iter().enumerate() {
      t_64[ki] = bench(
        || {
          let mut accs = [[0u64; 7]; 8];
          let chunk = data.len() / k;
          for i in 0..chunk {
            for j in 0..k {
              let (a, b_lo, b_hi) = &data[k * i + j];
              let acc = &mut accs[j];
              // Pass 1: a × b_lo → acc[0..5]
              let (r0, c) = mac(acc[0], a[0], *b_lo, 0);
              let (r1, c) = mac(acc[1], a[1], *b_lo, c);
              let (r2, c) = mac(acc[2], a[2], *b_lo, c);
              let (r3, c) = mac(acc[3], a[3], *b_lo, c);
              let (r4, of1) = acc[4].overflowing_add(c);
              let c1 = of1 as u64;
              acc[0] = r0;
              // Pass 2: a × b_hi → acc[1..6]
              let (r1, c) = mac(r1, a[0], *b_hi, 0);
              let (r2, c) = mac(r2, a[1], *b_hi, c);
              let (r3, c) = mac(r3, a[2], *b_hi, c);
              let (r4, c) = mac(r4, a[3], *b_hi, c);
              let (r5, c) = mac(acc[5], c1, 1, c);
              acc[1] = r1; acc[2] = r2; acc[3] = r3; acc[4] = r4;
              acc[5] = r5; acc[6] = acc[6].wrapping_add(c);
            }
          }
          black_box(&accs);
        },
        3, 10,
      );
    }

    let ns = |t: Duration| t.as_nanos() as f64 / n as f64;
    println!(
      "{:<10} {:>8.2}ns {:>8.2}ns {:>8.2}ns {:>8.2}ns {:>8.2}ns {:>8.2}ns {:>8.2}ns {:>8.2}ns {:>8.2}ns {:>8.2}ns {:>8.2}ns",
      n, ns(t_naive),
      ns(t_52[0]), ns(t_52[1]), ns(t_52[2]), ns(t_52[3]), ns(t_52[4]),
      ns(t_64[0]), ns(t_64[1]), ns(t_64[2]), ns(t_64[3]), ns(t_64[4]),
    );
  }
}

// ============================================================================
// Profile mode
// ============================================================================

fn profile(variant: &str) {
  let n = 1 << 20; // 1M iterations for profiling
  let iters = 50;

  println!("Profiling variant '{variant}' with N={n}, {iters} iterations...");

  match variant {
    "ff_4x64_current" | "ff_4x64_current_k1" => {
      let data = gen_field_pairs(n);
      for _ in 0..iters {
        let mut acc = [0u64; 9];
        for (a, b) in &data {
          let product = mul_4_by_4(a, b);
          let mut carry = 0u128;
          for i in 0..8 {
            let sum = (acc[i] as u128) + (product[i] as u128) + carry;
            acc[i] = sum as u64;
            carry = sum >> 64;
          }
          acc[8] = acc[8].wrapping_add(carry as u64);
        }
        black_box(&acc);
      }
    }
    "ff_4x64_fused" => {
      let data = gen_field_pairs(n);
      for _ in 0..iters {
        let mut acc = [0u64; 9];
        for (a, b) in &data {
          mac_4x4_into_fused(&mut acc, a, b);
        }
        black_box(&acc);
      }
    }
    "ff_4x64_asm" => {
      let data = gen_field_pairs(n);
      for _ in 0..iters {
        let mut acc = [0u64; 9];
        for (a, b) in &data {
          mac_4x4_into_asm(&mut acc, a, b);
        }
        black_box(&acc);
      }
    }
    "ff_5x52_lazy" => {
      let data = gen_field_pairs(n);
      let data_52: Vec<_> = data.iter().map(|(a, b)| (to_52(a), to_52(b))).collect();
      for _ in 0..iters {
        let mut cols = [0u128; 9];
        for (a, b) in &data_52 {
          mac_ff_52(&mut cols, a, b);
        }
        let _ = carry_propagate_52_to_9limb(black_box(&cols));
      }
    }
    "fi64_4x64_current" => {
      let data = gen_field_i64_pairs(n);
      for _ in 0..iters {
        let mut acc = [0u64; 6];
        for (a, b) in &data {
          let (r0, c) = mac(acc[0], a[0], *b, 0);
          let (r1, c) = mac(acc[1], a[1], *b, c);
          let (r2, c) = mac(acc[2], a[2], *b, c);
          let (r3, c) = mac(acc[3], a[3], *b, c);
          let (r4, of) = acc[4].overflowing_add(c);
          acc[0] = r0;
          acc[1] = r1;
          acc[2] = r2;
          acc[3] = r3;
          acc[4] = r4;
          acc[5] = acc[5].wrapping_add(of as u64);
        }
        black_box(&acc);
      }
    }
    _ => {
      eprintln!("Unknown variant: {variant}");
      eprintln!("Available: ff_4x64_current, ff_4x64_fused, ff_4x64_asm, ff_5x52_lazy, fi64_4x64_current");
      std::process::exit(1);
    }
  }
}

// ============================================================================
// Main
// ============================================================================

fn main() {
  let cli = Cli::parse();
  let sizes = vec![1 << 10, 1 << 14, 1 << 18, 1 << 20];

  // Always verify correctness first
  verify_correctness();

  match cli.command {
    Commands::All => {
      bench_ff(&sizes);
      bench_fi64(&sizes);
      bench_fi128(&sizes);
      bench_fi32(&sizes);
      bench_52_micro(&sizes);
      bench_ilp(&sizes);
      bench_reduce();
      bench_standalone_ops();
      bench_batching(&sizes);
    }
    Commands::Case { name } => match name.as_str() {
      "ff" => bench_ff(&sizes),
      "fi64" => bench_fi64(&sizes),
      "fi128" => bench_fi128(&sizes),
      "fi32" => bench_fi32(&sizes),
      "52micro" => bench_52_micro(&sizes),
      "ilp" => bench_ilp(&sizes),
      "reduce" => bench_reduce(),
      "standalone" => bench_standalone_ops(),
      "batch" => bench_batching(&sizes),
      _ => {
        eprintln!("Unknown case: {name}");
        eprintln!("Available: ff, fi64, fi128, fi32, ilp, reduce, standalone, batch");
        std::process::exit(1);
      }
    },
    Commands::Profile { variant } => profile(&variant),
  }
}
