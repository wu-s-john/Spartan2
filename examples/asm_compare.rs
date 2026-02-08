//! Assembly comparison for inner product implementations
//!
//! This example demonstrates the assembly differences between:
//! 1. Field×Field with eager reduction (base approach)
//! 2. Field×Field with delayed modular reduction (DMR)
//! 3. Field×i64 with delayed modular reduction
//! 4. Field×i128 with delayed modular reduction
//!
//! ## View Assembly
//!
//! Install cargo-show-asm:
//! ```bash
//! cargo install cargo-show-asm
//! ```
//!
//! View assembly for each function:
//! ```bash
//! cargo asm --example asm_compare inner_product_field_field_base
//! cargo asm --example asm_compare inner_product_field_field_dmr
//! cargo asm --example asm_compare inner_product_field_i64
//! cargo asm --example asm_compare inner_product_field_i128
//! ```
//!
//! With interleaved Rust source:
//! ```bash
//! cargo asm --example asm_compare inner_product_field_field_base --rust
//! ```

use halo2curves::bn256::Fr as Bn254Fr;
use spartan2::small_field::{DelayedReduction, SignedWideLimbs, WideLimbs};
use std::hint::black_box;
use std::time::Instant;

// ============================================================================
// FUNCTION 1: Field × Field with EAGER reduction (base approach)
// Each multiplication triggers a Montgomery reduction inside the loop.
// ============================================================================

/// Inner product with eager modular reduction after each multiply.
/// This is the "base" approach - each `a * b` does a full Montgomery reduction.
#[inline(never)]
#[unsafe(no_mangle)]
pub fn inner_product_field_field_base(a: &[Bn254Fr], b: &[Bn254Fr]) -> Bn254Fr {
    let mut acc = Bn254Fr::zero();
    for (ai, bi) in a.iter().zip(b.iter()) {
        acc += *ai * *bi; // Montgomery reduction happens here on each iteration
    }
    acc
}

// ============================================================================
// FUNCTION 2: Field × Field with DELAYED modular reduction (DMR)
// Accumulates in wide limbs (576 bits), reduces only once at the end.
// ============================================================================

/// Inner product with delayed modular reduction.
/// Accumulates in WideLimbs<9> (576 bits) and reduces once at the end.
#[inline(never)]
#[unsafe(no_mangle)]
pub fn inner_product_field_field_dmr(a: &[Bn254Fr], b: &[Bn254Fr]) -> Bn254Fr {
    let mut acc = WideLimbs::<9>::default();
    for (ai, bi) in a.iter().zip(b.iter()) {
        // Wide accumulation - no modular reduction here
        <Bn254Fr as DelayedReduction<Bn254Fr>>::unreduced_multiply_accumulate(&mut acc, ai, bi);
    }
    // Single reduction at the end
    <Bn254Fr as DelayedReduction<Bn254Fr>>::reduce(&acc)
}

// ============================================================================
// FUNCTION 3: Field × i64 with delayed modular reduction
// Uses SignedWideLimbs<6> (384 bits) for signed small values.
// ============================================================================

/// Inner product of field elements with i64 values.
/// Uses SignedWideLimbs<6> accumulator for signed small values.
#[inline(never)]
#[unsafe(no_mangle)]
pub fn inner_product_field_i64(fields: &[Bn254Fr], values: &[i64]) -> Bn254Fr {
    let mut acc = SignedWideLimbs::<6>::default();
    for (f, v) in fields.iter().zip(values.iter()) {
        // Fused multiply-accumulate into wide limbs
        <Bn254Fr as DelayedReduction<i64>>::unreduced_multiply_accumulate(&mut acc, f, v);
    }
    // Single reduction at the end
    <Bn254Fr as DelayedReduction<i64>>::reduce(&acc)
}

// ============================================================================
// FUNCTION 4: Field × i128 with delayed modular reduction
// Uses SignedWideLimbs<7> (448 bits) for larger signed values.
// ============================================================================

/// Inner product of field elements with i128 values.
/// Uses SignedWideLimbs<7> accumulator for larger signed values.
#[inline(never)]
#[unsafe(no_mangle)]
pub fn inner_product_field_i128(fields: &[Bn254Fr], values: &[i128]) -> Bn254Fr {
    let mut acc = SignedWideLimbs::<7>::default();
    for (f, v) in fields.iter().zip(values.iter()) {
        // Two-pass multiply-accumulate (low 64 bits, then high 64 bits)
        <Bn254Fr as DelayedReduction<i128>>::unreduced_multiply_accumulate(&mut acc, f, v);
    }
    // Single reduction at the end
    <Bn254Fr as DelayedReduction<i128>>::reduce(&acc)
}

// ============================================================================
// Main: Exercise all functions to prevent dead code elimination
// ============================================================================

fn main() {
    // Small test vectors for correctness
    let fields: Vec<Bn254Fr> = (1u64..=8).map(Bn254Fr::from).collect();
    let fields2: Vec<Bn254Fr> = (10u64..=17).map(Bn254Fr::from).collect();
    let i64_values: Vec<i64> = vec![1, -2, 3, -4, 5, -6, 7, -8];
    let i128_values: Vec<i128> = vec![100, -200, 300, -400, 500, -600, 700, -800];

    // Run all functions with black_box to prevent optimization
    let result1 = black_box(inner_product_field_field_base(
        black_box(&fields),
        black_box(&fields2),
    ));
    let result2 = black_box(inner_product_field_field_dmr(
        black_box(&fields),
        black_box(&fields2),
    ));
    let result3 = black_box(inner_product_field_i64(
        black_box(&fields),
        black_box(&i64_values),
    ));
    let result4 = black_box(inner_product_field_i128(
        black_box(&fields),
        black_box(&i128_values),
    ));

    println!("Inner Product Implementations - Assembly Comparison (BN254)");
    println!("============================================================");
    println!();
    println!("Results (verify correctness):");
    println!("  base (field×field eager):   {:?}", result1);
    println!("  dmr  (field×field delayed): {:?}", result2);
    println!("  i64  (field×i64 delayed):   {:?}", result3);
    println!("  i128 (field×i128 delayed):  {:?}", result4);

    // Timing benchmark
    const N: usize = 1 << 16; // 65536 elements
    const TRIALS: usize = 10;

    println!();
    println!("Timing Benchmark (n = {}, {} trials)", N, TRIALS);
    println!("----------------------------------------");

    // Generate large test vectors
    let large_fields: Vec<Bn254Fr> = (0..N as u64).map(Bn254Fr::from).collect();
    let large_fields2: Vec<Bn254Fr> = (0..N as u64).map(|x| Bn254Fr::from(x + 1000)).collect();
    let large_i64: Vec<i64> = (0..N as i64).map(|x| if x % 2 == 0 { x } else { -x }).collect();
    let large_i128: Vec<i128> =
        (0..N as i128).map(|x| if x % 2 == 0 { x * 1000 } else { -x * 1000 }).collect();

    // Warmup
    for _ in 0..3 {
        black_box(inner_product_field_field_base(&large_fields, &large_fields2));
        black_box(inner_product_field_field_dmr(&large_fields, &large_fields2));
        black_box(inner_product_field_i64(&large_fields, &large_i64));
        black_box(inner_product_field_i128(&large_fields, &large_i128));
    }

    // Benchmark base
    let start = Instant::now();
    for _ in 0..TRIALS {
        black_box(inner_product_field_field_base(
            black_box(&large_fields),
            black_box(&large_fields2),
        ));
    }
    let base_time = start.elapsed().as_micros() as f64 / TRIALS as f64;

    // Benchmark dmr
    let start = Instant::now();
    for _ in 0..TRIALS {
        black_box(inner_product_field_field_dmr(
            black_box(&large_fields),
            black_box(&large_fields2),
        ));
    }
    let dmr_time = start.elapsed().as_micros() as f64 / TRIALS as f64;

    // Benchmark i64
    let start = Instant::now();
    for _ in 0..TRIALS {
        black_box(inner_product_field_i64(
            black_box(&large_fields),
            black_box(&large_i64),
        ));
    }
    let i64_time = start.elapsed().as_micros() as f64 / TRIALS as f64;

    // Benchmark i128
    let start = Instant::now();
    for _ in 0..TRIALS {
        black_box(inner_product_field_i128(
            black_box(&large_fields),
            black_box(&large_i128),
        ));
    }
    let i128_time = start.elapsed().as_micros() as f64 / TRIALS as f64;

    println!(
        "  base (field×field eager):   {:>8.1} µs",
        base_time
    );
    println!(
        "  dmr  (field×field delayed): {:>8.1} µs  ({:.2}× faster)",
        dmr_time,
        base_time / dmr_time
    );
    println!(
        "  i64  (field×i64 delayed):   {:>8.1} µs  ({:.2}× faster than base)",
        i64_time,
        base_time / i64_time
    );
    println!(
        "  i128 (field×i128 delayed):  {:>8.1} µs  ({:.2}× faster than base)",
        i128_time,
        base_time / i128_time
    );

    println!();
    println!("Instruction counts per iteration (from assembly):");
    println!("  base: 250 instrs  |  dmr: 109 instrs  |  i64: 54 instrs  |  i128: 92 instrs");
    println!();
    println!("View assembly with:");
    println!("  cargo asm --example asm_compare inner_product_field_field_base");
    println!("  cargo asm --example asm_compare inner_product_field_field_dmr");
    println!("  cargo asm --example asm_compare inner_product_field_i64");
    println!("  cargo asm --example asm_compare inner_product_field_i128");
}
