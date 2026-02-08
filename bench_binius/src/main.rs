//! Benchmark comparison: Binius binary fields vs BN254 prime field
//!
//! This benchmark compares inner product throughput between:
//! 1. BN254 prime field (254-bit) - standard field multiplication
//! 2. Binius BinaryField128b (128-bit binary tower field)
//!
//! Run: cargo run --release

use binius_field::{BinaryField128b, Field as BiniusField};
use halo2curves::bn256::Fr as Bn254Fr;
use halo2curves::ff::Field;
use std::hint::black_box;
use std::time::Instant;

// ============================================================================
// BN254 Prime Field Inner Product (standard - no DMR)
// ============================================================================

/// Inner product for BN254 with standard Montgomery multiplication.
/// Each multiply does a full Montgomery reduction.
#[inline(never)]
#[unsafe(no_mangle)]
pub fn inner_product_bn254(a: &[Bn254Fr], b: &[Bn254Fr]) -> Bn254Fr {
    let mut acc = Bn254Fr::ZERO;
    for (ai, bi) in a.iter().zip(b.iter()) {
        acc += *ai * *bi;
    }
    acc
}

// ============================================================================
// Binius BinaryField128b Inner Product
// ============================================================================

/// Inner product for Binius 128-bit binary tower field.
/// Uses native XOR for addition and carry-less multiplication.
#[inline(never)]
#[unsafe(no_mangle)]
pub fn inner_product_binius_128b(a: &[BinaryField128b], b: &[BinaryField128b]) -> BinaryField128b {
    let mut acc = BinaryField128b::ZERO;
    for (ai, bi) in a.iter().zip(b.iter()) {
        acc += *ai * *bi;
    }
    acc
}

// ============================================================================
// Main: Benchmark comparison
// ============================================================================

fn main() {
    println!("Inner Product Benchmark: BN254 Prime Field vs Binius Binary Field");
    println!("==================================================================");
    println!();
    println!("Hardware: ARM64 (Apple Silicon)");
    println!("Note: Binius is optimized for x86 with GFNI/CLMUL instructions.");
    println!("      On ARM, it falls back to software implementations.");
    println!();

    const N: usize = 1 << 16; // 65536 elements
    const TRIALS: usize = 10;

    // Generate test vectors for BN254
    let bn254_a: Vec<Bn254Fr> = (0..N as u64).map(Bn254Fr::from).collect();
    let bn254_b: Vec<Bn254Fr> = (0..N as u64).map(|x| Bn254Fr::from(x + 1000)).collect();

    // Generate test vectors for Binius
    let binius_a: Vec<BinaryField128b> = (0..N as u128)
        .map(|x| BinaryField128b::new(x))
        .collect();
    let binius_b: Vec<BinaryField128b> = (0..N as u128)
        .map(|x| BinaryField128b::new(x + 1000))
        .collect();

    // Warmup
    for _ in 0..3 {
        black_box(inner_product_bn254(&bn254_a, &bn254_b));
        black_box(inner_product_binius_128b(&binius_a, &binius_b));
    }

    // Benchmark BN254
    let start = Instant::now();
    for _ in 0..TRIALS {
        black_box(inner_product_bn254(
            black_box(&bn254_a),
            black_box(&bn254_b),
        ));
    }
    let bn254_time = start.elapsed().as_micros() as f64 / TRIALS as f64;

    // Benchmark Binius
    let start = Instant::now();
    for _ in 0..TRIALS {
        black_box(inner_product_binius_128b(
            black_box(&binius_a),
            black_box(&binius_b),
        ));
    }
    let binius_time = start.elapsed().as_micros() as f64 / TRIALS as f64;

    println!("Timing Benchmark (n = {}, {} trials)", N, TRIALS);
    println!("----------------------------------------");
    println!(
        "  BN254 (254-bit prime):         {:>8.1} µs",
        bn254_time
    );
    println!(
        "  Binius (128-bit binary):       {:>8.1} µs",
        binius_time
    );
    println!();

    if binius_time < bn254_time {
        println!(
            "  Binius is {:.2}× faster than BN254 on ARM",
            bn254_time / binius_time
        );
    } else {
        println!(
            "  BN254 is {:.2}× faster than Binius on ARM",
            binius_time / bn254_time
        );
    }

    println!();
    println!("Field Characteristics:");
    println!("  BN254:  254-bit prime field, Montgomery representation");
    println!("          4 limbs × 64-bit, requires modular reduction");
    println!("  Binius: 128-bit binary tower field (F_2 extensions)");
    println!("          Native XOR for addition, CLMUL for multiplication");
    println!();
    println!("Note: On x86 with GFNI/CLMUL, Binius would be significantly faster.");
    println!("      This benchmark shows ARM fallback performance.");
    println!();
    println!("View assembly with:");
    println!("  cargo asm inner_product_bn254");
    println!("  cargo asm inner_product_binius_128b");
}
