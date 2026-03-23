# Baseline — Spartan ZK Prove (Post-CPU Optimization)

## Machine

- **CPU**: Apple M4 (4P + 6E cores, 10 total)
- **GPU**: Apple M4 10-core Metal GPU (~2.9 TFLOPS FP32, unified memory)
- **RAM**: 16 GB unified
- **Arch**: aarch64-apple-darwin (ARM64)
- **OS**: macOS Darwin 25.2.0
- **Rust**: nightly, `-Ctarget-cpu=native`, `lto = "fat"`, `codegen-units = 1`

## Benchmark

```bash
cargo run --release --example rs_shuffle_bp_full -- --zk
```

- Circuit: RS Shuffle + Re-encryption (Bellpepper)
- N = 52 cards, LEVELS = 6
- Curve: Pallas/Vesta cycle
- Constraints: 151,647 (padded to 262,144)
- Variables (rest): 153,980 (padded to 253,952)
- Hyrax commitment width: 8192 (`DEFAULT_COMMITMENT_WIDTH`)
- Hyrax rows: 31 MSMs of 8192 Pallas points each (~19 non-zero)

## Baseline Results (3 runs, 2026-03-22)

| Run | Total Prove | commit_rest | outer_sc | inner_sc | pcs  | nifs |
|-----|-------------|-------------|----------|----------|------|------|
| 1   | 180ms       | 97ms        | 15ms     | 14ms     | 20ms | 3ms  |
| 2   | 183ms       | 98ms        | 16ms     | 14ms     | 20ms | 4ms  |
| 3   | 183ms       | 97ms        | 16ms     | 16ms     | 21ms | 3ms  |
| **Median** | **183ms** | **97ms** | **16ms** | **14ms** | **20ms** | **3ms** |

## Bottleneck Analysis

| Phase | Median | % of Prove | What It Does |
|-------|--------|------------|--------------|
| commit_rest | 97ms | **53%** | 19 parallel Hyrax row MSMs (8192 Pallas points each) |
| pcs | 20ms | 11% | Hyrax evaluation proof (IPA + 1 standalone MSM) |
| outer_sc | 16ms | 9% | Cubic sumcheck with delayed modular reduction |
| inner_sc | 14ms | 8% | Quadratic sumcheck with delayed modular reduction |
| nifs | 3ms | 2% | ZK blinding via NIFS folding (pipelined with eval_sparse) |
| other | 33ms | 18% | mat_vec, eval_rx, eval_sparse, synthesis, etc. |

## CPU Optimizations Already Applied

1. MSM: transposed signed digit layout (cache-friendly window-major access)
2. MSM: XYZZ accumulator (single inversion at end instead of per-window)
3. MSM: batch affine bucket accumulation (Montgomery batch inversion)
4. Outer sumcheck: delayed modular reduction
5. Sumcheck: fused 2/3-poly binding (single Rayon dispatch)
6. Commitment width: 1024 → 8192 (fewer rows, better Pippenger amortization)
7. Standalone parallel MSM for PCS (halo2curves msm_best)
8. NIFS pipelining (random instance overlapped with eval_sparse)

## The Gap

Target: **<100ms**. Current: **183ms**. Need: **83ms reduction (45%)**.

The commit_rest MSM at 97ms is the primary target. It consists of ~19 non-zero MSMs of 8192 full-field-element Pallas points. GPU acceleration of these MSMs is the most promising path to closing the gap.
