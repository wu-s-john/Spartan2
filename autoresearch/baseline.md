# Baseline & Machine Context

## Machine

- **CPU**: Apple M4 (4 performance + 6 efficiency cores, 10 total)
- **RAM**: 16 GB
- **Arch**: aarch64-apple-darwin (ARM64)
- **OS**: macOS Darwin 25.2.0
- **Rust**: nightly, `.cargo/config.toml` sets `-Ctarget-cpu=native`
- **GPU**: None available for compute

## Benchmark

```bash
cargo run --release --example rs_shuffle_bp_full -- --zk
```

- Circuit: RS Shuffle + Re-encryption (Bellpepper)
- N = 52 cards, LEVELS = 6
- Curve: Pallas/Vesta cycle
- Constraints: 151,647 (padded to 262,144)
- Variables (rest): 153,980 (padded to 258,048)
- Hyrax commitment width: 1024 (DEFAULT_COMMITMENT_WIDTH)
- Hyrax rows: 252 MSMs of 1024 Pallas points each

## Baseline Results (5 runs, 2026-03-21)

### Spartan (non-ZK)

| Run | Prove  | Verify | commit_rest | outer_sc | inner_sc | pcs  |
|-----|--------|--------|-------------|----------|----------|------|
| 1   | 207ms  | 18ms   | 145ms       | 11ms     | 9ms      | 15ms |
| 2   | 202ms  | 18ms   | 149ms       | 9ms      | 7ms      | 15ms |
| 3   | 200ms  | 18ms   | 146ms       | 10ms     | 7ms      | 15ms |
| 4   | 195ms  | 17ms   | 143ms       | 10ms     | 6ms      | 15ms |
| 5   | 200ms  | 19ms   | 146ms       | 9ms      | 7ms      | 15ms |
| **Avg** | **201ms** | **18ms** | **146ms** | **10ms** | **7ms** | **15ms** |

### Spartan ZK

| Run | Prove  | Verify | commit_rest | outer_sc | inner_sc | pcs  | nifs |
|-----|--------|--------|-------------|----------|----------|------|------|
| 1   | 235ms  | 38ms   | 142ms       | 19ms     | 18ms     | 15ms | 16ms |
| 2   | 239ms  | 35ms   | 143ms       | 19ms     | 19ms     | 15ms | 18ms |
| 3   | 236ms  | 37ms   | 143ms       | 20ms     | 18ms     | 14ms | 16ms |
| 4   | 237ms  | 36ms   | 145ms       | 20ms     | 17ms     | 14ms | 16ms |
| 5   | 242ms  | 36ms   | 144ms       | 23ms     | 18ms     | 14ms | 16ms |
| **Avg** | **238ms** | **36ms** | **143ms** | **20ms** | **18ms** | **14ms** | **16ms** |

## Bottleneck Analysis

| Phase | Avg (ZK) | % of Prove | What It Does |
|-------|----------|------------|--------------|
| commit_rest | 143ms | 60% | 252 parallel Hyrax row MSMs (1024 Pallas points each) |
| outer_sc | 20ms | 8% | Cubic sumcheck with delayed modular reduction |
| inner_sc | 18ms | 8% | Quadratic sumcheck with delayed modular reduction |
| nifs | 16ms | 7% | ZK blinding via NIFS folding |
| pcs | 14ms | 6% | Hyrax evaluation proof (IPA + 1 MSM) |
| other | 27ms | 11% | mat_vec, eval_rx, eval_sparse, synthesis, etc. |

## Current Algorithms

- **MSM**: Signed scalar decomposition → bit-width routing → XYZZ Pippenger (serial per row, rayon across rows)
- **Hyrax PCS**: Row-parallel commit (width=1024), Eq-tree evaluation, linear IPA
- **Sumcheck**: Delayed modular reduction with wide-limb accumulators, rayon fold/reduce
- **Field arithmetic**: 5×52 column limbs with u128 accumulation (custom, not halo2curves)
- **NIFS**: Standard rerandomization for ZK property
