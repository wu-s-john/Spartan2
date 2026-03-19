# PCS Performance Regression Investigation & Fix

## Context

PCS prove takes 23ms for 128B SHA256 (262144 constraints, 131072 vars). It used to be ~10ms. The regression traces to commit `ecced9a` ("Use wider columns in Hyrax") which changed `num_cols` from `sqrt(n)` (=256 for 2^17 vars) to a hardcoded 1024.

This 4x increase in column width made the two MSMs in PCS prove (commit-LZ and IPA-delta) 4x larger (256 → 1024 elements). Both run **single-threaded** due to the threshold `coeffs.len() > 1024` at `msm.rs:140` being strict (1024 is NOT > 1024).

### Cost breakdown (current)
| Step | Size | Time | Why |
|------|------|------|-----|
| hyrax_prove_commit (LZ MSM) | 1024 full-scalar | 10ms | Serial Pippenger, 37 windows × 127 buckets |
| hyrax_prove_ipa (d_vec MSM) | 1024 full-scalar | 11ms | Serial Pippenger + random gen |
| prep + bind + overhead | - | 2ms | |
| **Total PCS** | | **23ms** | |

### Why it was ~10ms before
Before `ecced9a`, `num_cols = 2^(ceil(17/2)) = 256`. Two 256-element MSMs at ~3ms each ≈ 7-8ms total.

## Fix: Enable parallelism for 1024-element MSMs

### Change 1: `src/provider/msm.rs` line 140
```rust
// Before:
let num_threads = if coeffs.len() > 1024 && use_parallelism_internally {

// After:
let num_threads = if coeffs.len() >= 1024 && use_parallelism_internally {
```

This enables parallel MSM for exactly 1024 elements. With ~10 cores, each thread handles ~100 elements → ~1-2ms per MSM → PCS ≈ 5-6ms.

### Change 2 (consistency): `src/provider/msm.rs` line 275 (msm_bool)
Same off-by-one: `bits.len() > 1024` → `bits.len() >= 1024`. Not on PCS critical path but affects bool commits.

## Files to modify
- `src/provider/msm.rs` — lines 140, 275: change `>` to `>=`

## Verification
1. `cargo test` — ensure correctness
2. Run the SHA256 bench: expect `pcs_prove` to drop from ~23ms to ~5-6ms
3. Check that `commit_witness_precommitted` (which uses parallel MSMs over rows) doesn't regress — it won't because those MSMs have `use_parallelism_internally: false` (row-level parallelism is already external)
