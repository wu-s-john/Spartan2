# Spartan Benchmark: main vs benching/sha256-is-faster

## SHA256 Single Hash (msg=1024B)

| Metric | main | optimized | Δ | Speedup |
|--------|------|-----------|---|---------|
| **prove_total** | 184ms | 168ms | -16ms | **1.10x** |
| setup | 1318ms | 1261ms | -57ms | 1.05x |
| prep | 125ms | 109ms | -16ms | 1.15x |
| verify | 52ms | 47ms | -5ms | 1.11x |

| Phase | main | optimized | Δ | Speedup |
|-------|------|-----------|---|---------|
| **outer_sc** | 45ms | 33ms | -12ms | **1.36x** |
| **eval_rx** | 5ms | 11ms | +6ms | 0.45x |
| **inner_sc** | 37ms | 33ms | -4ms | **1.12x** |
| mat_vec | 10ms | 9ms | -1ms | 1.11x |
| eval_sparse | 30ms | 30ms | 0ms | 1.00x |
| poly_ABC | 12ms | 10ms | -2ms | 1.20x |
| poly_z | 1ms | 1ms | 0ms | 1.00x |
| pcs | 22ms | 20ms | -2ms | 1.10x |
| synth_pre | 123ms | 107ms | -16ms | 1.15x |
| commit_pre | 27ms | 18ms | -9ms | 1.50x |

---

## SHA256 Chain (msg=32B, chain=1028)

| Metric | main | optimized | Δ | Speedup |
|--------|------|-----------|---|---------|
| **prove_total** | 8770ms | 8751ms | -19ms | 1.00x |
| setup | 144807ms | 144832ms | +25ms | - |
| prep | 6281ms | 6850ms | +569ms | 0.92x |
| verify | 1973ms | 2024ms | +51ms | 0.98x |

| Phase | main | optimized | Δ | Speedup |
|-------|------|-----------|---|---------|
| **outer_sc** | 1294ms | 1048ms | -246ms | **1.23x** |
| **eval_rx** | 415ms | 244ms | -171ms | **1.70x** |
| **inner_sc** | 1221ms | 1101ms | -120ms | **1.11x** |
| mat_vec | 974ms | 995ms | +21ms | 0.98x |
| eval_sparse | 2378ms | 2381ms | +3ms | 1.00x |
| poly_ABC | 726ms | 865ms | +139ms | 0.84x |
| poly_z | 150ms | 226ms | +76ms | 0.66x |
| pcs | 546ms | 641ms | +95ms | 0.85x |
| synth_pre | 6146ms | 6709ms | +563ms | 0.92x |
| commit_pre | 1183ms | 1188ms | +5ms | 1.00x |

---

## Summary

**Consistent improvements across both benchmarks:**
- `outer_sc`: 23-36% faster
- `inner_sc`: 11-12% faster

**Small benchmark (1024B single hash):**
- Overall **10% faster** proving time (184ms → 168ms)
- Improvements across most phases

**Large benchmark (1028-chain):**
- Sumcheck optimizations save ~537ms
- Regressions in `synth_pre`, `poly_ABC`, `poly_z`, `pcs` offset gains
- Net effect: ~0% change in total proving time
