# NeutronNova Benchmark: main vs benching/sha256-is-faster

## NeutronNova ZK (instances=16, chain=64)

| Metric | main | optimized | Δ | Speedup |
|--------|------|-----------|---|---------|
| **prove** | 5940ms | 5179ms | -761ms | **1.15x** |
| setup | 9077ms | 9384ms | +307ms | 0.97x |
| prep | 2384ms | 2288ms | -96ms | 1.04x |
| verify | 622ms | 632ms | +10ms | 0.98x |

| Phase | main | optimized | Δ | Speedup |
|-------|------|-----------|---|---------|
| nifs_sc | 2042ms | 1987ms | -55ms | 1.03x |
| **outer_sc** | 849ms | 251ms | -598ms | **3.38x** |
| **inner_sc** | 411ms | 171ms | -240ms | **2.40x** |
| fold_W | 123ms | 110ms | -13ms | 1.12x |
| fold_U | 207ms | 206ms | -1ms | 1.00x |
| precom_syn | 16692ms | 16873ms | +181ms | 0.99x |
| commit_pre | 9705ms | 9926ms | +221ms | 0.98x |
| rerand | 625ms | 671ms | +46ms | 0.93x |
| gen_inst | 210ms | 217ms | +7ms | 0.97x |
| pcs | 44ms | 40ms | -4ms | 1.10x |

---

## NeutronNova ZK (instances=8, chain=128)

| Metric | main | optimized | Δ | Speedup |
|--------|------|-----------|---|---------|
| **prove** | 7738ms | 5885ms | -1853ms | **1.31x** |
| setup | 19337ms | 19373ms | +36ms | 1.00x |
| prep | 2975ms | 2649ms | -326ms | 1.12x |
| verify | 954ms | 959ms | +5ms | 1.00x |

| Phase | main | optimized | Δ | Speedup |
|-------|------|-----------|---|---------|
| nifs_sc | 1780ms | 1805ms | +25ms | 0.99x |
| **outer_sc** | 1737ms | 504ms | -1233ms | **3.45x** |
| **inner_sc** | 819ms | 296ms | -523ms | **2.77x** |
| fold_W | 134ms | 113ms | -21ms | 1.19x |
| fold_U | 313ms | 277ms | -36ms | 1.13x |
| precom_syn | 16022ms | 15406ms | -616ms | 1.04x |
| commit_pre | 8673ms | 8894ms | +221ms | 0.98x |
| rerand | 723ms | 688ms | -35ms | 1.05x |
| gen_inst | 243ms | 226ms | -17ms | 1.08x |
| pcs | 72ms | 78ms | +6ms | 0.92x |

---

## Analysis

### What's faster

**Spartan ZK sumcheck (outer_sc, inner_sc): 2.4-3.5x faster**

The optimized branch applies Delayed Modular Reduction (DMR) to the sumcheck functions in `src/sumcheck.rs`. Instead of doing a Montgomery reduction after every field multiplication, DMR accumulates products in wide limbs and reduces once at the end.

For N multiplications:
- Main: N reductions (one per multiply)
- Optimized: 1 reduction (at the end)

This saves ~30-40% per multiply, compounding to 2-3x overall speedup.

### What's unchanged

**NIFS sumcheck (nifs_sc): ~1.0x (no change)**

The NIFS folding rounds have two main costs:
1. `prove_helper` (~40%) - Already uses DMR in both branches
2. Layer folding (~60%) - Cannot benefit from DMR

Layer folding does one multiply per element and writes back immediately:
```rust
*l += (*h - *l) * r_b;  // single multiply, no accumulation possible
```

DMR requires accumulating multiple products before reducing. Single multiplies with immediate writeback cannot benefit.

### Potential future optimization

The layer folding currently runs every round:
```rust
for t in 0..ell_b {
  // ... sumcheck ...
  // Fold layers for next round - O(pairs × witness_size)
  for matrix_layer in [&mut A_layers, &mut B_layers, &mut C_layers] {
    low.iter_mut().zip(high.iter()).for_each(|(lo, hi)| {
      *l += (*h - *l) * r_b;
    });
  }
}
```

Total: O(n × witness_size × ell_b) with ell_b passes over memory.

Alternative: Defer folding to the end using one eq polynomial evaluation:
```rust
// One pass at the end
A_final[k] = Σ_i eq(r_bs, i) × A_original[i][k]
```

Total: O(n × witness_size) with 1 pass over memory.

This could improve cache locality and reduce total memory bandwidth.
