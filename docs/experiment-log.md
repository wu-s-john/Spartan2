# Experiment Log: AARCH64 Field Arithmetic

## Run Environment
- Machine: Apple M4 (10 cores, 16GB RAM)
- Rust: rustc 1.96.0-nightly (0c68443b0 2026-03-10)
- Date: 2026-03-11
- Compiler flags: `-Ctarget-cpu=native`, opt-level=3, lto=thin, codegen-units=1

## 1. Baseline Assembly Analysis

### mac_ff_52 — 5×52 column accumulator (WINNER)

**Hot loop**: 50 `mul`/`umulh` instructions (25 products × 2), ~50 `adds`/`adc` pairs for u128 addition.

**Register allocation**: 18 GP registers for 9 u128 accumulators + 10 for input limbs = 28 total needed. Only 2 stack spills (cols[6] and cols[7] high halves). Excellent for 31 available GP registers minus x0/x29/x30.

**Carry handling**: Clean `adds`/`adc` pairs — no LLVM carrying_add bug (#118162). No `cinc` overhead.

**Stride**: 80 bytes per iteration (2 × [u64; 5] = 10 × 8 bytes). Sequential access pattern.

**Verdict**: OPTIMAL — LLVM generated near-perfect code. No missed optimizations.

### mac_4x4_into_fused — 4×64 fused (2nd place)

**Hot loop**: 16 `mul`/`umulh` pairs + extensive carry chain with `cinc` (conditional increment).

**Issue**: Each column accumulation requires a serial carry chain through adds/cinc, creating a long dependency chain. The u128 carry propagation is the bottleneck — each `cinc` depends on the previous `adds` flag.

**Verdict**: GOOD codegen but inherently limited by serial carry dependency.

### mac_4x1_into_asm — 4×64 inline asm for field×i64

**Hot loop**: 4 `mul`/`umulh` pairs with `adds`/`adcs` carry chain.

**Issue**: Inline asm prevents LLVM from optimizing the surrounding loop (can't hoist loads, schedule across iterations). The Rust `mac()` chain compiles to better overall code because LLVM can interleave operations.

**Verdict**: WORSE than Rust — inline asm is a net negative here.

## 2. Wall-Clock Results

### Field×Field (ns/op)

| N | 4x64_current | 4x64_fused | 4x64_asm | 5x52_lazy | 8x32_neon |
|---|---|---|---|---|---|
| 1024 | 12.00 | 11.96 | 11.84 | **9.48** | 38.21 |
| 16384 | 6.57 | 6.24 | 6.24 | **4.87** | 20.20 |
| 262144 | 5.32 | 5.42 | 5.42 | **4.25** | 17.46 |
| 1048576 | 5.42 | 5.41 | 5.44 | **4.23** | 17.45 |

**Winner**: 5×52 lazy — **1.28× speedup** over baseline

### Field×i64 (ns/op)

| N | 4x64_current | 4x64_asm | 5x52_lazy | 8x32_scalar |
|---|---|---|---|---|
| 1024 | 1.22 | 2.77 | **0.94** | 2.28 |
| 16384 | 1.10 | 2.75 | **0.93** | 2.19 |
| 262144 | 1.24 | 2.66 | **0.94** | 2.24 |
| 1048576 | 1.22 | 2.62 | **0.95** | 2.24 |

**Winner**: 5×52 lazy — **1.28× speedup**

### Field×i128 (ns/op)

| N | 4x64_current | 4x64_asm | 5x52_lazy | 8x32_scalar |
|---|---|---|---|---|
| 1024 | 2.56 | 4.56 | **1.79** | 4.56 |
| 16384 | 2.28 | 3.42 | **1.72** | 4.36 |
| 262144 | 2.51 | 3.39 | **1.75** | 4.42 |
| 1048576 | 2.47 | 3.44 | **1.73** | 4.44 |

**Winner**: 5×52 lazy — **1.43× speedup**

### Field×i32 (ns/op)

| N | 4x64_current | 5x52_lazy | 8x32_scalar |
|---|---|---|---|
| 1024 | 0.90 | **0.85** | 1.06 |
| 16384 | 0.97 | **0.92** | 1.17 |
| 262144 | 0.99 | **0.94** | 1.20 |
| 1048576 | 0.98 | **0.93** | 1.18 |

**Winner**: 5×52 lazy — **1.05× speedup**

### ILP Batching — Field×Field (ns/op)

| N | cur_K1 | cur_K2 | cur_K4 | fused_K1 | fused_K2 | fused_K4 |
|---|---|---|---|---|---|---|
| 1024 | 5.33 | 4.88 | 5.33 | 5.45 | 5.70 | 5.37 |
| 16384 | 5.36 | 5.33 | 5.37 | 5.30 | 5.77 | 5.47 |
| 262144 | 5.33 | 5.36 | 5.43 | 5.41 | 5.76 | 6.02 |
| 1048576 | 5.39 | 5.31 | 5.44 | 5.29 | 5.79 | 6.21 |

**Observation**: ILP batching provides no benefit. Apple M4 already extracts sufficient ILP from single accumulator updates.

### Reduction Operations (ns/op, N=1M)

| Operation | ns/op |
|---|---|
| montgomery_reduce_9 | 21.70 |
| barrett_reduce_6 | 5.20 |
| barrett_reduce_7 | 8.45 |
| 52_carry_prop + mont_reduce_9 | 24.31 |
| 32_carry_prop + mont_reduce_9 | 26.79 |

**Observation**: 52-bit carry propagation adds only ~2.6ns to Montgomery reduction. This cost is amortized over N MACs (e.g., at N=1M, it's 0.0000026 ns/MAC — negligible).

## 3. Micro-Optimization Attempts on 5×52 Winner

### Attempt 1: Explicit local variables (mac_ff_52_locals)
Flatten `cols[0..9]` array into 9 explicit `&mut u128` parameters.

| N | 52_array | 52_locals | Delta |
|---|---|---|---|
| 1048576 | 4.20 | 4.20 | **0% — identical** |

**Verdict**: LLVM already maps array elements to registers. No improvement.

### Attempt 2: K=2 ILP batching for 5×52
Two independent `[u128; 9]` accumulator sets.

| N | 52_array | 52_K2 | Delta |
|---|---|---|---|
| 1048576 | 4.20 | 4.36 | **-4% — worse** |

**Verdict**: Register pressure (36 regs needed for 2 sets) causes spills. Regression.

### Attempt 3: Manual unroll by 2
Process 2 elements per loop iteration with single accumulator.

| N | 52_array | 52_unroll2 | Delta |
|---|---|---|---|
| 1048576 | 4.20 | 4.68 | **-11% — worse** |

**Verdict**: Code bloat increases instruction cache pressure. Regression.

### Convergence
3 consecutive attempts with ≤0% improvement. **Optimization converged.**

## 4. Key Findings

### Why 5×52 wins

1. **Zero-carry accumulation**: Products accumulate into u128 columns without any carry propagation during the hot loop. The carry chain is deferred to a single pass at reduction time.

2. **Maximum ILP**: Each column update (`cols[k] += a[i]*b[j]`) is independent of other columns. The CPU can execute multiple column updates in parallel across its execution units.

3. **Optimal instruction sequence**: `mul` + `umulh` produce a 128-bit product, then a single `adds`/`adc` pair accumulates into the u128 column. No serial carry dependency across columns.

4. **Contrast with 4×64**: The 4×64 representation requires carry propagation after each product addition, creating a serial dependency chain through the accumulator limbs. This serializes the MAC operation.

### Why 8×32 NEON lost badly

1. **No 64×64→128 NEON multiply**: NEON's `vmull_u32` only does 32×32→64, requiring 4× more multiplies than 5×52's 52×52→104.

2. **NEON→scalar transfer cost**: Moving results from NEON vector registers to GP scalar registers incurs latency penalties on Apple Silicon.

3. **More columns**: 8×8 schoolbook produces 15 columns vs 5×5's 9 columns — more memory traffic and carry propagation overhead.

### Why inline ASM lost

1. **Compiler barrier**: Inline asm creates an optimization barrier — LLVM can't hoist invariants, schedule across the boundary, or combine with surrounding code.

2. **Already-good Rust codegen**: For these specific kernels, LLVM generates near-optimal instruction sequences from idiomatic Rust. The asm provides no additional benefit.

## FINAL VERDICT

**Best method**: 5×52 column-accumulator (`mac_ff_52`, `mac_fi64_52`, `mac_fi128_52`)

**Speedups over baseline**:
- Field×Field: **1.28×**
- Field×i64: **1.28×**
- Field×i128: **1.43×**
- Field×i32: **1.05×**

**Recommended next step**: Wire `limbs52.rs` kernels into `delayed_reduction.rs` production code. The conversion (4×64 → 5×52) should happen once when field elements enter the accumulation loop, and carry propagation + reduction happens once at the end.
