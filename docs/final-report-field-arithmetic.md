# Final Report: Optimal Field Arithmetic for BN254 on Apple Silicon

## Executive Summary

- **Best method**: 5×52-bit column accumulators with lazy carry propagation
- **Speedup over baseline**: 1.28× for Field×Field, 1.28× for Field×i64, 1.43× for Field×i128, 1.05× for Field×i32
- **Why this is optimal**: Eliminates the serial carry chain that bottlenecks the 4×64 representation. Each u128 column accumulates independently, maximizing instruction-level parallelism. LLVM generates near-perfect register allocation (2 spills out of 28 registers needed). Three micro-optimization attempts produced 0% or negative gains, confirming convergence.

## Methodology

### What We Tested

5 representations × 4 MAC types × 4 sizes = 80+ measurement points:

| Representation | MAC types | Description |
|---|---|---|
| 4×64 current | ff, fi64, fi128, fi32 | Baseline: `mul_4_by_4` + accumulate with serial carry chain |
| 4×64 fused | ff | Fused MAC (no intermediate array), still serial carry |
| 4×64 inline ASM | ff, fi64, fi128 | Hand-written `mul`/`umulh`/`adds`/`adcs` |
| 5×52 lazy carry | ff, fi64, fi128, fi32 | Column accumulators `[u128; N]`, carry-free during accumulation |
| 8×32 NEON | ff, fi64, fi128, fi32 | `vmull_u32` widening multiply, column accumulators |

Plus: ILP batching (K=1,2,4) for 4×64; K=2 for 5×52; reduction benchmarks; 3 micro-optimization variants.

### How We Measured

- Wall-clock timing with `std::time::Instant`
- Median of 10 trials after 3 warmup iterations
- `black_box()` to prevent dead code elimination
- Sizes: N = 1024, 16384, 262144, 1048576
- Correctness verified: all variants produce identical results to reference implementation

### Machine Specs

- Apple M4, 10 cores, 16GB RAM
- rustc 1.96.0-nightly (0c68443b0 2026-03-10)
- `-Ctarget-cpu=native`, opt-level=3, lto=thin, codegen-units=1

## Round 1 Results: Initial Sweep

### Field×Field (ns/op at N=1M)

| Variant | ns/op | vs baseline |
|---|---|---|
| naive (halo2curves mul+add) | 14.12 | 0.38× |
| 4x64_current | 5.42 | 1.00× |
| 4x64_fused | 5.41 | 1.00× |
| 4x64_asm | 5.44 | 1.00× |
| **5x52_lazy** | **4.23** | **1.28×** |
| 8x32_neon | 17.45 | 0.31× |

### Field×i64 (ns/op at N=1M)

| Variant | ns/op | vs baseline |
|---|---|---|
| naive (halo2curves mul+add) | 13.82 | 0.09× |
| 4x64_current | 1.22 | 1.00× |
| 4x64_asm | 2.62 | 0.47× |
| **5x52_lazy** | **0.95** | **1.28×** |
| 8x32_scalar | 2.24 | 0.54× |

### Field×i128 (ns/op at N=1M)

| Variant | ns/op | vs baseline |
|---|---|---|
| 4x64_current | 2.47 | 1.00× |
| 4x64_asm | 3.44 | 0.72× |
| **5x52_lazy** | **1.73** | **1.43×** |
| 8x32_scalar | 4.44 | 0.56× |

### Field×i32 (ns/op at N=1M)

| Variant | ns/op | vs baseline |
|---|---|---|
| 4x64_current | 0.98 | 1.00× |
| **5x52_lazy** | **0.93** | **1.05×** |
| 8x32_scalar | 1.18 | 0.83× |

## Round 2 Results: Micro-Optimization of 5×52 Winner

### Attempt 1: Explicit local variables
Replace `cols: &mut [u128; 9]` with 9 individual `&mut u128` parameters.

**Result**: 4.20 ns/op → 4.20 ns/op = **0% change**

**Why**: LLVM already promotes array elements to registers. Assembly inspection confirms identical codegen.

### Attempt 2: K=2 ILP batching
Two independent `[u128; 9]` accumulator sets, interleaved processing.

**Result**: 4.20 ns/op → 4.36 ns/op = **-4% regression**

**Why**: 18 registers per accumulator × 2 sets = 36 registers needed, exceeding the 28 usable GP registers. Additional spills negate any ILP benefit.

### Attempt 3: Manual unroll by 2
Process 2 elements per loop iteration, single accumulator.

**Result**: 4.20 ns/op → 4.68 ns/op = **-11% regression**

**Why**: Doubled loop body increases instruction cache pressure without reducing loop overhead (which is already <1% of total time).

**Convergence confirmed**: 3 consecutive attempts with ≤0% improvement.

## Assembly Analysis: Comparative Method Breakdown

### Summary Comparison Table

All methods side by side, instruction counts per inner loop iteration (one MAC):

| Metric | Naive | 4×64 current | 4×64 fused | **5×52 lazy** | fi64 4×64 | fi64 5×52 |
|---|---|---|---|---|---|---|
| `mul` | 36 | 16 | 16 | **25** | 4 | 5 |
| `umulh` | 32 | 16 | 16 | **25** | 4 | 5 |
| `adds` (independent) | 57 | 32 | 38 | **25** | 8 | 5 |
| `adc` (no flags set) | 8 | 1 | 0 | **25** | 0 | **5** |
| `adcs` (serial carry!) | **24** | **7** | 0 | 0 | 0 | 0 |
| `cinc` (carry capture) | 50 | 24 | 32 | 0 | 8 | 0 |
| `ldp`/`ldr` (loads) | 4 | 4 | 5 | 8 | 3 | 6 |
| `stp`/`str` [sp] (spills) | 0 | 0 | 0 | **1** | 3 | **5** |
| **Total ALU ops** | **207** | **96** | **102** | **100** | **24** | **20** |
| **Serial carry ops** | **24** | **7** | **0** | **0** | **0** | **0** |
| **GP regs used** | 28 | 25 | 19 | **28** | 13 | 15 |
| **Critical path depth** | Deep | Medium | Low | **Lowest** | Low | Low |

Key insight: **adcs count is the bottleneck predictor**. The 5×52 method has zero serial carry operations — every column accumulates independently.

### Per-Method Detail

#### Naive (halo2curves `a * b`, ~207 ALU ops/iter)

Full Montgomery multiply + REDC per element. 36 mul/32 umulh for 4×4 schoolbook + 4 REDC rounds.
24 serial `adcs` in the carry chains create deep dependency. Uses all 28 GP registers, 0 spills.

```asm
; Montgomery REDC round — serial carry chain bottleneck
mul   x6, x4, x3              ; q = r[i] * MONT_INV
umulh x5, x6, x7              ; hi(q * MODULUS[0])
adds  x8, x8, x9              ; accumulate
adcs  x10, x10, x5            ; SERIAL: waits on previous adds
adcs  x11, x11, x12           ; SERIAL: waits on previous adcs
adcs  x13, x13, x14           ; SERIAL: waits on previous adcs
```

**Why slowest**: Does full reduction per MAC. 24 serial carry dependencies × 1 cycle/dep = 24-cycle serial chain in each iteration.

#### 4×64 current (mul_4_by_4 + accumulate, 96 ALU ops/iter)

Separate multiply phase (16 mul + 16 umulh) + 8-limb carry-chain accumulate.

```asm
; Accumulation phase — 7 serial adcs
adds  x8, x17, x8             ; acc[0] += product[0]
adcs  x13, x7, x13            ; acc[1] += product[1] + carry  (SERIAL)
adcs  x14, x1, x14            ; acc[2] += product[2] + carry  (SERIAL)
adcs  x15, x6, x15            ; acc[3] += product[3] + carry  (SERIAL)
adcs  x16, x5, x16            ; acc[4] += product[4] + carry  (SERIAL)
adcs  x17, x3, x17            ; acc[5] += product[5] + carry  (SERIAL)
adcs  x2, x4, x2              ; acc[6] += product[6] + carry  (SERIAL)
```

**Why medium**: 7 serial `adcs` create a 7-cycle dependency chain per iteration. Multiplies are independent but additions serialize.

#### 4×64 fused (mac_4x4_into_fused, 102 ALU ops/iter)

Eliminates carry chain by using `cinc` (conditional increment) instead of `adcs`.

```asm
umulh x16, x14, x6            ; hi(b[0] * a[0])
mul   x17, x14, x6            ; lo(b[0] * a[0])
adds  x8, x17, x8             ; acc[0] += lo
cinc  x16, x16, hs            ; carry = hi + (CF ? 1 : 0) — NO serial dep
umulh x17, x15, x6            ; hi(b[1] * a[0])
mul   x7, x15, x6             ; lo(b[1] * a[0])
adds  x13, x7, x13            ; acc[1] += lo
cinc  x17, x17, hs            ; carry capture — independent
adds  x13, x13, x16           ; acc[1] += carry_from_acc[0]
cinc  x7, x17, hs             ; propagate
```

**Why same speed as current**: Removes serial adcs (0 vs 7) but introduces 5 `b.lo` branches for overflow detection. Branch overhead offsets the ILP gain. Uses only 19 registers — lighter pressure but more control flow.

#### 5×52 lazy (mac_ff_52, 100 ALU ops/iter) — **WINNER**

25 fully independent column accumulations. Each is `mul + umulh + adds + adc` — the `adds`/`adc` pair accumulates a u128 product into a u128 column with **no carry chain between columns**.

```asm
ldp   x22, x21, [x0]          ; load a[0], a[1]
ldp   x23, x24, [x0, #32]     ; load b[0], b[1]
umulh x1, x24, x22             ; hi(b[1] * a[0])
mul   x25, x24, x22            ; lo(b[1] * a[0])
adds  x10, x25, x10            ; cols[1].lo += lo  (independent!)
adc   x11, x1, x11             ; cols[1].hi += hi  (independent!)
umulh x1, x27, x22             ; hi(b[2] * a[0])
mul   x26, x27, x22            ; lo(b[2] * a[0])
umulh x28, x21, x24            ; hi(a[1] * b[1])  — can run in parallel!
mul   x30, x21, x24            ; lo(a[1] * b[1])  — can run in parallel!
adds  x2, x26, x2              ; cols[2].lo += lo(b[2]*a[0])
adc   x1, x1, x20             ; cols[2].hi += hi
adds  x2, x2, x30              ; cols[2].lo += lo(a[1]*b[1])
```

**Why fastest**: Zero serial carry chains. All 25 column accumulations are independent — the CPU's OoO engine can overlap multiplies across columns. 28 GP registers used with only 1 stack spill (cols[6-7] high halves). Near-theoretical-minimum instruction count.

### fi64 Comparison

| Metric | fi64 4×64 | fi64 5×52 |
|---|---|---|
| Total ALU ops | 24 | 20 |
| mul/umulh pairs | 4 | 5 |
| Serial carry ops | 0 | 0 |
| Stack spills/iter | 3 stp | **5 stp** |
| Measured ns/op | 1.22 | **0.95** |

The fi64 5×52 loop is 20% fewer ALU ops but has 67% more stack spills. Despite this, it still wins by 1.28× because the spills are to L1 cache (3-4 cycle latency) while the 4×64 method has carry capture overhead (`cinc` after each `adds`).

```asm
; fi64 5x52 — complete column 0 (spill visible)
umulh x5, x4, x3              ; hi(a[0] * b)
mul   x4, x4, x3              ; lo(a[0] * b)
adds  x17, x4, x17            ; cols[0].lo += lo
adc   x0, x5, x0              ; cols[0].hi += hi
stp   x17, x0, [sp, #240]     ; SPILL cols[0] back to stack
```

### Register Pressure Analysis

| Method | Regs needed | Regs available | Spills | Spill cost |
|---|---|---|---|---|
| Naive | 28 | 28 | 0 | — |
| 4×64 current | 25 | 28 | 0 | — |
| 4×64 fused | 19 | 28 | 0 | — |
| **5×52 ff** | **29** | **28** | **1 stp + 3 ldr** | **~4 cycles** |
| fi64 4×64 | 13 | 28 | 3 stp (writeback) | ~6 cycles |
| fi64 5×52 | 15 | 28 | 5 stp (writeback) | ~10 cycles |

The 5×52 ff method is 1 register over budget, causing minimal spillage. The fi64 methods spill due to LLVM choosing to write back accumulators to the stack array each iteration rather than keeping them in registers.

### Pipeline Analysis

Apple M4 has 6 integer execution ports with:
- 4 multiply units (throughput: 4 `mul`/`umulh` per cycle)
- 4 add units (throughput: 4 `adds`/`adc` per cycle)

With 50 multiplies at 4/cycle throughput: minimum 12.5 cycles for multiplies.
With ~50 adds at 4/cycle: minimum 12.5 cycles for adds.
Total predicted minimum: ~13 cycles per MAC (multiply-bottlenecked).

At ~3.5 GHz: 13 cycles / 3.5 GHz ≈ 3.7 ns. **Measured: 4.2 ns** — within 14% of theoretical minimum, accounting for load latency and spill overhead.

### Standalone Field Multiply Benchmark

To confirm that 5×52 only helps for dot products (amortized overhead), we benchmarked individual field operations at N=1M:

| Operation | ns/op | Notes |
|---|---|---|
| field_mul (halo2curves) | 11.49 | Full Montgomery multiply |
| field_mul (5×52 pipeline) | 29.86 | to_52 → mac → carry_prop → REDC |
| field_add (halo2curves) | 2.05 | 4-limb add + conditional sub |

The 5×52 standalone multiply is **2.6× slower** than halo2curves because the conversion (to_52), carry propagation, and Montgomery REDC overhead are all per-element. In dot products, N MACs share one conversion pass and one reduction, amortizing this cost to near zero.

## What We Tried That Didn't Work (and Why)

### Inline Assembly (4×64)
- `mac_4x4_into_asm`: Identical performance to Rust fused (LLVM already generates optimal `mul`/`umulh`/`adds`/`adcs`)
- `mac_4x1_into_asm`: **2.1× slower** than Rust — asm barrier prevents loop optimization
- `mac_4x2_into_asm`: **1.4× slower** — same issue

**Lesson**: On Apple Silicon with modern LLVM, inline asm for arithmetic is counterproductive. LLVM's scheduling freedom matters more than controlling individual instructions.

### 8×32 NEON (`vmull_u32`)
- Field×Field: **4.1× slower** than 5×52
- Field×i64: **2.4× slower**

**Why it lost**:
1. No 64×64→128 NEON multiply exists. `vmull_u32` does 32×32→64, requiring 64 multiplies for 8×8 schoolbook vs 25 for 5×5.
2. NEON→GP register transfers add latency on Apple Silicon.
3. 15 columns vs 9 columns → more memory traffic and carry propagation.

**Lesson**: NEON is only useful for fields that naturally fit in 32-bit limbs (e.g., Goldilocks p = 2⁶⁴ - 2³² + 1). For 254-bit fields, scalar 52-bit limbs are superior.

### ILP Batching (K=2,4)
For both 4×64 and 5×52: K>1 showed 0-11% **regression**.

**Why**: Apple M4's out-of-order execution window (>300 instructions) already extracts all available ILP from a single accumulator's independent multiplies. Adding more accumulators only increases register pressure and spills.

### 4×64 Fused (no intermediate array)
`mac_4x4_into_fused` vs `mul_4_by_4` + add: **0% difference**.

**Why**: LLVM inlines and fuses them anyway. The compiler already eliminates the temporary array.

## Why Further Optimization Is Unlikely

### Evidence of Convergence

1. Three consecutive micro-optimization attempts (explicit locals, K=2, unroll) produced 0%, -4%, -11% changes.
2. Assembly inspection shows only 2 register spills — eliminating even those would save <1 cycle per MAC.
3. Measured throughput (4.2 ns/op) is within 14% of the multiply-unit throughput limit (3.7 ns).

### Remaining 14% Gap Analysis

The ~0.5 ns gap between measured (4.2) and theoretical (3.7) comes from:
- **Load latency**: 5 paired loads at L1 latency (3-4 cycles each, partially pipelined)
- **2 stack spills**: ~4 cycles for store/reload
- **Loop control**: 3 instructions per iteration

These are fundamental costs that cannot be eliminated without changing the algorithm (e.g., using a different multiplication scheme like Karatsuba, which for 5×5 saves only 2 multiplies but adds many additions — not worthwhile at this size).

### Comparison to External Implementations

- **Plonky3 BN254**: Uses interleaved Montgomery (reduces after each multiply). This is optimal for single multiplications but defeats delayed reduction. For dot products, our 5×52 lazy approach is fundamentally faster because it does N MACs before a single reduction.
- **Expander Goldilocks**: Uses no NEON for 64-bit field arithmetic, confirming our finding that NEON doesn't help for >32-bit fields.

## Recommendation

### Wire into production code

Replace the current 4×64 MAC operations in `delayed_reduction.rs` with 5×52 column accumulators:

1. **Field×Field** (`DelayedReduction<F>`, line 274): Use `mac_ff_52` with `[u128; 9]` accumulator
2. **Field×i64** (`DelayedReduction<i64>`, line 163): Use `mac_fi64_52` with `[u128; 5]` accumulator
3. **Field×i128** (`DelayedReduction<i128>`, line 212): Use `mac_fi128_52` with `[u128; 7]` accumulator
4. **Field×i32** (`DelayedReduction<i32>`, line 116): Use `mac_fi64_52` with i32→u64 extension

### Conversion strategy

- Convert field elements from 4×64 to 5×52 once at loop entry (cost: ~1ns per element, amortized)
- Accumulate using `mac_*_52` kernels (zero carry overhead per MAC)
- Convert back: `carry_propagate_52_to_Nlimb` → existing Barrett/Montgomery reduction (cost: ~2.6ns per reduction, once at loop exit)

### Expected end-to-end impact

For the Spartan SHA-256 sumcheck (Procedure 9):
- Sumcheck inner loops are dominated by field×i64 and field×i128 MACs
- At 1.28-1.43× speedup on these operations, expect **~15-25% reduction** in total sumcheck prover time
- Exact impact depends on the fraction of time spent in MAC operations vs. other work (memory access, Lagrange extension, etc.)

### Files to modify

| File | Change |
|---|---|
| `src/small_field/delayed_reduction.rs` | Replace MAC kernels with 5×52 variants |
| `src/small_field/limbs52.rs` | Already complete — production-ready |
| `src/small_field/limbs32.rs` | Can be removed or kept for reference |
| `src/small_field/aarch64.rs` | Can be removed or kept for reference |
