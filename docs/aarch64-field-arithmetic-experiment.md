# AARCH64 Field Arithmetic Optimization Experiment

## Overview

This document describes an experiment to measure whether alternative limb representations, hand-written assembly, and ILP batching provide speedups over the current Rust/LLVM-generated code for BN254 finite field arithmetic on Apple Silicon.

The experiment tests **5 methods x 4 ILP levels x 3 workloads x 3 sizes = 180 data points**, producing definitive data on the optimal approach for Spartan2's prover hot path.

## Background

### Current Implementation (4x64 limbs)

Spartan2 represents BN254 Fr field elements as 4 x 64-bit limbs in Montgomery form:

```
a = a_0 + a_1 * 2^64 + a_2 * 2^128 + a_3 * 2^192   (mod p)
```

The prover's hot path is **dot products** of the form:

```
result = SUM_i  field_i * value_i
```

where `value_i` is either another field element (field x field), an i64 (field x small), or an i128 (field x small*small).

These dot products use **delayed modular reduction**: accumulate many unreduced products into a wide integer accumulator, then reduce modulo p once at the end.

| Accumulator type | Width | Used for |
|---|---|---|
| `WideLimbs<9>` (576 bits) | 9 x 64-bit limbs | field x field products |
| `SignedWideLimbs<6>` (384 bits) | 6 x 64-bit limbs (pos + neg) | field x i64 products |
| `SignedWideLimbs<7>` (448 bits) | 7 x 64-bit limbs (pos + neg) | field x i128 products |

### The Problem

The current inner loop for field x field (`delayed_reduction.rs:274`) does:

```rust
// 1. Compute 4x4 multiply into temp array (16 mul/umulh pairs)
let product = mul_4_by_4(field_a.to_limbs(), field_b.to_limbs()); // temp [u64; 8]

// 2. Add temp array to 9-limb accumulator with serial carry chain
let mut carry = 0u128;
for i in 0..8 {
    let sum = (acc.0[i] as u128) + (product[i] as u128) + carry;
    acc.0[i] = sum as u64;
    carry = sum >> 64;
}
acc.0[8] = acc.0[8].wrapping_add(carry as u64);
```

Two inefficiencies:
1. **Temporary array**: `mul_4_by_4` materializes a `[u64; 8]` that is immediately consumed
2. **Serial carry chain**: Every MAC has a 9-step serial dependency chain through the accumulator

### Why Not NEON?

For 4x64 prime field arithmetic, the hot operations are 64x64 -> 128 integer multiplies with carry chains. These map to scalar AArch64 instructions (`mul`, `umulh`, `madd`, `adds`, `adcs`), not NEON. The NEON multiply instructions (`vmull_u32`) operate on 32-bit lanes — useful for binary field arithmetic (like Binius) but not for 64-bit limb prime fields.

However, if we change the limb representation to 8x32, NEON becomes viable via `vmull_u32` (widening 32x32 -> 64, 2 lanes at once).

## Experimental Methods

### Method 1: 4x64 Current (Baseline)

The existing code. Creates temporary `[u64; 8]`, adds to 9-limb accumulator with serial carry chain on every MAC.

- **Multiply cost**: 16 `mul`/`umulh` pairs per product
- **Carry chain**: 9 dependent `adds`/`adcs` per MAC
- **ILP within one MAC**: None (fully serial carry chain)

### Method 2: 4x64 Fused MAC

Eliminates the temporary array by fusing multiply and accumulate into one operation:

```rust
fn mac_4x4_into(acc: &mut [u64; 9], a: &[u64; 4], b: &[u64; 4]) {
    // For each b limb, multiply against all a limbs and add to accumulator
    for j in 0..4 {
        let (r, c) = mac(acc[0+j], a[0], b[j], 0);  acc[0+j] = r;
        let (r, c) = mac(acc[1+j], a[1], b[j], c);  acc[1+j] = r;
        let (r, c) = mac(acc[2+j], a[2], b[j], c);  acc[2+j] = r;
        let (r, c) = mac(acc[3+j], a[3], b[j], c);  acc[3+j] = r;
        // carry propagation into acc[4+j..8]
        let (v, c) = acc[4+j].overflowing_add(c); acc[4+j] = v;
        // ... propagate through remaining limbs
    }
}
```

- **Multiply cost**: Same 16 `mul`/`umulh` pairs
- **Carry chain**: Same 9 dependent ops, but no temp array overhead
- **Expected benefit**: Fewer loads/stores, better register utilization

### Method 3: 4x64 Inline Assembly

Hand-written `asm!` blocks using explicit `mul`/`umulh`/`adds`/`adcs` carry chains:

```rust
#[cfg(target_arch = "aarch64")]
fn mac_4x4_into_aarch64(acc: &mut [u64; 9], a: &[u64; 4], b: &[u64; 4]) {
    unsafe {
        core::arch::asm!(
            // Load a[0..4] into x4-x7
            "ldp x4, x5, [{a}]",
            "ldp x6, x7, [{a}, #16]",
            // Load acc[0..4] into x8-x11
            "ldp x8, x9, [{acc}]",
            "ldp x10, x11, [{acc}, #16]",
            // ... explicit mul/umulh/adds/adcs chains
            a = in(reg) a.as_ptr(),
            acc = in(reg) acc.as_mut_ptr(),
            // ...
        );
    }
}
```

- **Multiply cost**: Same 16 `mul`/`umulh` pairs
- **Expected benefit**: Optimal instruction scheduling, no missed `madd` combines
- **Risk**: LLVM may already generate equivalent code

### Method 4: 5x52 Lazy Column Accumulators

Changes the limb representation to eliminate serial carry chains entirely.

**Representation**: 5 limbs of 52 bits each.

```
a = a_0 + a_1 * 2^52 + a_2 * 2^104 + a_3 * 2^156 + a_4 * 2^208
```

**Conversion** from 4x64 (zero-cost bit shifting):

```rust
fn to_52(a: &[u64; 4]) -> [u64; 5] {
    const MASK: u64 = (1u64 << 52) - 1;
    [
        a[0] & MASK,                                         // bits 0-51
        ((a[0] >> 52) | (a[1] << 12)) & MASK,               // bits 52-103
        ((a[1] >> 40) | (a[2] << 24)) & MASK,               // bits 104-155
        ((a[2] >> 28) | (a[3] << 36)) & MASK,               // bits 156-207
        a[3] >> 16,                                          // bits 208-255
    ]
}
```

**Key insight**: Each cross-product is only 104 bits (52 + 52), leaving 24 bits of headroom in a `u128` column accumulator. This means we can accumulate up to ~16 million MACs without any carry propagation.

**Inner loop** — 25 independent column additions per MAC, zero serial dependencies:

```rust
// 9 column accumulators (indices 0..8 for 5x5 multiply)
let mut cols = [0u128; 9];

for (a, b) in pairs {
    let a = to_52(a.to_limbs());
    let b = to_52(b.to_limbs());

    // Column 0: a[0]*b[0]
    cols[0] += (a[0] as u128) * (b[0] as u128);
    // Column 1: a[0]*b[1] + a[1]*b[0]
    cols[1] += (a[0] as u128) * (b[1] as u128) + (a[1] as u128) * (b[0] as u128);
    // Column 2: a[0]*b[2] + a[1]*b[1] + a[2]*b[0]
    cols[2] += (a[0] as u128) * (b[2] as u128) + (a[1] as u128) * (b[1] as u128)
             + (a[2] as u128) * (b[0] as u128);
    // ... columns 3-8 follow the same Comba pattern
}

// ONE carry propagation + reduction at the very end
let result = carry_propagate_and_reduce(&cols);
```

**Trade-offs**:

| Aspect | 4x64 | 5x52 |
|---|---|---|
| Cross-products per multiply | 16 | 25 |
| Serial dependency per MAC | 9 carry steps | 0 (all independent) |
| Column headroom in u128 | 0 (needs wide acc) | 24 bits (~16M lazy MACs) |
| ILP within one MAC | None | Full (25 independent ops) |
| Conversion cost | None | ~20 shift/mask ops per element |
| Reduction | Per MAC (carry chain) | Once at end (amortized) |

### Method 5: 8x32 NEON

Changes to 8 x 32-bit limbs, enabling NEON widening multiply.

**Representation**: Trivial split of 4x64.

```rust
fn to_32(a: &[u64; 4]) -> [u32; 8] {
    [
        a[0] as u32, (a[0] >> 32) as u32,
        a[1] as u32, (a[1] >> 32) as u32,
        a[2] as u32, (a[2] >> 32) as u32,
        a[3] as u32, (a[3] >> 32) as u32,
    ]
}
```

**NEON inner loop** using `vmull_u32` (2 cross-products per instruction):

```rust
use core::arch::aarch64::*;

unsafe fn mac_8x8_neon(cols: &mut [u64; 16], a: &[u32; 8], b: &[u32; 8]) {
    for j in 0..8 {
        let bj = vdup_n_u32(b[j]);           // broadcast b[j] to 2 lanes
        for i in (0..8).step_by(2) {
            let a_pair = vld1_u32(&a[i]);     // load a[i], a[i+1]
            let prod = vmull_u32(a_pair, bj); // [a[i]*b[j], a[i+1]*b[j]] as u64x2
            cols[i+j]   += vgetq_lane_u64(prod, 0);
            cols[i+j+1] += vgetq_lane_u64(prod, 1);
        }
    }
}
```

**Trade-offs**:

| Aspect | 4x64 | 8x32 NEON |
|---|---|---|
| Cross-products per multiply | 16 | 64 |
| NEON ops per multiply | 0 | 32 (2 products per `vmull_u32`) |
| Column accumulator type | Wide 9-limb | Simple `[u64; 16]` |
| Headroom per column | 0 | 64 bits (~2^32 lazy MACs) |
| SIMD throughput | N/A | 2 muls per instruction |

## ILP Batching (K = 1, 2, 4, 8)

Orthogonal to limb representation. Instead of one accumulator with a long dependency chain, use K independent accumulators that interleave:

```
K=1 (baseline):
  acc += a[0]*b[0]; acc += a[1]*b[1]; acc += a[2]*b[2]; ...

K=4:
  acc0 += a[0]*b[0]; acc1 += a[1]*b[1]; acc2 += a[2]*b[2]; acc3 += a[3]*b[3];
  acc0 += a[4]*b[4]; acc1 += a[5]*b[5]; acc2 += a[6]*b[6]; acc3 += a[7]*b[7];
  ...
  result = reduce(acc0 + acc1 + acc2 + acc3)
```

This breaks serial dependency chains, giving Apple Silicon's out-of-order execution more independent work to schedule.

### Register Pressure Constraints

| Method | State per accumulator | ARM64 GP regs (31) | Practical max K |
|---|---|---|---|
| 4x64 fused | 9 x u64 = 72 bytes | 9 regs | K=3 |
| 5x52 lazy | 9 x u128 = 144 bytes | 18 regs | K=1-2 |
| 8x32 NEON | 16 x u64 = 128 bytes | 8 NEON regs (v0-v7) | K=1-2 |

**Prediction**: K=4 works well for 4x64 (small accumulator state). K=2 is the sweet spot for 5x52 and 8x32 (larger accumulators). K=8 will likely regress everywhere due to register spills to stack.

Note: 5x52 already has high internal ILP (25 independent column adds per MAC), so the benefit of K>1 is smaller than for 4x64.

## Workloads

### Workload 1: Field x Field Dot Product

```
result = SUM_{i=0}^{N-1}  field_i * field_i
```

This is the hottest path in the eq-split sumcheck (standard Spartan prover). Each product is a full 256x256-bit multiply, accumulated into a wide integer, reduced once at the end.

### Workload 2: Field x i64 Dot Product

```
result = SUM_{i=0}^{N-1}  field_i * small_i     where small_i is i64
```

This is the hot path in the small-value sumcheck accumulator building (Procedure 9). The "small x large" (sl) multiply is cheaper than "large x large" (ll) because only one operand is 256 bits.

### Workload 3: Field x i128 Dot Product

```
result = SUM_{i=0}^{N-1}  field_i * product_i   where product_i = small_a * small_b (i128)
```

This is the "small x small x large" path in the accumulator, where two small values are multiplied first (native i64 x i64 -> i128), then the i128 result is multiplied by a field element.

## Benchmark Design

### Test Matrix

**5 methods x 4 ILP levels x 3 workloads x 3 sizes**

Sizes: N = 2^10 (1K), 2^14 (16K), 2^18 (256K), 2^22 (4M), 2^24 (16M)

Field: BN254 Fr

### Output Tables

#### Table 1: Field x Field Dot Product (N = 2^18)

```
+------------------+----------+----------+----------+----------+
| Method           |   K=1    |   K=2    |   K=4    |   K=8    |
+------------------+----------+----------+----------+----------+
| 4x64_current     |   xx ms  |   xx ms  |   xx ms  |   xx ms  |
| 4x64_fused       |   xx ms  |   xx ms  |   xx ms  |   xx ms  |
| 4x64_asm         |   xx ms  |   xx ms  |   xx ms  |   xx ms  |
| 5x52_lazy        |   xx ms  |   xx ms  |   xx ms  |   xx ms  |
| 8x32_neon        |   xx ms  |   xx ms  |   xx ms  |   xx ms  |
+------------------+----------+----------+----------+----------+
```

#### Table 2: Field x i64 Dot Product (N = 2^18)

Same format.

#### Table 3: Field x i128 Dot Product (N = 2^18)

Same format.

#### Table 4: Scaling — Best Method at N = 2^10, 2^14, 2^18, 2^22, 2^24

```
+------------------+----------+----------+----------+----------+----------+
| Workload         |   2^10   |   2^14   |   2^18   |   2^22   |   2^24   |
+------------------+----------+----------+----------+----------+----------+
| FF baseline      |   xx ms  |   xx ms  |   xx ms  |   xx ms  |   xx ms  |
| FF best          |   xx ms  |   xx ms  |   xx ms  |   xx ms  |   xx ms  |
| FF speedup       |   x.xxX  |   x.xxX  |   x.xxX  |   x.xxX  |   x.xxX  |
+------------------+----------+----------+----------+----------+----------+
| Fi64 baseline    |   xx ms  |   xx ms  |   xx ms  |   xx ms  |   xx ms  |
| Fi64 best        |   xx ms  |   xx ms  |   xx ms  |   xx ms  |   xx ms  |
| Fi64 speedup     |   x.xxX  |   x.xxX  |   x.xxX  |   x.xxX  |   x.xxX  |
+------------------+----------+----------+----------+----------+----------+
| Fi128 baseline   |   xx ms  |   xx ms  |   xx ms  |   xx ms  |   xx ms  |
| Fi128 best       |   xx ms  |   xx ms  |   xx ms  |   xx ms  |   xx ms  |
| Fi128 speedup    |   x.xxX  |   x.xxX  |   x.xxX  |   x.xxX  |   x.xxX  |
+------------------+----------+----------+----------+----------+----------+
```

### Timing Methodology

- **Warmup**: 2-3 iterations before measuring
- **Trials**: 10 iterations, report median (not mean, avoids outliers)
- **`black_box()`**: Prevents dead code elimination
- **Correctness check**: Every variant asserts its result equals the baseline

### Profiling Methodology

For each variant, three layers of analysis:

**Layer 1 — Wall-clock timing** (the tables above)

Simple `Instant::now()` with `black_box()`, median of 10 trials.

**Layer 2 — Flamegraph profiling**

The benchmark binary accepts a `profile <variant>` CLI mode that runs one variant in a long loop. Profiled with:

```bash
RUSTFLAGS="-C debuginfo=2 -C force-frame-pointers=yes \
  -C split-debuginfo=unpacked -C target-cpu=native" \
  cargo flamegraph --profile bench --example field_asm_bench \
  -o flamegraph_<variant>.svg -- profile <variant>
```

Answers: "Within this dot product, what fraction of time is multiply vs carry propagation vs conversion vs reduction?"

**Layer 3 — Assembly inspection**

```bash
cargo asm --release --example field_asm_bench "<function_name>" --rust
```

Verify LLVM generates expected instruction patterns:

| Method | Expected assembly pattern |
|---|---|
| 4x64 fused | `mul`/`umulh` pairs + `adds`/`adcs` carry chain |
| 4x64 asm | Hand-written `mul`/`umulh`/`adds`/`adcs` (should match) |
| 5x52 lazy | Independent `mul`/`umulh` pairs + independent `adds` (no carry chain between columns) |
| 8x32 NEON | `vmull_u32` (widening NEON multiply) + `vaddq_u64` (NEON accumulate) |

Red flags to watch for:
- `str`/`ldr` to `[sp, ...]` in hot loop = **register spills** (K too high)
- `umov` (NEON -> scalar transfer) in 8x32 = **kills SIMD throughput**
- Missing `madd` = LLVM missed a fused multiply-add opportunity
- Single `str`/`ldr` where `stp`/`ldp` could work = suboptimal memory access

## File Structure

```
src/small_field/
  aarch64.rs        NEW  4x64 assembly kernels (#[cfg(target_arch = "aarch64")])
  limbs52.rs        NEW  5x52 column-accumulator kernels
  limbs32.rs        NEW  8x32 NEON kernels (#[cfg(target_arch = "aarch64")])

examples/
  field_asm_bench.rs  NEW  Benchmark harness (all variants + profiling mode)
```

All new files are **additive** — nothing in the existing codebase is modified. The existing `DelayedReduction` implementations remain untouched until we have data.

## Decision Criteria

| Result | Action |
|---|---|
| 5x52 or 8x32 wins by >30% | Implement as new `DelayedReduction` backend, benchmark end-to-end on SHA-256 |
| Assembly wins 10-20% over fused Rust | Wire asm behind `#[cfg(target_arch = "aarch64")]` |
| Fused Rust wins 10-20% (no asm needed) | Replace current code with fused version (simplest, portable) |
| ILP K=4 wins significantly | Restructure sumcheck inner loops to use multi-accumulator pattern |
| No clear winner (<5% difference) | LLVM already generates near-optimal code; focus effort on algorithmic improvements |

## Compiler Settings

```toml
# .cargo/config.toml
[target.aarch64-apple-darwin]
rustflags = ["-Ctarget-cpu=native"]

# Cargo.toml
[profile.release]
opt-level = 3
debug = 1
lto = "thin"
codegen-units = 1
incremental = false

[profile.bench]
inherits = "release"
debug = 2
strip = false
split-debuginfo = "unpacked"
```

## References

- [Speeding Up Sum-Check Proving](https://eprint.iacr.org/2024/1046) — The small-value optimization paper
- [Spartan2 PR #98](https://github.com/microsoft/Spartan2/pull/98) — Delayed modular reduction implementation
- [ARM Architecture Reference Manual](https://developer.arm.com/documentation/ddi0602/latest/) — AArch64 instruction reference
- [Rust core::arch::aarch64](https://doc.rust-lang.org/core/arch/aarch64/) — NEON intrinsics
- [ethproofs.org CSP Benchmarks](https://ethproofs.org/csp-benchmarks) — SHA-256 benchmark format
