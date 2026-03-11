# Plonky3 aarch64 Field Arithmetic: How They Make It Fast

## Overview

This document analyzes how Plonky3 implements high-performance field arithmetic on Apple Silicon (aarch64). Plonky3 targets three families of ZK-friendly primes, each with a different optimization strategy exploiting the prime's algebraic structure. The techniques range from NEON SIMD vectorization to hand-written inline assembly with instruction-level parallelism (ILP) scheduling.

**Key difference from BN254**: Plonky3's primes are small (31-bit or 64-bit), not 256-bit. This fundamentally changes the optimization landscape — there are no multi-limb carry chains. Instead, the bottleneck is throughput of *many independent* single-limb field operations, which is ideal for SIMD.

## The Three Fields

| Field | Prime P | Bit width | Representation | NEON width |
|---|---|---|---|---|
| **BabyBear / KoalaBear** | 2³¹ − 2²⁷ + 1 / 2³¹ − 2²⁴ + 1 | 31 bits | Montgomery form (`â = a·2³² mod P`) | 4 × u32 |
| **Mersenne-31** | 2³¹ − 1 | 31 bits | Canonical (possibly = P) | 4 × u32 |
| **Goldilocks** | 2⁶⁴ − 2³² + 1 | 64 bits | Non-canonical (any u64) | 2 × u64 (limited) |

The 31-bit fields fit 4 elements in a single NEON `uint32x4_t` register. Goldilocks requires 64-bit lanes, so NEON can only hold 2 elements — and critically, NEON has no 64×64→128 multiply instruction, forcing a scalar fallback for multiplication.

## 1. Montgomery Multiplication for 31-bit Fields (BabyBear / KoalaBear)

### The Math

Elements are stored in Montgomery form: instead of `a`, store `â = a · 2³² mod P`. Multiplication computes:

```
â · b̂ · 2⁻³² mod P
```

which equals `(a·b)^hat` — the Montgomery form of the true product. The reduction avoids division by P entirely.

**Algorithm** (from Brent & Zimmermann, adapted):

```
Constants: B = 2³², μ = P⁻¹ mod B
Input:     C = â · b̂,  where −P² ≤ C ≤ P²
Output:    D ∈ (−P, P)  such that  D ≡ C · B⁻¹  (mod P)

1. Q := C · μ  mod B        (signed mod, keep low 32 bits)
2. D := (C − Q · P) / B     (exact division — no remainder)
```

The division by B is exact because Q was chosen to make C − Q·P divisible by B. The proof:
- Q·P ≡ (μ·C)·P ≡ P⁻¹·C·P ≡ C (mod B), so C − Q·P ≡ 0 (mod B). ∎

### The NEON Assembly

The implementation uses a clever trick with `sqdmulh` (Signed saturating Doubling multiply returning High half). For 32-bit inputs a, b, `sqdmulh` computes `⌊2·a·b / 2³²⌋` — extracting bits 31..62 of the full 64-bit product in a single instruction.

**Source**: `monty-31/src/aarch64_neon/packing.rs:438-524`

Target instruction sequence for one vectorized multiply (4 elements in parallel):

```asm
sqdmulh  c_hi.4s, lhs.4s, rhs.4s       // c_hi = bits 31..62 of lhs*rhs
mul      mu_rhs.4s, rhs.4s, MU.4s       // precompute μ·rhs mod 2³²
mul      q.4s, lhs.4s, mu_rhs.4s        // Q = lhs · (μ·rhs) mod 2³²
sqdmulh  qp_hi.4s, q.4s, P.4s          // qp_hi = bits 31..62 of Q·P
shsub    res.4s, c_hi.4s, qp_hi.4s     // D = (c_hi − qp_hi) / 2  (halving sub)
cmgt     underflow.4s, qp_hi.4s, c_hi.4s  // detect D < 0
mls      res.4s, underflow.4s, P.4s     // if D < 0: D += P
```

**Why `sqdmulh` + `shsub`?** The "doubling" in `sqdmulh` and the "halving" in `shsub` cancel out, recovering the exact high-word division needed for step 2 of Montgomery reduction. This avoids a widening multiply (`smull`) followed by an explicit shift, saving instructions.

**Performance on Apple M4**:

| Metric | Value |
|---|---|
| Instructions | 7 NEON (with canonicalization) |
| Throughput | 1.75 cyc/vec = **2.29 elements/cycle** |
| Latency (rhs→result) | 14 cycles |
| Latency (lhs→result) | 11 cycles |

Without canonicalization (`CANONICAL=false`, for intermediate products):

| Metric | Value |
|---|---|
| Instructions | 4 NEON |
| Throughput | 1.0 cyc/vec = **4.0 elements/cycle** |
| Latency | 8 cycles |

### Optimization: Precomputed μ·rhs Reuse

For power chains (x², x³, x⁵, x⁷), the code precomputes `μ·x` once and reuses it across multiple multiplications where one operand is the same:

```rust
// Cube: x³ in 2 multiplications, but only 1 μ-precomputation
let mu_val = mulby_mu(val);                                    // μ·val
let val_2 = mul_with_precomp::<_, false>(val, val, mu_val);    // val² (non-canonical)
let val_3 = mul_with_precomp::<_, true>(val_2, val, mu_val);   // val³ (canonical)
```

The intermediate `val²` uses `CANONICAL=false`, saving 2 NEON instructions (the `cmgt` + `mls` reduction). This is safe because the next multiplication accepts inputs in (−P, P).

**Source**: `monty-31/src/aarch64_neon/packing.rs:526-598`

### Optimization: `confuse_compiler`

```rust
fn confuse_compiler(x: uint32x4_t) -> uint32x4_t {
    asm!("/*{0:v}*/", inlateout(vreg) x => y, ...);
}
```

This is a no-op `asm!` block (an assembly comment) that prevents LLVM from applying a specific harmful optimization. LLVM sometimes wants to fuse or reorder the `cmgt + mls` canonicalization in ways that increase latency on the critical path. The `asm!` block makes the value opaque to the optimizer without actually emitting any instructions.

**Source**: `monty-31/src/aarch64_neon/packing.rs:332-356`

### Optimization: Fused FFT Butterfly

The DIF butterfly computes `(x + y, (x − y) · ω)` for twiddle factor ω. The naive approach would modular-reduce `x − y` before multiplying by ω. Instead:

```rust
let sum  = uint32x4_mod_add(x, y, P);                      // canonical [0, P)
let diff = vreinterpretq_s32_u32(vsubq_u32(x, y));         // RAW subtract, no reduction
let product = montgomery_mul(diff, roots);                   // (x−y)·ω via Montgomery
```

The raw subtraction produces a value in (−P, P) as a signed integer, which is exactly the valid input range for Montgomery multiplication. This **saves 2 NEON instructions per butterfly** — significant since FFT is butterfly-dominated.

**Source**: `monty-31/src/aarch64_neon/packing.rs:118-148`, `monty-31/src/dft/forward.rs:82-98`

## 2. Mersenne-31 Multiplication

### The Math

For P = 2³¹ − 1, reduction exploits: 2³¹ ≡ 1 (mod P).

Given a·b, split the 62-bit product into:

```
a · b = hi · 2³¹ + lo
```

where hi, lo are both 31-bit. Then a·b ≡ hi + lo (mod P).

### The NEON Assembly

**Source**: `mersenne-31/src/aarch64_neon/packing.rs:243-294`

```asm
sqdmulh  prod_hi31.4s, lhs.4s, rhs.4s    // hi = ⌊2·a·b / 2³²⌋ ≈ bits 31..62
mul      t.4s, lhs.4s, rhs.4s              // lo32 = a·b mod 2³²
mls      t.4s, prod_hi31.4s, P.4s          // t = lo32 − hi·P = hi + lo (mod 2³²)
sub      u.4s, t.4s, P.4s                  // u = t − P
umin     res.4s, t.4s, u.4s                // branchless: res = min(t, u)
```

**Why `mls` with P works**: P = 2³¹ − 1, so `hi·P = hi·2³¹ − hi`. Subtracting `hi·P` from `lo32`:

```
lo32 − hi·(2³¹ − 1) = lo32 − hi·2³¹ + hi = lo31 + hi   (mod 2³²)
```

which is exactly the Mersenne reduction formula.

**The `umin` trick**: If `t < P`, then `u = t − P` wraps to a huge unsigned value, so `min(t, u) = t`. If `t ≥ P`, then `u` is the correct reduced value and `min(t, u) = u`. This is a branchless conditional reduction in one instruction.

**Performance on Apple M4**:

| Metric | Value |
|---|---|
| Instructions | 5 NEON |
| Throughput | 1.25 cyc/vec = **3.2 elements/cycle** |
| Latency | 10 cycles |

This is significantly faster than Montgomery (Mersenne reduction is simpler — no μ precomputation needed).

## 3. Modular Addition / Subtraction (All 31-bit Fields)

### The NEON Pattern

**Source**: `field/src/packed/aarch64_neon.rs:35-80`

Addition:
```asm
add   t.4s, a.4s, b.4s        // t = a + b (may overflow [0, 2P))
sub   u.4s, t.4s, P.4s        // u = t − P (wraps if t < P)
umin  res.4s, t.4s, u.4s      // branchless select correct one
```

Subtraction:
```asm
sub   t.4s, a.4s, b.4s        // t = a − b (may underflow)
add   u.4s, t.4s, P.4s        // u = t + P (wraps if t ≥ 0)
umin  res.4s, t.4s, u.4s      // branchless select
```

Both are 3 NEON instructions, 0.75 cyc/vec throughput (5.33 elements/cycle). The `umin` pattern avoids branches entirely — critical for avoiding branch mispredictions in tight loops.

## 4. Goldilocks Field Arithmetic (64-bit)

### The Challenge

P = 2⁶⁴ − 2³² + 1. Elements are 64 bits, so:
- NEON `uint64x2_t` holds only **2 elements** (vs 4 for 31-bit fields)
- NEON has **no 64×64→128 multiply** instruction
- Multiplication must use scalar `mul` / `umulh` instructions

### Key Identity

Define ε = 2³² − 1 = −P mod 2⁶⁴. Then 2⁶⁴ ≡ ε (mod P).

For a 128-bit product `hi·2⁶⁴ + lo = a·b`, split hi into 32-bit halves `hi = hi_hi·2³² + hi_lo`:

```
a·b ≡ lo + hi·ε
    ≡ lo + (hi_hi·2³² + hi_lo)·ε
    ≡ lo − hi_hi + hi_lo·ε      (mod P)
```

The last step uses 2³²·ε = 2⁶⁴ − 2³² ≡ −1 (mod P), so hi_hi·2³²·ε ≡ −hi_hi.

### Scalar Multiply Assembly

**Source**: `goldilocks/src/aarch64_neon/utils.rs:20-66`

```asm
// 128-bit product: hi:lo = a × b
mul   lo, a, b              // lo = (a·b) mod 2⁶⁴
umulh hi, a, b              // hi = (a·b) >> 64

// Reduction: result = lo − hi_hi + hi_lo·ε
lsr   t0, hi, #32           // t0 = hi_hi = hi >> 32
subs  t1, lo, t0            // t1 = lo − hi_hi, set carry flag
csetm t2:w, cc              // t2 = 0xFFFFFFFF if borrow, else 0
sub   t1, t1, t2            // conditional: subtract ε on borrow

and   t0, hi, epsilon       // t0 = hi_lo = hi & (2³² − 1)
mul   t0, t0, epsilon       // t0 = hi_lo × ε

adds  result, t1, t0        // result = t1 + t0, set carry
csetm t2:w, cs              // t2 = 0xFFFFFFFF if carry
add   result, result, t2    // conditional: add ε on overflow
```

**The `csetm` pattern**: `csetm reg:w, cc` sets all 32 bits of the register to 1 if the carry flag indicates borrow (cc = carry clear). Since ε = 0xFFFFFFFF, this directly produces ε as a conditional adjustment value. Subtracting −1 (all 1s sign-extended to 64 bits) effectively adds 1, but the semantics work out to adding/subtracting P because ε = −P mod 2⁶⁴.

Total: **11 scalar instructions per multiply**.

### Dual-Lane Interleaved Multiply

**Source**: `goldilocks/src/aarch64_neon/packing.rs:290-361`

Computes two independent `a₀·b₀ mod P` and `a₁·b₁ mod P` simultaneously. Instructions from both lanes are interleaved to exploit ILP on Apple Silicon's 2 integer multiply units:

```asm
// Both products start simultaneously (M4 has 2 mul units)
mul   lo0, a0, b0
mul   lo1, a1, b1
umulh hi0, a0, b0
umulh hi1, a1, b1

// Reductions interleaved
lsr   hi_hi0, hi0, #32
lsr   hi_hi1, hi1, #32
subs  tmp0, lo0, hi_hi0
csetm adj0:w, cc
subs  tmp1, lo1, hi_hi1
csetm adj1:w, cc
sub   tmp0, tmp0, adj0
sub   tmp1, tmp1, adj1
...
```

**Additional optimization**: Replaces `mul t0, t0, epsilon` with shift-subtract:

```asm
lsl   t0, hi_lo0, #32            // hi_lo << 32
sub   hi_lo_eps0, t0, hi_lo0     // hi_lo × (2³² − 1) = (hi_lo << 32) − hi_lo
```

This avoids tying up a multiply unit for the ε multiplication, since `x·(2³² − 1) = x·2³² − x`.

Total: **~22 scalar instructions for 2 multiplications**.

### Scalar Addition / Subtraction

**Source**: `goldilocks/src/aarch64_neon/utils.rs:127-149`

```asm
// Addition: a + b mod P
adds  result, a, b            // result = a + b, set carry if overflow
csetm adj:w, cs               // adj = ε if carry (overflow means result ≥ 2⁶⁴)
add   result, result, adj     // subtract P by adding ε = −P mod 2⁶⁴

// Subtraction: a − b mod P
subs  result, a, b            // result = a − b, set borrow flag
csetm adj:w, cc               // adj = ε if borrow
sub   result, result, adj     // add P by subtracting ε = −P mod 2⁶⁴
```

Only **3 instructions** each, branchless.

### Fused Multiply-Add

**Source**: `goldilocks/src/aarch64_neon/utils.rs:68-124`

Computes `a·b + c mod P` by accumulating c into the 128-bit product before reduction:

```asm
mul   lo, a, b
umulh hi, a, b
adds  lo, lo, c              // lo += c
adc   hi, hi, xzr            // hi += carry
// ... then standard reduction
```

This saves ~2 instructions vs separate multiply + add, and is used extensively in Poseidon2's internal permutation.

### Division by Powers of 2

**Source**: `goldilocks/src/aarch64_neon/poseidon2_asm.rs:12-58`

Division by 2 uses the identity: if x is odd, x/2 ≡ (x >> 1) + (P+1)/2 (mod P).

```asm
lsr   result, x, #1           // result = x >> 1
and   tmp, x, #1              // parity bit
cmp   tmp, #0
csel  tmp, shift, xzr, ne     // if odd: tmp = (P+1)/2, else 0
add   result, result, tmp
```

Division by 2³² exploits `2⁻³² ≡ 1 − 2³² (mod P)`:

```asm
lsr   hi, x, #32              // hi = x >> 32
and   lo, x, #0xFFFFFFFF      // lo = x & 0xFFFFFFFF
add   sum, hi, lo             // sum = hi + lo
lsl   t, lo, #32              // t = lo << 32
subs  result, sum, t          // result = sum − t
csetm adj:w, cc
sub   result, result, adj     // conditional add P
```

These are used in the Poseidon2 MDS layer to replace multiplications by diagonal constants that happen to be powers-of-2 fractions.

### NEON Addition/Subtraction (Shifted Representation)

**Source**: `goldilocks/src/aarch64_neon/packing.rs:181-253`

Since NEON has no unsigned 64-bit comparison, the code XORs with 2⁶³ to convert to a "shifted" representation where signed comparison (`vcgtq_s64`) works as unsigned comparison:

```rust
const SIGN_BIT: uint64x2_t = [1 << 63; 2];

fn shift(x: uint64x2_t) -> uint64x2_t {
    veorq_u64(x, SIGN_BIT)    // x XOR 2⁶³
}
```

Addition then becomes:
1. Shift y: `y_s = y XOR 2⁶³`
2. Canonicalize y_s (if y ≥ P, subtract P)
3. Add x + y_s (with overflow detection via signed compare)
4. Shift result back

This is ~7-8 NEON instructions — more expensive than the 3-instruction scalar `add_asm`. The NEON path is only worthwhile when many additions can be batched, because it processes 2 elements per instruction.

## 5. Poseidon2 Permutation: Putting It All Together

The Poseidon2 internal permutation is where all these primitives compose. Each round applies:

1. **Round constant addition**: `s₀ += rc`
2. **S-box** (power map): `s₀ ← s₀⁷` (BabyBear/KoalaBear/Goldilocks) or `s₀ ← s₀⁵` (Mersenne-31)
3. **Internal MDS matrix multiply**: `state = diag(d) · state + (Σ state) · 1`

### Goldilocks S-box: x⁷ via Assembly

**Source**: `goldilocks/src/aarch64_neon/poseidon2_asm.rs:120-156`

```rust
s0 = add_asm(s0, rc);                // round constant
let s0_2 = mul_asm(s0, s0);          // s0²
let s0_3 = mul_asm(s0_2, s0);        // s0³  (depends on s0²)
let s0_4 = mul_asm(s0_2, s0_2);      // s0⁴  (depends on s0², parallel with s0³)
s0 = mul_asm(s0_3, s0_4);            // s0⁷  (depends on both)
```

4 multiplications, critical path = 3 sequential muls = **~9 multiply-cycles** on M4. The s0³ and s0⁴ computations are independent and can execute in parallel on the M4's 2 multiply units.

### Latency Hiding via Dual-Lane Processing

The code processes two independent Poseidon2 states (lane a and lane b) simultaneously, interleaving their S-box computations:

```rust
s0_a = add_asm(s0_a, rc);
s0_b = add_asm(s0_b, rc);
let s0_2_a = mul_asm(s0_a, s0_a);    // lane a: s0²
let s0_2_b = mul_asm(s0_b, s0_b);    // lane b: s0² (parallel with lane a)
let s0_3_a = mul_asm(s0_2_a, s0_a);
let s0_3_b = mul_asm(s0_2_b, s0_b);
// ...
```

With M4's 2 multiply units and ~700-entry reorder buffer, the OOO engine can overlap lane a's multiply latency with lane b's independent operations.

### BabyBear/KoalaBear S-box: x⁷ via NEON

**Source**: `monty-31/src/aarch64_neon/packing.rs:575-598`

```rust
let mu_val = mulby_mu(val);                                       // μ·val (reused)
let val_2 = mul_with_precomp::<_, false>(val, val, mu_val);       // val²
let mu_val_2 = mulby_mu(val_2);                                   // μ·val²
let val_3 = mul_with_precomp::<_, false>(val_2, val, mu_val);     // val³
let mu_val_3 = mulby_mu(val_3);                                   // μ·val³
let val_4 = mul_with_precomp::<_, false>(val_2, val_2, mu_val_2); // val⁴
let val_7 = mul_with_precomp::<_, true>(val_4, val_3, mu_val_3);  // val⁷
```

Key optimizations:
- `μ·val` is precomputed once and reused for val², val³ (since one operand is `val`)
- `μ·val₂` and `val₃` are independent → computed in parallel
- `μ·val₃` and `val₄` are independent → computed in parallel
- Only the final multiplication uses `CANONICAL=true` (saves 2 instructions on each intermediate)

Throughput: **5.25 cyc/vec for 4 elements** (0.76 elements/cycle per S-box application).

### Monty31 Division by 2^N (MDS Optimization)

**Source**: `monty-31/src/aarch64_neon/utils.rs:32-112`

For primes of the form P = r·2ʲ + 1 (where r is odd and j is the two-adicity), multiplication by `2⁻ᴺ` avoids a full Montgomery multiply:

```
x · 2⁻ᴺ = hi + lo − lo · (P−1)/2ᴺ    where  hi = x >> N,  lo = x & (2ᴺ − 1)
```

NEON implementation:
```asm
and   lo.4s, input.4s, mask.4s          // lo = input & (2^N − 1)
vshr  hi.4s, input.4s, #N               // hi = input >> N
vmls  res.4s, lo.4s, odd_factor.4s      // res = hi − lo·((P−1)/2^N)
add   u.4s, res.4s, P.4s               // branchless reduction
umin  res.4s, res.4s, u.4s
```

The `vmls` (vector multiply-subtract) fuses the constant multiplication and subtraction into one instruction. This is used for diagonal matrix entries in the Poseidon2 MDS layer.

## 6. Performance Summary

### Throughput (elements/cycle on Apple M4 P-core)

| Operation | Mersenne-31 | BabyBear | Goldilocks |
|---|---|---|---|
| **Multiply** | 3.2 (NEON 4-wide) | 2.3 (NEON 4-wide) | ~0.8−1.0 (scalar 2-wide) |
| **Add/Sub** | 5.3 (NEON 4-wide) | 5.3 (NEON 4-wide) | ~2.0 (scalar) |
| **Cube** | — | 1.45 (NEON) | ~0.5 (scalar) |
| **x⁵** | 3.2 (same as mul) | 1.0 (NEON) | ~0.4 (scalar) |
| **x⁷** | — | 0.76 (NEON) | ~0.3 (scalar) |
| **FFT butterfly** | ~2.5 | ~2.5 (fused) | ~0.6 (scalar) |

### Instruction Counts per Operation (4 elements)

| Operation | Mersenne-31 | BabyBear (Montgomery) | Goldilocks |
|---|---|---|---|
| Multiply | 5 NEON | 7 NEON (4 non-canonical) | 22 scalar (2 els) |
| Add | 3 NEON | 3 NEON | 3 scalar |
| Sub | 3 NEON | 3 NEON | 3 scalar |
| Negate | 1 NEON | 3 NEON | 3 NEON |

## 7. Optimization Techniques Catalog

### Technique 1: `sqdmulh` for Cheap High-Half Extraction

`sqdmulh` (signed saturating doubling multiply high) computes `⌊2·a·b / 2³²⌋` in one instruction. Used by both Montgomery (to get bits 31..62 of C and Q·P) and Mersenne-31 (to get the high 31-bit half). This is the single most impactful NEON trick — it replaces what would otherwise be a widening multiply + shift.

### Technique 2: `umin` for Branchless Reduction

For values in [0, 2P), the pattern `min(x, x − P)` selects the correct reduced value without branching. If `x < P`, then `x − P` wraps to a huge unsigned number, so `min` picks `x`. If `x ≥ P`, then `x − P` is correct. Used universally across all 31-bit field add/sub operations.

### Technique 3: `csetm` + Arithmetic for Conditional ε-Adjustment

For Goldilocks, the pattern:
```asm
subs  result, a, b
csetm adj:w, cc         // adj = 0xFFFFFFFF if borrow
sub   result, result, adj
```
exploits that ε = 0xFFFFFFFF = −P mod 2⁶⁴. `csetm` produces ε directly as the conditional mask, and subtracting it (or adding it) adjusts by exactly ±P.

### Technique 4: Shift-Subtract Instead of Multiply-by-ε

The identity `x · (2³² − 1) = (x << 32) − x` replaces a 64-bit `mul` with an `lsl` + `sub`, freeing a multiply unit. Used in the dual-lane Goldilocks multiply but not yet in the single-lane version.

### Technique 5: Non-Canonical Intermediates

Montgomery products in (−P, P) are valid inputs to the next multiplication, so intermediate results skip the 2-instruction canonicalization step. Only the final result in a chain needs to be reduced to [0, P).

### Technique 6: μ-Precomputation Reuse

For power chains where one operand repeats (x², x³, x⁴ all involve x), `μ·x` is computed once and reused, saving one `mul` instruction per multiplication that shares the operand.

### Technique 7: Fused Butterfly (Skip Intermediate Reduction)

In FFT DIF butterflies, `x − y` produces a value in (−P, P) which is directly valid for Montgomery multiplication. Skipping the modular reduction on the difference saves 2 NEON instructions per butterfly.

### Technique 8: ILP-Scheduled Inline Assembly

For Goldilocks where NEON can't help with multiplication, instructions from two independent multiplications are interleaved in a single `asm!` block. The M4's 2 integer multiply units and 6 ALU pipes can execute both in parallel.

### Technique 9: Compiler Barrier (`confuse_compiler`)

A no-op `asm!` block that emits only a comment but prevents LLVM from reordering instructions across it. Used to stop LLVM from "optimizing" the canonicalization path into a higher-latency sequence.

## 8. Comparison with BN254 (Spartan2)

| Aspect | BN254 (4×64 limbs) | Plonky3 31-bit fields | Plonky3 Goldilocks |
|---|---|---|---|
| Element size | 256 bits | 31 bits | 64 bits |
| NEON usable for mul? | No (need 64×64→128) | **Yes** (32×32→32 sufficient) | No (need 64×64→128) |
| Elements per NEON reg | 0 (scalar only) | **4** | 2 (add/sub only) |
| Mul strategy | 4×4 schoolbook + carry chain | Montgomery via `sqdmulh` | Scalar `mul`/`umulh` + ε reduce |
| Carry chain length | 9 dependent `adds`/`adcs` | **0** (single-instruction) | 0 (flag-based) |
| ILP within one mul | None (serial carry) | Full (4 independent lanes) | Moderate (2-wide interleave) |
| Delayed reduction? | Yes (wide accumulators) | No (per-op reduction) | No (per-op reduction) |
| Primary bottleneck | Carry chain latency | NEON multiply throughput | Scalar multiply latency |

**Key takeaway**: Plonky3's choice of small primes (31-bit) is itself the biggest optimization — it enables NEON SIMD vectorization that simply isn't available for 256-bit fields. The 4× throughput advantage of NEON `uint32x4_t` over scalar 64-bit math is the dominant factor.

## References

- [Plonky3 repository](https://github.com/Plonky3/Plonky3) — Source code
- [ARM Architecture Reference Manual](https://developer.arm.com/documentation/ddi0602/latest/) — AArch64 instruction reference
- [Modern Computer Arithmetic](https://members.loria.fr/PZimmermann/mca/pub226.html) — Brent & Zimmermann, Algorithm 2.7 (Montgomery multiplication)
- [Rust core::arch::aarch64](https://doc.rust-lang.org/core/arch/aarch64/) — NEON intrinsics documentation
