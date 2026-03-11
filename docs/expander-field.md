# Expander: aarch64 NEON Field Arithmetic Analysis

## Overview

This document analyzes how the [Expander](https://github.com/PolyhedraZK/Expander) ZK prover uses ARM NEON SIMD intrinsics to accelerate field operations on Apple Silicon. Expander targets **small prime fields** (31-bit) and **binary extension fields** (GF(2^128)), which fit naturally into NEON's 128-bit / 4×u32 vector lanes — a fundamentally different regime from Spartan2's 256-bit BN254 field.

### Key Architectural Difference from Spartan2

| Aspect | Spartan2 (BN254) | Expander (BabyBear/M31) |
|---|---|---|
| Element size | 256 bits (4×64 limbs) | 31 bits (1×u32) |
| Elements per NEON register | 0 (doesn't fit) | 4 |
| SIMD strategy | Scalar `mul`/`umulh` carry chains | NEON `uint32x4_t` parallel lanes |
| Hot instruction | `mul`/`umulh` (64×64→128) | `sqdmulh`/`vmulq` (32×32→high32) |
| Parallelism model | ILP batching across accumulators | Data-parallel (16 elements at once) |

Expander packs **16 field elements** into `[uint32x4_t; 4]` (4 NEON registers × 4 lanes each) and operates on all 16 simultaneously. This is viable because each element fits in a single 32-bit lane.

## Fields Implemented

| Field | Modulus P | Representation | NEON Vectorized? |
|---|---|---|---|
| BabyBear | 2^31 - 2^27 + 1 = 0x78000001 | Montgomery | Yes (deeply optimized) |
| Mersenne-31 | 2^31 - 1 | Standard | Yes |
| GF(2^128) | x^128 + x^7 + x^2 + x + 1 | Polynomial (bitwise) | Yes (PMULL) |
| Goldilocks | 2^64 - 2^32 + 1 | Standard | **No** (scalar fallback) |

Goldilocks uses scalar operations because 64-bit elements only yield 2 lanes per 128-bit register — insufficient to justify vectorization overhead. The code comment: *"Working on vectors seems to be slower since we only pack 2 elements per slot."*

## 1. BabyBear: Montgomery Multiplication via NEON

**Source**: `arith/babybear/src/babybearx16/babybear_neon.rs`

This is the most sophisticated optimization in the codebase. It implements Montgomery multiplication using 7 NEON instructions by exploiting the `SQDMULH` instruction's semantics.

### Mathematical Foundation

**Montgomery multiplication** computes D = a · b · B^(-1) mod P for B = 2^32.

Given:
- P = 0x78000001 (31-bit prime)
- B = 2^32
- μ = P^(-1) mod B

The algorithm (Brent & Zimmermann, Algorithm 2.7):

```
Input:  a, b ∈ [0, P)
Output: D ≡ a · b · B^(-1) (mod P), with D ∈ [0, P)

1. C := a · b                         // full product, ≤ P^2 < 2^62
2. Q := smod_B(μ · C)                 // Q = (μ·C) mod B, signed: Q ∈ [-B/2, B/2)
3. D := (C - Q·P) / B                 // exact division (C ≡ Q·P mod B)
4. if D < 0: D := D + P               // normalize to [0, P)
```

**Why step 3 is exact**: Q·P ≡ μ·C·P ≡ P^(-1)·C·P ≡ C (mod B), so B | (C - Q·P).

**Why D is in range**: Since |C| ≤ P^2 and |Q·P| ≤ (B/2)·P, we get |C - Q·P| < P·B, hence |D| < P.

### The SQDMULH Trick

The critical insight is using `SQDMULH` (Signed Saturating Doubling Multiply High), which computes:

```
sqdmulh(a, b) = ⌊(2 · a · b) / 2^32⌋ = (a · b) >> 31
```

This gives bits [31..62] of the product — one bit more than a standard high-half multiply. The algorithm needs `(C - Q·P) / 2^32`, but `SQDMULH` gives values shifted by 31 (not 32). The fix: use `SHSUB` (Signed Halving Subtract), which computes `(a - b) / 2` in one instruction:

```
D = (C - Q·P) / 2^32
  = ((C >> 31) - (Q·P >> 31)) / 2
  = shsub(sqdmulh(a, b), sqdmulh(Q, P))
```

### Assembly: 7 Instructions for 4 Multiplications

```
; Inputs: lhs.4s, rhs.4s  (4 field elements each, in uint32x4_t)
; Constants: MU.4s = [P^(-1) mod 2^32; 4],  P.4s = [0x78000001; 4]
;
; Math per lane:
;   C     = lhs · rhs
;   Q     = smod_B(μ · C) = (lhs · (μ · rhs)) mod 2^32
;   c_hi  = C >> 31
;   qp_hi = (Q · P) >> 31
;   D     = (c_hi - qp_hi) / 2
;   res   = D + P  if D < 0,  else D

sqdmulh  c_hi.4s,  lhs.4s,  rhs.4s       ; c_hi  = (2·lhs·rhs) >> 32  = C >> 31
mul      mu_rhs.4s, rhs.4s, MU.4s         ; mu_rhs = μ · rhs  (mod 2^32)
mul      q.4s,     lhs.4s,  mu_rhs.4s     ; Q = lhs · mu_rhs  (mod 2^32) = smod_B(μ·C)
sqdmulh  qp_hi.4s, q.4s,   P.4s          ; qp_hi = (2·Q·P) >> 32  = (Q·P) >> 31
shsub    d.4s,     c_hi.4s, qp_hi.4s     ; D = (c_hi - qp_hi) / 2 = (C - Q·P) / 2^32
cmgt     underflow.4s, qp_hi.4s, c_hi.4s ; underflow = (D < 0) ? -1 : 0
mls      res.4s,   underflow.4s, P.4s     ; res = D - underflow·P  (adds P if negative)
```

**Throughput**: 1.75 cycles per vector = **2.29 elements/cycle** (4 muls in 1.75 cyc)

**Latency**: 11 cycles (lhs path), 14 cycles (rhs path, through mu_rhs)

### Instruction Budget Comparison

| Operation | Instructions | Throughput (cyc/vec) | Elements/cycle |
|---|---|---|---|
| Montgomery mul | 7 | 1.75 | 2.29 |
| Add | 3 | 0.75 | 5.33 |
| Sub | 3 | 0.75 | 5.33 |
| Neg | 3 | 0.75 | 5.33 |

## 2. BabyBear: Branchless Addition and Subtraction

### Addition

**Math**: (a + b) mod P, where a, b ∈ [0, P). So a + b ∈ [0, 2P - 2).

**Insight**: Since 2P - 2 < 2^32, unsigned wrapping handles everything. Let t = a + b and u = t - P (mod 2^32). If t < P, then u wraps to a large value > P, and `UMIN` picks t. If t ≥ P, then u = t - P ∈ [0, P-2], and `UMIN` picks u.

```
add   t.4s, lhs.4s, rhs.4s     ; t = a + b
sub   u.4s, t.4s,   P.4s       ; u = (a + b - P) mod 2^32
umin  res.4s, t.4s, u.4s       ; res = min(t, u) unsigned → correct result
```

### Subtraction

**Math**: (a - b) mod P. If a ≥ b, result = a - b. If a < b, result = a - b + P.

```
sub   diff.4s,      lhs.4s, rhs.4s       ; diff = (a - b) mod 2^32
cmhi  underflow.4s, rhs.4s, lhs.4s       ; underflow = (b > a) ? 0xFFFFFFFF : 0
mls   res.4s,       underflow.4s, P.4s   ; res = diff - (-1)·P = diff + P  (if underflow)
```

### The `confuse_compiler` Hack

For subtraction, the compiler recognizes that `underflow` is either 0 or 0xFFFFFFFF and tries to replace `MLS` (multiply-subtract) with `AND + ADD`, which is **slower on Apple M1**. To prevent this, Expander uses an inline assembly barrier:

```rust
fn confuse_compiler(x: uint32x4_t) -> uint32x4_t {
    let y;
    unsafe {
        asm!(
            "/*{0:v}*/",                    // no-op: just a comment referencing the register
            inlateout(vreg) x => y,
            options(nomem, nostack, preserves_flags, pure),
        );
        // Tell the compiler x == y so it can still constant-fold
        if transmute::<_, [u32; 4]>(x) != transmute::<_, [u32; 4]>(y) {
            unreachable_unchecked();
        }
    }
    y
}
```

This is a no-op in the generated code, but it breaks LLVM's knowledge of the value, preventing the AND+ADD transform while still allowing constant folding through the `unreachable_unchecked` hint. A clever workaround for an LLVM mis-optimization on Apple Silicon.

### Negation

**Math**: -a mod P = P - a if a ≠ 0, else 0.

```
sub   t.4s,       P.4s, val.4s       ; t = P - val
cmeq  is_zero.4s, val.4s, #0         ; is_zero = (val == 0) ? 0xFFFFFFFF : 0
bic   res.4s,     t.4s,  is_zero.4s  ; res = t AND (NOT is_zero)
```

Uses `BIC` (bit clear) instead of a branch: if val = 0, `is_zero` is all-ones and `BIC` clears t to 0. Otherwise `is_zero` is 0 and `BIC` passes t through unchanged.

## 3. Mersenne-31: Exploiting P = 2^31 - 1

**Source**: `arith/mersenne31/src/m31x16/m31_neon.rs`

Mersenne primes allow a simpler reduction because 2^31 ≡ 1 (mod P).

### Multiplication

**Math**: a · b mod (2^31 - 1).

For any product C = a · b (up to 62 bits):

```
C = C_lo + C_hi · 2^31     where C_lo = C mod 2^31, C_hi = C >> 31
  ≡ C_lo + C_hi  (mod 2^31 - 1)
```

The NEON translation uses `SQDMULH` to get C >> 31 (the high part) and `MUL` to get C mod 2^32 (the low part), then reconstructs:

```
sqdmulh  prod_hi.4s, a.4s, b.4s     ; prod_hi ≈ (a·b) >> 31
mul      prod_lo.4s, a.4s, b.4s     ; prod_lo = (a·b) mod 2^32
mls      t.4s, prod_lo.4s, prod_hi.4s, P.4s   ; t = prod_lo - prod_hi·(2^31 - 1)
                                               ;   = prod_lo + prod_hi - prod_hi·2^31
                                               ;   ≡ C_lo + C_hi  (mod P)
umin     res.4s, t.4s, sub(t, P)    ; final reduction
```

The `MLS` instruction computes `prod_lo - prod_hi × P`, and since P = 2^31 - 1, this algebraically equals `(C mod 2^31) + (C >> 31)` — the standard Mersenne reduction.

### Addition / Subtraction / Reduction

All use the same `UMIN`-based branchless pattern:

```
; reduce_sum: given x that may be ≥ P, produce x mod P
umin  res.4s, x.4s, sub(x, P).4s

; add: same as BabyBear
add   t.4s, a.4s, b.4s
umin  res.4s, t.4s, sub(t, P).4s

; sub: uses add-then-umin instead of cmhi+mls
sub   diff.4s, a.4s, b.4s
add   u.4s, diff.4s, P.4s
umin  res.4s, diff.4s, u.4s
```

The subtraction approach differs from BabyBear — it uses `ADD P + UMIN` instead of `CMHI + MLS`. Both are 3 instructions, but the M31 variant avoids the need for `confuse_compiler` since `UMIN` doesn't trigger the problematic LLVM transform.

### Multiply-by-constants

`mul_by_2` and `mul_by_5` use left-shift + reduce chains:

```
; mul_by_2:
shl    double.4s, x.4s, #1          ; double = x << 1 = 2x
umin   res.4s, double, sub(double, P)

; mul_by_5 = 4x + x:
shl    double.4s, x.4s, #1          ; 2x
reduce double
shl    quad.4s, double.4s, #1        ; 4x
reduce quad
add    res.4s, quad.4s, x.4s        ; 4x + x = 5x
reduce res
```

## 4. GF(2^128): Carryless Multiplication with PMULL

**Source**: `arith/gf2_128/src/gf2_ext128/neon.rs`

GF(2^128) = GF(2)[x] / f(x) where f(x) = x^128 + x^7 + x^2 + x + 1.

### Addition

In characteristic 2, addition = XOR. One instruction:

```
eor  res.16b, a.16b, b.16b
```

Subtraction is identical to addition (a - b = a + b in GF(2)).

### Multiplication: Karatsuba + PMULL

**Math**: Given a(x) = a_0 + a_1·x^64 and b(x) = b_0 + b_1·x^64 (splitting at degree 64):

```
a · b = a_0·b_0 + (a_0·b_1 + a_1·b_0)·x^64 + a_1·b_1·x^128
```

Using Karatsuba's identity to save one multiplication:

```
a_0·b_1 + a_1·b_0 = (a_0 + a_1)·(b_0 + b_1) - a_0·b_0 - a_1·b_1
```

(In GF(2), subtraction = XOR = addition, so - is the same as +.)

**Step 1: Three 64×64 → 128 polynomial multiplications** using `PMULL`:

```
pmull    lo.1q,    a.1d,    b.1d           ; a_0 · b_0
pmull2   hi.1q,    a.2d,    b.2d           ; a_1 · b_1
ext      a_rot.16b, a.16b, a.16b, #8      ; swap halves: [a_1, a_0]
ext      b_rot.16b, b.16b, b.16b, #8
eor      a_sum.16b, a_rot.16b, a.16b      ; [a_0⊕a_1, a_0⊕a_1]
eor      b_sum.16b, b_rot.16b, b.16b      ; [b_0⊕b_1, b_0⊕b_1]
pmull    mid.1q,   a_sum.1d, b_sum.1d      ; (a_0⊕a_1) · (b_0⊕b_1)
eor      mid, mid, lo                      ; subtract a_0·b_0
eor      mid, mid, hi                      ; subtract a_1·b_1 → cross term
```

`PMULL` (`vmull_p64`) is a hardware instruction that performs **carryless (polynomial) multiplication** of two 64-bit values, producing a 128-bit result. This is the GF(2) equivalent of integer multiply.

**Step 2: Assemble 256-bit result** by positioning the cross term:

```
; 256-bit product = [hi : mid : lo] with overlap
ext    mid_hi, mid, zero, #8              ; upper 64 bits of cross term
ext    mid_lo, zero, mid, #8              ; lower 64 bits of cross term
eor    lo, lo, mid_lo                     ; low 128 bits
eor    hi, hi, mid_hi                     ; high 128 bits
```

**Step 3: Reduce modulo f(x) = x^128 + x^7 + x^2 + x + 1.**

Since x^128 ≡ x^7 + x^2 + x + 1, each high bit at position 128+k maps to bits at k+7, k+2, k+1, k. The shifts 31, 30, 25 are the complements within 32-bit words (32 - 1, 32 - 2, 32 - 7):

```
; hi.4s holds the high 128 bits (coefficients of x^128 through x^255)
ushr   t7.4s,  hi.4s, #31               ; for x^1 contribution
ushr   t2.4s,  hi.4s, #30               ; for x^2 contribution
ushr   t1.4s,  hi.4s, #25               ; for x^7 contribution
eor    t.4s, t7.4s, t2.4s
eor    t.4s, t.4s,  t1.4s
ext    t_rot, t, t, #12                 ; rotate by 32 bits for cross-word carries
; ... mask, split, and XOR back into lo ...
shl    s1.4s, hi.4s, #1                 ; x << 1
shl    s2.4s, hi.4s, #2                 ; x << 2
shl    s7.4s, hi.4s, #7                 ; x << 7
eor    lo, lo, s1
eor    lo, lo, s2
eor    lo, lo, s7
eor    res, lo, hi                       ; final result
```

### Multiply by x

Shifts the 128-bit polynomial left by 1. If the top coefficient (x^127) was set, reduces by XOR with 0x87 (the low terms of the irreducible polynomial):

```
; a = [lo_64, hi_64]
shl    shifted.2d, a.2d, #1             ; left shift each 64-bit half
; carry bit 63 of lo_64 into bit 0 of hi_64
orr    shifted, shifted, carry_mask
; if bit 127 was set, XOR with reduction constant
eor    res.2d, shifted.2d, [0x87, 0].2d ; conditional on extracted high bit
```

## 5. Goldilocks: No NEON Vectorization

**Source**: `arith/goldilocks/src/goldilocksx8/goldilocks_neon.rs`

Goldilocks (P = 2^64 - 2^32 + 1) elements are 64 bits wide. A 128-bit NEON register holds only 2 elements — too few to amortize vectorization overhead. Expander falls back to scalar operations on an array of 8 elements:

```rust
fn mul_internal(a: &NeonGoldilocks, b: &NeonGoldilocks) -> NeonGoldilocks {
    let mut res = NeonGoldilocks::zero();
    for i in 0..8 {
        res.v[i] = a.v[i] * b.v[i];  // scalar Goldilocks multiply
    }
    res
}
```

This is a design data point: NEON vectorization is only worthwhile when ≥ 4 elements fit per register.

## 6. Summary: Key NEON Techniques for Field Arithmetic

### Instruction Selection Patterns

| Pattern | Instructions | Use Case |
|---|---|---|
| Branchless modular reduction | `ADD/SUB` + `UMIN` | When sum/diff ∈ [0, 2P) |
| Conditional correction | `CMxx` + `MLS` | When result may be negative |
| Bit clear for zero check | `CMEQ #0` + `BIC` | Negation (avoid branch on zero) |
| Montgomery high-half | `SQDMULH` | Gets (a·b) >> 31 in one instruction |
| Montgomery exact halving | `SHSUB` | Computes (a - b) / 2 in one instruction |
| Polynomial multiply | `PMULL` | GF(2) carryless 64×64 → 128 multiply |

### Compiler Workarounds

The `confuse_compiler` pattern (BabyBear subtraction/reduction) is notable: it uses inline assembly as a value barrier to prevent LLVM from replacing `MLS` with `AND + ADD`. On Apple M1, `MLS` on NEON vector registers is faster than the alternative sequence. The `unreachable_unchecked` hint restores constant folding without letting LLVM see the identity transform.

This is a reminder that **intrinsics don't guarantee instruction selection** — the compiler can and will rearrange equivalent expressions, sometimes suboptimally for a specific microarchitecture.

### Throughput Summary (per 4-element NEON vector)

| Field | Operation | Instructions | Cycles/vec | Elements/cycle |
|---|---|---|---|---|
| BabyBear | Multiply (Montgomery) | 7 | 1.75 | 2.29 |
| BabyBear | Add | 3 | 0.75 | 5.33 |
| BabyBear | Sub | 3 | 0.75 | 5.33 |
| BabyBear | Neg | 3 | 0.75 | 5.33 |
| Mersenne-31 | Multiply | 4 | ~1.0 | ~4.0 |
| Mersenne-31 | Add | 3 | 0.75 | 5.33 |
| GF(2^128) | Add (XOR) | 1 | 0.25 | 4.0 |
| GF(2^128) | Multiply (Karatsuba) | ~25 | ~8 | 0.125 |

### Relevance to Spartan2

These techniques don't directly apply to BN254 (256-bit field elements don't fit in NEON lanes). However, the patterns are instructive:

1. **`SQDMULH` for extracting high bits** — could inspire similar use of `UMULH` patterns in 4×64 scalar code
2. **`SHSUB` for exact halving** — a reminder to look for compound operations that map to single AArch64 instructions
3. **Branchless reduction via `UMIN`** — the principle (exploit unsigned wraparound to avoid branches) applies to any field
4. **Compiler barriers** — when LLVM generates suboptimal instruction sequences on M1, targeted inline asm barriers can fix it without writing full hand-rolled assembly
5. **Know when NOT to vectorize** — Goldilocks (64-bit) doesn't use NEON; similarly, BN254 (256-bit) likely won't benefit from NEON lane operations

## References

- [Expander source](https://github.com/PolyhedraZK/Expander) — `arith/` directory
- [Modern Computer Arithmetic](https://members.loria.fr/PZimmermann/mca/mca-cup-0.5.9.pdf), Brent & Zimmermann — Algorithm 2.7 (Montgomery multiplication)
- [ARM NEON Intrinsics Reference](https://developer.arm.com/architectures/instruction-sets/intrinsics/)
- [Plonky3 field implementations](https://github.com/Plonky3/Plonky3) — Expander's BabyBear Montgomery mul is adapted from Plonky3's `monty-31` crate
