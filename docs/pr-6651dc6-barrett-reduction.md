# PR: Add small-value delayed reduction with Barrett algorithm

**Commit:** `6651dc629ac68504c59a9a20eda6e1564b17781b`

## Summary

This PR adds infrastructure for delayed modular reduction when accumulating field × small-integer products. This is a prerequisite for the upcoming small-value sumcheck optimization, which is a significantly larger change.

## Motivation

In the sumcheck protocol, we frequently accumulate sums of products: Σ(field × value). Previously, we supported delayed reduction only for **field × field** products:

```
field × field → Both operands in Montgomery form (factor R²)
             → Use Montgomery REDC (9-limb → 4-limb)
             → Final result automatically in Montgomery form
```

However, small-value sumcheck uses coefficients that fit in native integers (i32, i64, i128). These are **not** in Montgomery form:

```
field × small_int → Field in Montgomery form, small_int is raw
                  → Product has factor R (not R²)
                  → Montgomery REDC would divide by R, giving wrong result
```

## Why Barrett Reduction?

**Montgomery reduction** computes `x × R⁻¹ mod p`. When both operands are in Montgomery form, `(a·R) × (b·R) = ab·R²`, and REDC gives `ab·R` — exactly what we want.

**Barrett reduction** computes `x mod p` directly, without any R factor. When accumulating `(field·R) × small_int`, the product is `field·R·small_int`. Barrett gives us `field·small_int mod p` in Montgomery form — exactly right.

The algorithm replaces expensive division by multiplication with a precomputed reciprocal:

```
μ = ⌊2^512 / p⌋           // precomputed constant

To compute x mod p:
1. q = ⌊(x × μ) / 2^512⌋  // quotient estimate via multiply + shift
2. r = x − q × p           // remainder
3. if r ≥ p: r −= p        // conditional correction
```

## Implementation Efficiency

The implementation is optimal with **at most 1 conditional subtract** for finalization:

- For BN254 (where `2p < 2^256`): The remainder fits in 4 limbs, enabling a fast path with `mul_3x4_lo4` and a single `if r ≥ p: r -= p`
- For fields like T256 (where `2p ≈ 2^256`): Uses 5-limb arithmetic with one correction

The bound is tight because:
- Barrett's quotient estimate `q` satisfies `q ≤ ⌊x/p⌋ ≤ q + 2`
- Therefore `r = x - q·p` satisfies `0 ≤ r < 3p`
- After one subtraction: `0 ≤ r < 2p`
- After at most one more: `0 ≤ r < p` ✓

In practice for our field sizes, a single correction suffices (verified by `debug_assert!`).

## Verification of Barrett Constants

All precomputed constants are verified at test time against reference implementations using `num-bigint`:

| Constant | Verification |
|----------|--------------|
| `BARRETT_MU` | Computed as `⌊2^512 / p⌋` using BigUint, compared against hardcoded limbs |
| `R384_MOD` | Computed as `2^384 mod p` using BigUint, compared against hardcoded limbs |
| `USE_4_LIMB_BARRETT` | Implicitly verified — if wrong, Barrett reduction would produce non-canonical results caught by `debug_assert!` |

The test helpers in `field_reduction_constants.rs` implement this:

```rust
pub(crate) fn test_barrett_mu_impl<F: BarrettReductionConstants>() {
  let p = limbs_to_biguint(&F::MODULUS);
  let two_pow_512 = BigUint::from(1u64) << 512;
  let expected = &two_pow_512 / &p;
  let actual = limbs5_to_biguint(&F::BARRETT_MU);
  assert_eq!(actual, expected, "BARRETT_MU mismatch");
}

pub(crate) fn test_r384_mod_impl<F: BarrettReductionConstants>() {
  let p = limbs_to_biguint(&F::MODULUS);
  let two_pow_384 = BigUint::from(1u64) << 384;
  let expected = &two_pow_384 % &p;
  let actual = limbs_to_biguint(&F::R384_MOD);
  assert_eq!(actual, expected, "R384_MOD mismatch");
}
```

Each field provider (BN254, Pasta, P256, T256) runs these tests via macros:

```rust
crate::test_barrett_reduction_constants!(scalar_brc, Scalar);
crate::test_barrett_reduction!(scalar_br, Scalar, barrett_reduce_6::<Scalar>);
```

Additionally, the Barrett reduction itself is tested against field arithmetic:
- **Single product**: `reduce(field × small) == field * F::from(small)`
- **Sum of products**: `reduce(Σ field_i × small_i) == Σ (field_i * F::from(small_i))`
- **Stress test**: 2000 random products accumulated and reduced (release builds only)

## Future Work: Specialized Pasta Reduction (Already Implemented)

The Pasta curves (Pallas and Vesta) have primes in a special **pseudo-Mersenne** form that enables significantly faster reduction. From `src/provider/pasta.rs`:

```rust
// Pallas scalar field (Fq):
"40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001"

// Vesta scalar field (Fp):
"40000000000000000000000000000000224698fc094cf91b992d30ed00000001"
```

Breaking down the structure in 64-bit limbs (little-endian):

```
         limb[3]            limb[2]            limb[1]            limb[0]
    ┌──────────────────┬──────────────────┬──────────────────┬──────────────────┐
Fq: │ 4000000000000000 │ 0000000000000000 │ 224698fc0994a8dd │ 8c46eb2100000001 │
Fp: │ 4000000000000000 │ 0000000000000000 │ 224698fc094cf91b │ 992d30ed00000001 │
    └──────────────────┴──────────────────┴──────────────────┴──────────────────┘
           ↑                    ↑                         ↑
          2^254            always zero              ε (~125 bits)
```

Both primes have the form `p = 2²⁵⁴ + ε` where:
- `limb[3] = 0x4000000000000000` contributes exactly `2^62 × 2^192 = 2^254`
- `limb[2] = 0` (the 128-191 bit range is empty)
- `ε = limb[0] + limb[1] × 2^64` spans only ~125 bits

This structure enables **Solinas reduction**, which exploits the congruence `2^254 ≡ -ε (mod p)`:

```
To reduce x (a 5-6 limb value):
1. Split: x = x_lo + x_hi × 2^254       // x_lo is low 254 bits
2. Fold:  r = x_lo - x_hi × ε           // multiply by ~125-bit ε, not 256-bit p
3. Correct: if r < 0 or r ≥ p: adjust
```

This is faster than generic Barrett because:

| Operation | Generic Barrett | Pasta Solinas |
|-----------|-----------------|---------------|
| Main multiply | 3×5 = 15 muls (q × μ) | 4×2 = 8 muls (x_hi × ε) |
| Secondary multiply | 3×4 = 12 muls (q × p) | — (folded inline) |
| Total muls | ~27 | ~8 |

**Note:** The specialized `pasta_reduce_6` implementation is already complete and tested. It will be included in a follow-up PR after this one is approved and merged, to keep the review scope manageable.

## What's Included

| Component | Purpose |
|-----------|---------|
| `BarrettReductionConstants` | Trait with μ = ⌊2^512/p⌋, R384_MOD, and fast-path flag |
| `barrett_reduce_6/7` | Generic Barrett for 6-7 limb inputs |
| `SignedWideLimbs<N>` | Accumulator for signed products (separate pos/neg) |
| `SmallValueField<V>` | Trait for small integer ↔ field conversion |
| `DelayedReduction<i32/i64/i128>` | Accumulation + single reduction |

## Overflow Capacity

| Accumulator | Product Type | Capacity |
|-------------|--------------|----------|
| `WideLimbs<6>` | field × i32/i64 | 2^66 products |
| `SignedWideLimbs<7>` | field × i128 | 2^66 products |

Sumcheck polynomials are bounded by practical sizes (≤2^40), so overflow is impossible.

## Why Split This Out?

The small-value sumcheck PR that uses this infrastructure is substantially larger. Splitting the Barrett reduction foundation into its own PR makes review more manageable and establishes a clean abstraction boundary.
