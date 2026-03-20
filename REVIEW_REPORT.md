# Security & Correctness Review Report

**Branch:** `parallel-shuffling`
**Commits reviewed:** `02af1cf..HEAD` (5 commits)
**Date:** 2026-03-19
**Reviewer:** Claude Opus 4.6 (automated deep review)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Critical Issues](#critical-issues)
3. [Warnings](#warnings)
4. [Per-Commit Reviews](#per-commit-reviews)
5. [Adversarial Analysis](#adversarial-analysis)
6. [Suggestions](#suggestions)

---

## Executive Summary

Five commits were reviewed covering parallel re-encryption for an RS shuffle, variable count optimization, scalar multiplication optimization, sparse matrix evaluation (CSC + column remapping), and Pippenger MSM with signed digits.

**Four of five commits are mathematically sound** with no critical issues. The one commit with critical issues (`57caee3`) has two fundamental soundness gaps in the circuit linkage — the permutation proven by the grand product is not connected to the ciphertext allocation at the constraint level, and no public inputs exist for the verifier to check. The infrastructure to fix this (`IndexedCiphertext` with 5-challenge compression) already exists in the codebase but is not yet wired up.

| Commit | Description | Verdict | Critical | Warnings |
|--------|-------------|---------|----------|----------|
| `57caee3` | Parallel re-encryption for RS shuffle | **UNSOUND** | 3 | 5 |
| `547e1b4` | Reduce variable count below power-of-2 boundary | Sound | 0 | 1 |
| `12628d6` | Scalar mul with fixed-base and shared power tables | Sound | 0 | 2 |
| `5337265` | CSC + column remapping for eval_sparse | Sound | 0 | 2 (pre-existing) |
| `6c9ad68` | Pippenger MSM: signed digits, optimal window | Sound | 0 | 4 |

---

## Critical Issues

### C1. Permutation and ciphertext allocation are not linked in the constraint system

**Commit:** `57caee3`
**File:** `examples/rs_shuffle_bp_full.rs`, lines 147–189
**Severity:** Soundness break

The circuit has two independent parts:

- **Part 1 (lines 147–171):** Grand product checks on `witness_var.uns_levels` and `witness_var.sorted_levels`. These constrain the `idx` values to form a valid permutation σ through all RS levels. The final level's `sorted_levels[LEVELS-1][i].idx` is an `AllocatedNum` that tells the circuit which original element ended up at position `i`.

- **Part 2 (lines 177–189):** Ciphertext allocation uses `self.permutation[i]` — a **native Rust `usize`**, not a circuit variable — to select which input ciphertext to place at each position:

```rust
let src = self.permutation[i];       // native indexing, NOT constrained
let ct_var = ElGamalCiphertextVar::<ECEngine>::alloc(
    cs.namespace(|| format!("shuffled_ct_{}", i)),
    &self.input_ciphertexts[src],    // fresh private witness, unconstrained
)?;
```

The `sorted_levels[LEVELS-1][i].idx` (the circuit-level permutation output) is **never referenced** in Part 2. The ciphertexts are allocated as fresh private witnesses with no constraint tying them to the proven permutation.

**Impact:** A malicious prover can pass the grand product with permutation σ but allocate ciphertexts in any arbitrary order σ'. The verifier cannot detect the mismatch. This is a complete break of shuffle correctness.

**Root cause:** The `IndexedCiphertext` type (in `permutation.rs:102–246`) exists with 5-challenge compression (`α₀·idx + α₁·c1.x + α₂·c1.y + α₃·c2.x + α₄·c2.y`) but is not wired up. The `num_challenges` comment says `2 + 5` but only returns `2`:

```rust
fn num_challenges(&self) -> usize {
    // 2 for permutation grand product + 5 for indexed ciphertext grand product
    2   // ← should be 7 once linked
}
```

**Fix:** Add an indexed ciphertext grand product that constrains:
```
∏ compress(i, input_ct[i])  =  ∏ compress(sorted_levels[LEVELS-1][i].idx, shuffled_ct[i])
```
This uses the **circuit-level** `idx` from the permutation proof to tie it to the ciphertext variables.

---

### C2. No public inputs — proof is vacuous to the verifier

**Commit:** `57caee3`
**File:** `examples/rs_shuffle_bp_full.rs`, lines 93–94, 214
**Severity:** Soundness break

```rust
fn public_values(&self) -> Result<Vec<Scalar>, SynthesisError> {
    Ok(vec![])  // no public inputs
}
```

Neither input ciphertexts nor output ciphertexts are exposed as public inputs. The output deck is assigned to `_output_deck` (unused). The proof only demonstrates "there exist some inputs and outputs related by re-encryption" — the verifier has no way to check *what* was shuffled or *what* came out.

**Impact:** The proof provides zero guarantees to a verifier. For a shuffle proof to be meaningful, at minimum the input and output decks must be committed to or exposed as public inputs.

**Fix:** Use `ElGamalCiphertextVar::alloc_input` (which already exists in `data_structures.rs:318–338` and calls `inputize` on all coordinates) for the input and output ciphertexts, or commit to them via the precommitted mechanism.

---

### C3. Generator power table is unconstrained — prover can use arbitrary basis

**Commit:** `57caee3`
**File:** `src/rs_shuffle_bp/encryption.rs`, `scalar_mul_fixed_base`; `examples/rs_shuffle_bp_full.rs`, lines 305–307
**Severity:** Soundness break (in recursive/multi-shuffler setting)

The generator's power table `gen_powers[i] = 2^i · G` is precomputed natively and passed into the circuit as **constant coefficients** via `scalar_mul_fixed_base`. These values are baked into the R1CS shape (and thus the verification key) but are never constrained in-circuit.

```rust
let gen_powers = precompute_fixed_base_powers(gen_coords, curve_a, num_bits);
```

In a single-proof setting this is safe — the VK is a trusted artifact. But in the intended protocol, multiple shufflers shuffle on top of each other, and a **recursive verifier** aggregates their proofs. The recursive verifier takes VK digests as inputs but cannot inspect the R1CS matrices to verify that the correct generator was used. A malicious shuffler could bake in a different generator (e.g., one where they know the discrete log of PK), making their re-encryption trivially reversible.

By contrast, the PK power table (`pk_powers`) is already computed **in-circuit** via `double_incomplete` and is therefore constrained.

**Impact:** In a multi-shuffler recursive setting, a malicious shuffler can use an arbitrary generator without detection. This breaks the unlinkability guarantee of the re-encryption.

**Fix:** Compute `gen_powers` in-circuit via `double_incomplete` (same as PK), making G a public input. This adds ~1,016 variables (254 doublings, shared across all 52 cards) — a one-time cost. The `scalar_mul_fixed_base` optimization must be replaced with `scalar_mul_with_powers` using the in-circuit table.

---

## Warnings

### W1. Parallel witness values are not constraint-checked during generation

**Commit:** `57caee3`
**File:** `src/rs_shuffle_bp/encryption.rs`, `WitnessCS::enforce` (no-op)

The `WitnessCS` type's `enforce` method does nothing. During the parallel witness path, no constraints are verified. If native/gadget computation diverges, this would only surface at proof verification time. The test `test_parallel_witness_synthesis` compares the count of aux variables between parallel and serial paths, but does not compare values element-by-element.

---

### W2. Parallel witness aux variable ordering is fragile

**Commit:** `57caee3`
**File:** `src/rs_shuffle_bp/encryption.rs`, parallel witness path

Correctness depends on each mini-CS producing exactly the same number of aux variables in the same order as the serial path. The current gadget is deterministic in allocation, but any future change introducing data-dependent allocation would silently corrupt the witness.

---

### W3. Example uses small deterministic randomization scalars

**Commit:** `57caee3`
**File:** `examples/rs_shuffle_bp_full.rs`, line 284

```rust
randomizations.push(Scalar::from((i + 10) as u64));
```

Values 10–61 are tiny scalars. Bit decomposition will have mostly-zero high bits, potentially masking bugs in scalar multiplication for high bits. Production must use uniformly random field elements.

---

### W4. `native_scalar_mul` returns wrong result for scalar = 0

**Commit:** `57caee3`
**File:** `src/rs_shuffle_bp/encryption.rs`, `native_scalar_mul`

When scalar = 0, the function returns the base point instead of the identity. While randomization scalars should never be zero in practice (Pr = 2⁻²⁵⁵), the function's mathematical contract is violated.

---

### W5. `precompute_fixed_base_powers` panics on points of small order

**Commit:** `57caee3`
**File:** `src/rs_shuffle_bp/encryption.rs`, `dz.invert().unwrap()`

If the base point has order < `num_bits`, a doubling will hit the point at infinity (Z=0) and panic. Safe for Pallas/Vesta generators (large prime order) but the precondition is undocumented.

---

### W6. `sub_incomplete` unsatisfiable for scalar = 1 (completeness issue)

**Commit:** `547e1b4`
**File:** `src/gadgets/ecc.rs`, `scalar_mul_non_infinity`

When scalar = 1, the slack removal computes `acc - base` where `acc == base`. This triggers `sub_incomplete` with `self.x == other.x`, making the lambda constraint unsatisfiable. This is a **completeness** issue (proof generation fails), not a soundness issue (no false proofs). Probability for random scalars: 2⁻²⁵⁵.

---

### W7. `add_incomplete` precondition could theoretically be violated

**Commits:** `547e1b4`, `12628d6`
**File:** `src/gadgets/ecc.rs`

Both `scalar_mul_non_infinity` and `scalar_mul_with_powers` use incomplete addition which assumes the accumulator never collides with a power table entry (x₁ ≠ x₂). Violating this requires solving a discrete log problem — **not practically exploitable**, but the assumption should be documented.

---

### W8. Parallel witness allocates `pk_powers` separately in each mini_cs

**Commit:** `12628d6`
**File:** `src/rs_shuffle_bp/encryption.rs`, `reencrypt_deck_parallel_witness`

Each mini_cs allocates pk_power values as fresh variables. The `input_aux_count` correctly tracks how many to skip when copying back. No bug, but this is a subtle correctness-critical section — any miscounting would silently corrupt the witness.

---

### W9. Pre-existing: SmallCoeff single-entry fast path assumes ±1

**Commit:** `5337265` (pre-existing, not introduced)
**File:** `src/r1cs/mod.rs`, `accumulate_column` for SmallCoeff

```rust
if start + 1 == end {
    return if values[start].is_positive() {
        rx_vals[row_indices[start] as usize]
    } else {
        -rx_vals[row_indices[start] as usize]
    };
}
```

A single-entry column with a non-unit coefficient (e.g., value=2) is incorrectly treated as ±1. The field-path `accumulate_column_field` correctly handles this with a three-way branch.

---

### W10. Pre-existing: Panic risk on malformed column indices

**Commit:** `5337265` (pre-existing, not introduced)
**File:** `src/r1cs/mod.rs`, `col_to_dense[ci]`

Will panic if `ci >= num_buf_cols`. The invariant that all column indices are < `num_buf_cols` holds by construction but is not asserted.

---

### W11. MSM test only covers n=8, missing new code path coverage

**Commit:** `6c9ad68`
**File:** `src/provider/msm.rs`, `test_general_msm`

With n=8, the code takes the `bases.len() < 32` branch (c=3). The new window-size optimization (`c_base + 1` comparison) and signed-digit decomposition with diverse scalars at scale are **never tested**. Tests with n ≥ 32 (ideally n ≥ 1024) would exercise the new code paths.

---

### W12. Cost function uses old bucket count formula

**Commit:** `6c9ad68`
**File:** `src/provider/msm.rs`, line 56

```rust
((256 + c - 1) / c) * (bases.len() + (1 << c) - 1)
```

The signed-digit implementation uses `2^(c-1)` buckets, but the cost function still uses `2^c - 1`. This may select a slightly suboptimal window size but cannot cause incorrect results.

---

### W13. Hardcoded 256-bit scalar width in MSM

**Commit:** `6c9ad68`
**File:** `src/provider/msm.rs`, lines 56, 68, 103

The code assumes 256-bit scalars throughout. Correct for Pallas/Vesta but couples the MSM to a specific scalar field size. Pre-existing assumption.

---

### W14. `raw as i32` cast in signed-digit decomposition lacks guard

**Commit:** `6c9ad68`
**File:** `src/provider/msm.rs`, line 121

For c ≥ 31, `1i32 << c` would overflow. In practice c ≤ 24 for any realistic MSM size, but no debug assertion guards this.

---

## Per-Commit Reviews

### Commit `57caee3` — Add parallel re-encryption for RS shuffle (native + gadget witness)

**Files:** `src/rs_shuffle_bp/encryption.rs`, `examples/rs_shuffle_bp_full.rs`, `src/rs_shuffle_bp/mod.rs`

Introduces parallel ElGamal re-encryption using rayon for witness generation and a bellpepper gadget for in-circuit re-encryption. Adds a `WitnessCS` type for parallel witness synthesis and a full end-to-end example circuit combining permutation grand-product checks with re-encryption.

- Native Jacobian projective EC arithmetic is correct
- Parallel witness generation preserves ordering via `par_iter().enumerate()` + rayon's ordered `collect()`
- The `WitnessCS` approach (generate witness in mini constraint systems, copy aux vars back) is sound given deterministic allocation
- **Two critical issues** (C1, C2) related to circuit linkage, not the re-encryption itself

---

### Commit `547e1b4` — Reduce variable count below power-of-2 boundary to halve padded rest vars

**Files:** `src/gadgets/ecc.rs`, `src/rs_shuffle_bp/encryption.rs`, `src/rs_shuffle_bp/data_structures.rs`, `examples/rs_shuffle_bp_full.rs`

Seven micro-optimizations that reduce variable count from 258,871 to 258,040, crossing below 2¹⁸ = 262,144 to halve padding. Optimizations include:

1. Skip last unused `double_incomplete` in scalar mul loop
2. New `sub_incomplete` method (fold negation into LC, avoid separate variable)
3. Inline bit recomposition constraint (eliminate `le_bits_to_num` intermediate variable)
4. Share generator allocation across all 52 cards
5. Eliminate per-card `alloc_zero`
6. Use `AllocatedPointNonInfinity` for ciphertexts (remove `is_infinity` per point)
7. Use `AllocatedPointNonInfinity` for public key

All optimizations are algebraically equivalent to the originals. The inline bit recomposition saves 1 variable + 1 constraint per scalar decomposition. Migration from `AllocatedPoint` to `AllocatedPointNonInfinity` is safe because the original also never checked on-curve membership.

**Verdict:** Sound. The only concern is the completeness issue for scalar = 1 (W6), which has negligible probability.

---

### Commit `12628d6` — Optimize scalar mul with fixed-base and shared power tables (−41% constraints)

**Files:** `src/gadgets/ecc.rs`, `src/rs_shuffle_bp/encryption.rs`, `examples/rs_shuffle_bp_full.rs`

Two scalar multiplication optimizations:

1. **`scalar_mul_fixed_base`** (for r·G): Generator is a compile-time constant, so all 254 doublings are precomputed natively. In-circuit, only `add_constant` + `conditionally_select` per bit — no doubling constraints.

2. **`scalar_mul_with_powers`** (for r·PK): Public key PK is shared across all 52 cards. Power table (2^i · PK) is computed in-circuit once via `double_incomplete`, then shared across all cards.

Both use the same "assume bit[0]=1, then subtract slack" pattern as `scalar_mul_non_infinity`. Fixed-base power table values are baked into R1CS constraints as constants (prover cannot substitute). Shared `pk_powers` are read-only references in linear combinations (no cross-contamination between cards).

**Verdict:** Sound. The 41% constraint reduction is legitimate.

---

### Commit `5337265` — Add CSC + column remapping to field path for eval_sparse (31ms → 3ms)

**File:** `src/r1cs/mod.rs`

Extracts CSC construction and column remapping into shared helpers (`build_csc`, `build_column_remap_and_csc`). Changes the field-element `bind_row_vars_combined` from CSR row-major to CSC column-major iteration.

- CSR-to-CSC conversion is a standard two-pass stable-partition transpose — mathematically correct
- Column remapping is a bijection by construction
- Field arithmetic is associative/commutative so summation order doesn't affect results
- CSR fallback after deserialization (empty CSC fields via `#[serde(skip, default)]`) is properly guarded
- All 8 `col_presence` bitmask combinations are handled

**Verdict:** Sound. Only pre-existing issues noted (W9, W10).

---

### Commit `6c9ad68` — Optimize Pippenger MSM: pre-compute repr, optimal window, signed digits (−10% commit_rest)

**File:** `src/provider/msm.rs`

Three optimizations to `cpu_msm_serial`:

1. **Pre-compute `to_repr()`** once per scalar (avoid repeated Montgomery reductions)
2. **Optimal window size** by comparing cost at `c` and `c+1`
3. **Signed-digit decomposition** with carry propagation, halving bucket count from `2^c - 1` to `2^(c-1)`

The signed-digit decomposition is mathematically correct:
- Carry propagation handles all cases including max scalar and overflow window
- Carry can never propagate past the overflow window (raw ≤ 1 in overflow, always ≤ half)
- Bucket indices are always in bounds (verified for all digit ranges)
- Summation-by-parts correctly handles buckets with negative contributions (linearity)

**Verdict:** Sound. Main concern is insufficient test coverage of new code paths (W11).

---

## Adversarial Analysis

### Attack: Inconsistent permutation between Parts 1 and 2
**Target:** `57caee3`
**Result:** **Successful.** A malicious prover can use σ for the grand product and σ' for ciphertext allocation. No constraint links them. Complete break of shuffle correctness.

### Attack: r=0 identity re-encryption
**Target:** `57caee3`
**Result:** Cryptographically valid (r=0 is a valid re-encryption) but defeats the purpose of re-randomization. Protocol-level concern, not a circuit bug.

### Attack: Exploit reduced padding
**Target:** `547e1b4`
**Result:** **Not exploitable.** Padding variables are not referenced by any constraint. Committed in Hyrax but irrelevant to satisfiability.

### Attack: Exploit `AllocatedPointNonInfinity` (removed `is_infinity`)
**Target:** `547e1b4`
**Result:** **No new attack surface.** The original `AllocatedPoint` also never constrained points to be on the curve. Removal of `is_infinity` doesn't weaken any checks.

### Attack: Forge scalar mul by substituting power table values
**Target:** `12628d6`
**Result:** **Not exploitable.** Fixed-base values are baked into R1CS constraints. Shared `pk_powers` are constrained by `double_incomplete` chains. Prover cannot substitute.

### Attack: Cross-contamination via shared power table
**Target:** `12628d6`
**Result:** **Not exploitable.** Shared variables are read-only references in linear combinations. Each card creates its own lambda/x/y variables.

### Attack: CSC conversion loses or duplicates matrix entries
**Target:** `5337265`
**Result:** **Not exploitable.** Two-pass counting + allocation ensures every entry is written exactly once.

### Attack: Signed-digit decomposition produces wrong scalar
**Target:** `6c9ad68`
**Result:** **Not exploitable.** Verified for scalars 0, 1, 2, 3, max, and all-0xFF. Carry propagation is correct. Bucket indices always in bounds.

### Attack: Break commitment via incorrect MSM
**Target:** `6c9ad68`
**Result:** **Not exploitable.** Signed-digit decomposition is mathematically equivalent to standard decomposition. Group operation linearity ensures identical results.

---

## Suggestions

| # | Commit | Suggestion |
|---|--------|------------|
| S1 | `57caee3` | Add element-by-element witness comparison test (parallel vs serial) |
| S2 | `57caee3` | Make `native_scalar_mul` handle scalar = 0 or document precondition |
| S3 | `57caee3` | Replace `try_into().ok().unwrap()` with `try_into().expect("context")` |
| S4 | `57caee3` | Document the security model in the example circuit (public inputs, prover claims, verifier guarantees) |
| S5 | `547e1b4` | Add `#[cfg(debug_assertions)]` precondition check to `sub_incomplete` for `self.x != other.x` |
| S6 | `547e1b4` | Document scalar ≠ 0 precondition on `scalar_mul_non_infinity` |
| S7 | `12628d6` | Assert `powers.len() >= scalar_bits.len()` at function entry |
| S8 | `12628d6` | Document incomplete addition safety assumption with DL hardness reference |
| S9 | `5337265` | Fix pre-existing SmallCoeff single-entry fast path to handle non-unit coefficients |
| S10 | `5337265` | Add debug assertion in `build_csc` to verify `write_pos[c] == col_ptr[c+1]` |
| S11 | `5337265` | Group per-matrix CSC data into a sub-struct to reduce boilerplate |
| S12 | `6c9ad68` | Add MSM test with n ≥ 1024 to exercise window-size optimization and signed digits at scale |
| S13 | `6c9ad68` | Add `debug_assert_eq!(carry, 0)` after signed-digit loop |
| S14 | `6c9ad68` | Update cost function to use `2^(c-1)` bucket count instead of `2^c - 1` |
| S15 | `6c9ad68` | Add `debug_assert!(c <= 30)` to guard `i32` signed-digit arithmetic |
