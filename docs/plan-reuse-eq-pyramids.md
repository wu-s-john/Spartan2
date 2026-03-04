# Plan: Reuse Eq Pyramids Between Accumulator and EqSumCheckInstance

## Goal
Eliminate redundant eq polynomial computation by building pyramids once in the accumulator builder and reusing them in EqSumCheckInstance.

## Math Summary

**Factorization:**
```
eq(τ[l₀:ℓ], x) = eq(τ[l₀], x₀) × eq(τ[l₀+1:l₀+in_vars], x_in) × eq(τ[l₀+in_vars:ℓ], x_out)
                 └─ eval_eq_left ─┘   └── e_in_pyramid (popped) ──┘   └── e_xout_pyramid ──┘
```

**Split convention (accumulator):**
- `in_vars = ⌈suffix_vars/2⌉`
- `xout_vars = ⌊suffix_vars/2⌋`

**Pyramid structure:**
- `e_in_pyramid`: built from τ[l₀ : l₀+in_vars], has in_vars+1 layers
  - Top layer (size 2^in_vars) used by accumulator
  - Pop top layer → in_vars layers for EqSumCheckInstance
- `e_xout_pyramid`: built from τ[l₀+in_vars : ℓ], has xout_vars+1 layers

**EqSumCheckInstance mapping:**
- `first_half = in_vars` (rounds 1..in_vars process e_in variables)
- `second_half = xout_vars` (rounds in_vars+1..suffix_vars process e_xout variables)
- `eval_eq_left`: initialized with prefix eq factor, accumulates τ[l₀] in round 1

## Files to Modify

1. **`src/lagrange_accumulator/accumulator_builder.rs`**
   - Modify `build_accumulators_spartan` to build and return full pyramids
   - Return type: `(LagrangeAccumulators<F, 2>, Vec<Vec<F>>, Vec<Vec<F>>)` for (accumulators, e_in_pyramid, e_xout_pyramid)

2. **`src/sumcheck.rs`** (eq_sumcheck module)
   - Add `EqSumCheckInstance::from_pyramids()` constructor
   - Parameters: `e_in_pyramid`, `e_xout_pyramid`, `taus`, `eval_eq_left`

3. **`src/small_sumcheck.rs`**
   - Update to receive pyramids from accumulator builder
   - Pop e_in_pyramid's top layer before passing to EqSumCheckInstance
   - Wire up the new constructor

## Implementation Steps

### Step 1: Modify accumulator_builder.rs

```rust
pub fn build_accumulators_spartan<F, SV>(
  az: &MultilinearPolynomial<SV>,
  bz: &MultilinearPolynomial<SV>,
  taus: &[F],
  l0: usize,
) -> (LagrangeAccumulators<F, 2>, Vec<Vec<F>>, Vec<Vec<F>>)
```

- Change `precompute_eq_tables` to build full pyramids (not just top layers)
- Accumulator uses top layer internally: `e_in_pyramid.last()`, `e_xout_pyramid.last()`
- Return full pyramids: `(accumulators, e_in_pyramid, e_xout_pyramid)`

### Step 2: Add from_pyramids constructor

```rust
impl<E: Engine> EqSumCheckInstance<E> {
    /// Creates EqSumCheckInstance from precomputed eq pyramids.
    ///
    /// # Arguments
    /// - `e_in_pyramid`: Pyramid for inner variables, ALREADY POPPED (in_vars layers)
    /// - `e_xout_pyramid`: Full pyramid for outer variables (xout_vars+1 layers)
    /// - `taus`: Suffix taus τ[l₀:ℓ]
    /// - `eval_eq_left`: Accumulated eq factor from prefix rounds
    pub fn from_pyramids(
        e_in_pyramid: Vec<Vec<E::Scalar>>,   // popped by caller, in_vars layers
        e_xout_pyramid: Vec<Vec<E::Scalar>>, // full, xout_vars+1 layers
        taus: &[E::Scalar],                  // suffix taus τ[l₀:ℓ]
        eval_eq_left: E::Scalar,             // prefix eq factor
    ) -> Self {
        let in_vars = e_in_pyramid.len();        // in_vars layers after pop
        let xout_vars = e_xout_pyramid.len() - 1; // xout_vars+1 layers

        // Compute eq_tau_0_2_3 for all suffix taus (same logic as new())
        let eq_tau_0_2_3 = taus
            .par_iter()
            .map(|tau| {
                let tau2 = tau.double();
                let tau3 = tau2 + tau;
                let tau5 = tau3 + tau2;
                (E::Scalar::ONE - tau, tau3 - E::Scalar::ONE, tau5 - E::Scalar::ONE.double())
            })
            .collect();

        Self {
            init_num_vars: in_vars + xout_vars,
            first_half: in_vars,
            second_half: xout_vars,
            round: 1,
            taus: taus.to_vec(),
            eval_eq_left,
            poly_eq_left: e_in_pyramid,
            poly_eq_right: e_xout_pyramid,
            eq_tau_0_2_3,
        }
    }
}
```

### Step 3: Update small_sumcheck.rs

```rust
// Build accumulators and get full pyramids
let (accumulators, mut e_in_pyramid, e_xout_pyramid) =
    build_accumulators_spartan(poly_A_small, poly_B_small, &taus, l0);

let mut small_value_sumcheck =
    SmallValueSumCheck::from_accumulators(accumulators);

// ... run small-value rounds ...

// Pop top layer from e_in_pyramid (caller's responsibility)
// Top layer was used by accumulator, EqSumCheckInstance needs the rest
e_in_pyramid.pop();

// Create EqSumCheckInstance with precomputed pyramids
let eq_instance = EqSumCheckInstance::from_pyramids(
    e_in_pyramid,
    e_xout_pyramid,
    &taus[l0..],
    small_value_sumcheck.eq_alpha(),
);
```

## Verification

1. Run existing tests: `cargo test -- --skip test_msm_ux`
2. Verify sumcheck proofs still pass
3. Check that eq evaluations match between old and new implementations
