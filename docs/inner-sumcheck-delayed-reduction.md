# Inner Sumcheck Delayed Reduction Optimizations

This document describes three optimization opportunities for the Spartan inner sumcheck using delayed modular reduction.

## Background

The inner sumcheck proves:
```
sum_y ABC(y) · z(y) = claim
```

Where:
- `ABC(y) = A(y) + r·B(y) + r²·C(y)` (sparse matrix evaluations bound to outer sumcheck point r_x)
- `z(y)` is the witness vector `[W | 1 | public_values | challenges]`
- The sumcheck runs for `log(num_vars)` rounds

Current implementation in `prove_inner_and_pcs` (spartan.rs):
1. Compute `poly_ABC[i] = evals_A[i] + r * evals_B[i] + r² * evals_C[i]` (one pass)
2. Call `prove_quad(poly_ABC, poly_z)` which runs log(n) rounds

---

## Optimization 1: Delayed Reduction in prepare_poly_ABC

**Location**: `src/spartan.rs` lines 328-333

**Current code**:
```rust
let poly_ABC = (0..evals_A.len())
  .into_par_iter()
  .map(|i| evals_A[i] + r * evals_B[i] + r * r * evals_C[i])
  .collect::<Vec<E::Scalar>>();
```

**Problem**: Each iteration does 2 multiplications with intermediate reductions.

**Optimized version**:
```rust
let r_sq = r * r;  // Precompute once
let poly_ABC = (0..evals_A.len())
  .into_par_iter()
  .map(|i| {
    // Use fused multiply-accumulate with delayed reduction
    let mut acc = <E::Scalar as DelayedReduction<E::Scalar>>::Accumulator::zero();
    // Start with evals_A[i] (convert to accumulator)
    let base = evals_A[i];
    E::Scalar::unreduced_multiply_accumulate(&mut acc, &r, &evals_B[i]);
    E::Scalar::unreduced_multiply_accumulate(&mut acc, &r_sq, &evals_C[i]);
    base + E::Scalar::reduce(&acc)
  })
  .collect::<Vec<E::Scalar>>();
```

**Savings**: ~n fewer reductions (one per element instead of two).

**Effort**: Easy

---

## Optimization 2: Eq-Split for Inner Sumcheck

**Concept**: Apply the same eq-split technique used in the outer sumcheck (`EqSumCheckInstance`) to the inner sumcheck.

**How outer sumcheck does it**:
- Split eq(τ, x) = eq_out(τ_out, x_out) × eq_in(τ_in, x_in)
- Maintain separate eq_out and eq_in polynomials
- For each outer index, accumulate inner sum in wide limbs, reduce once
- Then accumulate outer sum in wide limbs, reduce once
- Result: O(2^{n/2}) reductions instead of O(2^n) per round

**For inner sumcheck**:
The inner sumcheck computes `sum_y ABC(y) · z(y)`. We can introduce an implicit eq polynomial:
```
sum_y eq(r, y) · ABC(y) · z(y)  where eq starts as all 1s
```

Then split y = (y_out, y_in) and use the same two-phase accumulation:
```rust
struct InnerSumcheckInstance<E: Engine> {
  // Similar to EqSumCheckInstance but for quadratic ABC·z
  poly_eq_left: Vec<Vec<E::Scalar>>,   // eq evaluations for outer half
  poly_eq_right: Vec<E::Scalar>,        // eq evaluations for inner half
  first_half: usize,
  second_half: usize,
  // ... tracking state
}

impl InnerSumcheckInstance<E> {
  fn evaluation_points_quad_delayed(
    &self,
    round_idx: usize,
    evals_A: &[E::Scalar],
    evals_B: &[E::Scalar],
    evals_C: &[E::Scalar],
    poly_z: &MultilinearPolynomial<E::Scalar>,
    r: &E::Scalar,
    r_sq: &E::Scalar,
  ) -> (E::Scalar, E::Scalar) {
    // Two-phase accumulation like evaluation_points_cubic_with_three_inputs_delayed
    // Phase 1: Inner loop accumulates in wide limbs (no reduction)
    // Phase 2: Reduce once, multiply by outer eq, accumulate in wide limbs
    // Final: Single reduction
  }
}
```

**Savings**: O(2^n) → O(2^{n/2}) reductions per round

**Effort**: Medium (need to create InnerSumcheckInstance struct)

---

## Optimization 3: Fused ABC·z with Factored Form

**Key Insight**: Instead of computing `poly_ABC = A + r·B + r²·C` and then running prove_quad, keep A, B, C factored throughout ALL rounds.

**Mathematical basis**:
```
eval_0 = sum_i (A[i] + r·B[i] + r²·C[i]) · z[i]
       = sum_i A[i]·z[i] + r · sum_i B[i]·z[i] + r² · sum_i C[i]·z[i]
       = sum_Az + r·sum_Bz + r²·sum_Cz
```

By factoring out r and r², we can accumulate three separate sums in wide limbs and reduce only 3 times instead of n times!

**Full implementation**:
```rust
/// Proves inner sumcheck with factored A, B, C and delayed reduction.
///
/// Instead of materializing poly_ABC, keeps A, B, C separate and uses
/// the identity: (A + r·B + r²·C)·z = A·z + r·(B·z) + r²·(C·z)
fn prove_quad_factored_delayed<E: Engine>(
  claim: &E::Scalar,
  num_rounds: usize,
  evals_A: Vec<E::Scalar>,
  evals_B: Vec<E::Scalar>,
  evals_C: Vec<E::Scalar>,
  poly_z: Vec<E::Scalar>,
  r: &E::Scalar,
  r_sq: &E::Scalar,
  transcript: &mut E::TE,
) -> Result<(SumcheckProof<E>, Vec<E::Scalar>, Vec<E::Scalar>), SpartanError>
where
  E::Scalar: DelayedReduction<E::Scalar>,
{
  type Acc<S> = <S as DelayedReduction<S>>::Accumulator;

  let mut challenges = Vec::with_capacity(num_rounds);
  let mut polys = Vec::with_capacity(num_rounds);
  let mut claim_per_round = *claim;

  // Mutable factored polynomials
  let mut poly_A = evals_A;
  let mut poly_B = evals_B;
  let mut poly_C = evals_C;
  let mut poly_z = poly_z;

  for round in 0..num_rounds {
    let n = poly_A.len() / 2;

    // === Compute eval_0 and eval_2 with 6 reductions total ===
    let mut acc_Az_0 = Acc::<E::Scalar>::zero();
    let mut acc_Bz_0 = Acc::<E::Scalar>::zero();
    let mut acc_Cz_0 = Acc::<E::Scalar>::zero();
    let mut acc_Az_2 = Acc::<E::Scalar>::zero();
    let mut acc_Bz_2 = Acc::<E::Scalar>::zero();
    let mut acc_Cz_2 = Acc::<E::Scalar>::zero();

    for i in 0..n {
      let (a_lo, a_hi) = (poly_A[i], poly_A[i + n]);
      let (b_lo, b_hi) = (poly_B[i], poly_B[i + n]);
      let (c_lo, c_hi) = (poly_C[i], poly_C[i + n]);
      let (z_lo, z_hi) = (poly_z[i], poly_z[i + n]);

      // eval_0: accumulate A·z, B·z, C·z at point 0
      E::Scalar::unreduced_multiply_accumulate(&mut acc_Az_0, &a_lo, &z_lo);
      E::Scalar::unreduced_multiply_accumulate(&mut acc_Bz_0, &b_lo, &z_lo);
      E::Scalar::unreduced_multiply_accumulate(&mut acc_Cz_0, &c_lo, &z_lo);

      // eval_2: accumulate at point 2 (bound values)
      let a_bound = a_hi + a_hi - a_lo;
      let b_bound = b_hi + b_hi - b_lo;
      let c_bound = c_hi + c_hi - c_lo;
      let z_bound = z_hi + z_hi - z_lo;

      E::Scalar::unreduced_multiply_accumulate(&mut acc_Az_2, &a_bound, &z_bound);
      E::Scalar::unreduced_multiply_accumulate(&mut acc_Bz_2, &b_bound, &z_bound);
      E::Scalar::unreduced_multiply_accumulate(&mut acc_Cz_2, &c_bound, &z_bound);
    }

    // 6 reductions total (vs ~2n in current implementation)
    let sum_Az_0 = E::Scalar::reduce(&acc_Az_0);
    let sum_Bz_0 = E::Scalar::reduce(&acc_Bz_0);
    let sum_Cz_0 = E::Scalar::reduce(&acc_Cz_0);
    let eval_0 = sum_Az_0 + *r * sum_Bz_0 + *r_sq * sum_Cz_0;

    let sum_Az_2 = E::Scalar::reduce(&acc_Az_2);
    let sum_Bz_2 = E::Scalar::reduce(&acc_Bz_2);
    let sum_Cz_2 = E::Scalar::reduce(&acc_Cz_2);
    let eval_2 = sum_Az_2 + *r * sum_Bz_2 + *r_sq * sum_Cz_2;

    // Build univariate polynomial
    let evals = vec![eval_0, claim_per_round - eval_0, eval_2];
    let poly = UniPoly::from_evals(&evals)?;

    transcript.absorb(b"p", &poly);
    let challenge = transcript.squeeze(b"c")?;
    challenges.push(challenge);
    polys.push(poly.compress());

    claim_per_round = poly.evaluate(&challenge);

    // === Bind A, B, C, z separately (keeps factored form) ===
    // Can parallelize this with rayon
    for i in 0..n {
      poly_A[i] = poly_A[i] + challenge * (poly_A[i + n] - poly_A[i]);
      poly_B[i] = poly_B[i] + challenge * (poly_B[i + n] - poly_B[i]);
      poly_C[i] = poly_C[i] + challenge * (poly_C[i + n] - poly_C[i]);
      poly_z[i] = poly_z[i] + challenge * (poly_z[i + n] - poly_z[i]);
    }
    poly_A.truncate(n);
    poly_B.truncate(n);
    poly_C.truncate(n);
    poly_z.truncate(n);
  }

  // Final evaluations
  let final_ABC = poly_A[0] + *r * poly_B[0] + *r_sq * poly_C[0];

  Ok((
    SumcheckProof { compressed_polys: polys },
    challenges,
    vec![final_ABC, poly_z[0]],
  ))
}
```

**Reduction count comparison**:

| Approach | Reductions per round | Total for n=2^26 |
|----------|---------------------|------------------|
| Current prove_quad | ~2n | ~134M per round |
| Factored + delayed | 6 | 6 per round |

**Memory tradeoff**:
- Current: 1 vector (poly_ABC)
- Factored: 3 vectors (poly_A, poly_B, poly_C)

**Effort**: Medium

---

## Combining Optimizations 2 and 3

The eq-split (Optimization 2) and factored form (Optimization 3) can be combined:

1. Use eq-split to reduce the iteration structure from O(2^n) to O(2^{n/2}) inner × O(2^{n/2}) outer
2. Within each iteration, keep A, B, C factored and accumulate separate sums
3. Use delayed reduction throughout

This would give the best of both worlds:
- O(2^{n/2}) loop iterations (eq-split)
- 6 reductions per phase instead of O(2^{n/2}) (factored + delayed)

**Combined reduction count**: ~O(1) reductions per round regardless of n!

---

## Implementation Priority

1. **Optimization 3 (Factored + Delayed)**: Highest impact, medium effort
   - Reduces reductions from ~2n to 6 per round
   - Clear implementation path

2. **Optimization 2 (Eq-Split)**: High impact, medium effort
   - Can reuse patterns from EqSumCheckInstance
   - Reduces loop iterations

3. **Optimization 1 (prepare_poly_ABC)**: Low impact if doing Optimization 3
   - Only relevant if keeping current structure
   - Easy to implement

---

## Testing Strategy

1. Create equivalence tests comparing optimized vs current implementation
2. Benchmark with varying num_vars (16, 20, 24, 26)
3. Profile to verify reduction count decrease
4. Test with both Pasta and BN254 curves
