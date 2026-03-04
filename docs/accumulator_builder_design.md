# Accumulator Builder Design

## Two Eq Tables

The accumulator builder uses only **two precomputed eq tables**:

1. **`forward_pyramid[i][y]`** = eq(τ[i..l/2], y) for y ∈ {0,1}^{l/2-i}, where i ∈ 0..l0
2. **`e_xout[x_out]`** = eq(τ[l/2..l], x_out) for x_out ∈ {0,1}^{l/2}

## Variable Layout

For witness polynomials with l variables:
- **Prefix**: l0 bits (positions 0..l0) — extended from {0,1}^{l0} to U_D^{l0}
- **Middle**: l/2 - l0 bits (positions l0..l/2) — part of forward_pyramid's y index
- **x_out**: l/2 bits (positions l/2..l) — indexed by e_xout

Total suffix bits: (l/2 - l0) + l/2 = l - l0

## Mathematical Formula

For accumulator A_i(v, u):

```
A_i(v, u) = Σ_{middle} Σ_{x_out} forward_pyramid[i][(y, middle)] × e_xout[x_out] × [Az_ext(β) · Bz_ext(β)]
```

where β = (v, u, y) with:
- v ∈ U_D^i (prefix of β)
- u ∈ Û_D (coordinate at position i)
- y ∈ {0,1}^{l0-i} (binary suffix within the l0 prefix)

## Loop Structure

- **Outer parallel loop**: x_out ∈ 0..2^{l/2}
- **Inner loop**: middle ∈ 0..2^{l/2 - l0}
- **Total iterations**: 2^{l/2} × 2^{l/2-l0} = 2^{l - l0}

For each (x_out, middle) combination:
1. Gather 2^{l0} prefix values
2. Extend prefix from {0,1}^{l0} → U_D^{l0}
3. Compute products and scatter

## Code

```rust
pub fn build_accumulators_spartan<F, SV>(
  az: &MultilinearPolynomial<SV>,
  bz: &MultilinearPolynomial<SV>,
  l0: usize,
  forward_pyramid: &[Vec<F>],  // forward_pyramid[i] = eq(τ[i..l/2], ·), size 2^{l/2-i}
  e_xout: &[F],                // eq(τ[l/2..l], ·), size 2^{l/2}
) -> LagrangeAccumulators<F, 2>
where
  F: SmallValueEngine<SV>,
  SV: SmallValue,
{
  let base: usize = 3;
  let l = az.Z.len().trailing_zeros() as usize;
  let half_l = l / 2;
  let prefix_size = 1usize << l0;
  let middle_bits = half_l - l0;
  let middle_size = 1usize << middle_bits;  // 2^{l/2 - l0}
  let num_x_out = e_xout.len();             // 2^{l/2}
  let suffix_vars = l - l0;
  let ext_size = base.pow(l0 as u32);

  let BetaContributions { contributions: beta_contributions, num_betas } =
    compute_beta_contributions::<2>(l0);

  let betas_with_infty: Vec<usize> = (0..num_betas)
    .filter(|&i| (0..l0).any(|d| (i / base.pow(d as u32)) % base == 0))
    .collect();

  type State<F2, SV2> = SpartanThreadState<F2, SV2, 2>;

  let fold_results: Vec<State<F, SV>> = (0..num_x_out)
    .into_par_iter()
    .fold(
      || State::<F, SV>::new(l0, num_betas, prefix_size, ext_size),
      |mut state: State<F, SV>, x_out_bits| {

        // Inner loop over middle ∈ 0..2^{l/2 - l0}
        for middle in 0..middle_size {
          // Suffix layout: (middle << half_l) | x_out
          let suffix = (middle << half_l) | x_out_bits;

          // Gather prefix values
          for prefix in 0..prefix_size {
            let idx = (prefix << suffix_vars) | suffix;
            state.az.boolean_evals[prefix] = az.Z[idx];
            state.bz.boolean_evals[prefix] = bz.Z[idx];
          }

          // Extend l0 prefix from {0,1}^{l0} → U_D^{l0}
          let az_size = extend_to_lagrange_domain::<SV, 2>(
            &state.az.boolean_evals,
            &mut state.az.extended_evals,
            &mut state.az.extended_scratch,
          );
          let az_ext = &state.az.extended_evals[..az_size];

          let bz_size = extend_to_lagrange_domain::<SV, 2>(
            &state.bz.boolean_evals,
            &mut state.bz.extended_evals,
            &mut state.bz.extended_scratch,
          );
          let bz_ext = &state.bz.extended_evals[..bz_size];

          // Compute products and scatter directly (need middle for forward_pyramid index)
          for &beta_idx in &betas_with_infty {
            let prod = SV::wide_mul(az_ext[beta_idx], bz_ext[beta_idx]);

            // Phase 1: e_xout × product → reduce to field
            state.partial_sums[beta_idx] =
              <F as DelayedReduction<SV::Product>>::Accumulator::zero();
            <F as DelayedReduction<SV::Product>>::unreduced_multiply_accumulate(
              &mut state.partial_sums[beta_idx],
              &e_xout[x_out_bits],
              &prod,
            );
            let val = <F as DelayedReduction<SV::Product>>::reduce(&state.partial_sums[beta_idx]);

            if val == F::ZERO { continue; }

            // Phase 2: Scatter with forward_pyramid weighting
            for pref in &beta_contributions[beta_idx] {
              let round = pref.round_0 as usize;
              // Combine y_from_beta with middle to get full forward_pyramid index
              // forward_pyramid[i] covers τ[i..l/2], which is (y_from_beta || middle)
              let fp_y = ((pref.y_idx as usize) << middle_bits) | middle;

              <F as DelayedReduction<F>>::unreduced_multiply_accumulate(
                &mut state.acc.rounds[round].data_mut()[pref.v_idx as usize][pref.u_idx as usize],
                &forward_pyramid[round][fp_y],
                &val,
              );
            }
          }
        }

        state
      },
    )
    .collect();

  // Merge thread-local accumulators
  let merged = fold_results
    .into_iter()
    .reduce(|mut a, b| {
      a.acc.merge(&b.acc);
      a
    })
    .expect("num_x_out > 0");

  // Final reduction: wide limbs → field elements
  let mut result: LagrangeAccumulators<F, 2> = LagrangeAccumulators::new(l0);
  for (round_idx, round) in merged.acc.rounds.iter().enumerate() {
    for (v_idx, row) in round.data().iter().enumerate() {
      for (u_idx, wide_elem) in row.iter().enumerate() {
        if !wide_elem.is_zero() {
          result.rounds[round_idx].data_mut()[v_idx][u_idx] =
            <F as DelayedReduction<F>>::reduce(wide_elem);
        }
      }
    }
  }

  result
}
```

## Key Design Decisions

1. **Scatter inside middle loop**: We need `middle` for the `forward_pyramid` index, so we can't accumulate over middle first.

2. **Delayed reduction**:
   - Phase 1: `e_xout × product` reduced to field element
   - Phase 2: `forward_pyramid × val` accumulated into wide limbs in thread-local accumulators

3. **Combined forward_pyramid index**: `fp_y = (y_from_beta << middle_bits) | middle`
   - `y_from_beta`: l0 - i bits (from idx4 decomposition)
   - `middle`: l/2 - l0 bits

4. **Thread-local accumulators**: Each parallel task has its own `LagrangeAccumulators<WideAcc, 2>`, merged at the end.
