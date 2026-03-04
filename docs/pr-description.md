> **Note:** This PR builds on the Barrett reduction infrastructure from #111. The changes span `src/big_num/`, `src/lagrange_accumulator/`, and `src/small_sumcheck.rs`. Reviewers are encouraged to focus on `src/lagrange_accumulator/` and `src/small_sumcheck.rs` — the `src/big_num/` changes are covered by #111.

## Summary

Implement the small-value sumcheck optimization (Algorithm 6) from ["Speeding Up Sum-Check Proving"](https://eprint.iacr.org/2025/1117) (Bagad, Dao, Domb, Thaler, IACR 2025/1117). This replaces expensive field multiplications with native integer arithmetic during the first ℓ₀ rounds when polynomial values are guaranteed small.

This PR provides the core infrastructure for the optimization. Integration into the Spartan prover is planned for a follow-up PR.

## Benchmarks

Measured on M1 Max MacBook Pro with `jemalloc`, BN254 scalar field, ℓ₀ = 3.

| num_vars | n | baseline (DMR) | small-value | speedup |
|----------|---|----------------|-------------|---------|
| 16 | 65,536 | 4.98 ms | 3.43 ms | **1.45×** |
| 17 | 131,072 | 8.83 ms | 4.79 ms | **1.84×** |
| 18 | 262,144 | 15.07 ms | 7.54 ms | **2.00×** |
| 19 | 524,288 | 27.81 ms | 12.84 ms | **2.17×** |
| 20 | 1,048,576 | 49.96 ms | 26.37 ms | **1.90×** |
| 21 | 2,097,152 | 190.66 ms* | 50.73 ms | **3.76×*** |
| 22 | 4,194,304 | 152.75 ms | 96.09 ms | **1.59×** |
| 23 | 8,388,608 | 285.86 ms | 181.02 ms | **1.58×** |
| 24 | 16,777,216 | 546.06 ms | 333.13 ms | **1.64×** |
| 25 | 33,554,432 | 1,095.2 ms | 653.01 ms | **1.68×** |
| 26 | 67,108,864 | 2,150.1 ms | 1,314.5 ms | **1.64×** |

These results are consistent with the **~1.6-2× speedup** and has achieved better overall performance than #98.

## Key Components

**Lagrange accumulator infrastructure** (`src/lagrange_accumulator/`):
- `LagrangeAccumulators`: precomputed A_i(v, u) values for all rounds
- `LagrangeIndex/LagrangePoint/LagrangeHatPoint`: type-safe domain indices
- `LagrangeBasisFactory`: barycentric Lagrange basis with O(D) evaluation
- `extend_to_lagrange_domain`: batch extension from {0,1}^ℓ₀ to U_D^ℓ₀
- `EqRoundFactor`: tracks α = eq(τ_{<i}, r_{<i}) across rounds

**Performance optimizations**:
- Thread-local `SpartanThreadState` eliminates per-iteration allocations
- Delayed modular reduction via `SignedWideLimbs` accumulators
- Skip binary betas (Az·Bz = Cz on {0,1}^n for satisfying witnesses)
- Batched eq-weighted binding in transition phase

**API**:
- `SmallValue` trait: `WideMul + Copy + Zero + Add + Sub + Send + Sync`
- `SmallValueEngine<SV>`: blanket impl consolidates field requirements
- `prove_cubic_small_value<E, SV, const LB>`: main entry point

## Test Plan

- [x] `prove_cubic_small_value` produces identical proofs to `prove_cubic_with_three_inputs` (equivalence tests)
- [x] `cargo test -- --skip test_msm_ux`
- [x] `cargo clippy`
