# Small-Value R1CS Pipeline: Remaining Optimizations

## Status

The end-to-end i32/i8 pipeline is working with two APIs:
- **`prove_int`**: monolithic (shape + witness + prove in one call)
- **`prove_small_value`**: prep/prove split via `setup_small` + `prep_prove_small` + `prove_small_value`

Current benchmark at 128B message (262K constraints):

```
                 synth_pre  commit_pre  r1cs_rest  commit_rest  mat_vec  outer_sc  eval_rx  eval_sparse  inner_sc  pcs   prep   prove   total
prep_int              7          3          0          0          0        5         2           6          3       10    10     30     36
int                   7          3          0          0          0        4         1          10          3       10    23     44     38
large                38          3          1          0          2       11         1           7          4       10    41     40     77
prep_int speedup   5.4×      1.0×          -          -          -     2.2×      0.5×        1.2×       1.3×    1.0×  4.1×   1.3×   2.1×
```

**Overall speedup: ~2.1×.** The prep/prove split (`prep_int`) moves shape extraction to
a one-time `setup_small` cost (25ms), giving 4.1× faster prep.

Goal: ≥2.5× (ideally 3×+).

---

## Remaining i8 → Field Conversions in `prove_small_value`

Three places in `prove_small_value` (spartan.rs) still convert i8 to field elements.
Only one is a bulk conversion; the other two are small or reuse already-converted data.

### Conversion #1: pub_i8 → pub_field for transcript (small, unavoidable)

```rust
let pub_field: Vec<E::Scalar> = pub_i8.iter()
    .map(|&v| if v == 0 { E::Scalar::ZERO } else { E::Scalar::ONE })
    .collect();
transcript.absorb(b"public_values", &pub_field.as_slice());
```

Required for transcript compatibility with the verifier, which absorbs field elements.
Only `num_public` elements (32 for SHA-256 hash output). Low priority to optimize.

### Conversion #2: W_i8 → W_field for PCS prove (bulk, unavoidable with current PCS)

```rust
let W_field: Vec<E::Scalar> = prep.W.iter()
    .map(|&v| if v == 0 { E::Scalar::ZERO } else { E::Scalar::ONE })
    .collect();
```

**This is the only bulk conversion** — converts the entire witness (131K elements at 128B)
to field elements for `E::PCS::prove()`. The Hyrax evaluation argument
(`hyrax_prove_bind` + `hyrax_prove_commit` + `hyrax_prove_ipa`) needs the polynomial as
field elements because it does L·Z matrix-vector multiply and IPA with field scalars.

The commitment itself is already optimized with `commit_i8` (subset-sum of generators).
This conversion is only for the *evaluation proof*, which is separate from the commitment.

**Fix:** Add a `PCS::prove_binary` that accepts `&[i8]` and does the IPA inner products
with conditional addition instead of scalar multiplication. Same idea as `commit_i8` but
for the evaluation argument path.

### Conversion #3: U_field construction for eval_X (small, reuses pub_field)

```rust
let U_field = SplitR1CSInstance::<E> { ..., public_values: pub_field, ... };
let U_regular = U_field.to_regular_instance()?;
```

Reuses the already-converted `pub_field` from conversion #1. No additional allocation.

### PCS commits (already optimized)

The three witness portion commits (shared, precommitted, rest) use `PCS::commit_i8`
which does subset-sum of generators directly from `&[i8]` — no field conversion needed.
This was the previous conversion #1-3 bottleneck, now eliminated.

---

## eval_sparse: field × i32 optimization

`bind_row_vars_combined_int` iterates all matrix nonzeros doing `field × i32`.
The fast path handles ±1 (conditional add/sub) but still has branch overhead per entry.

**Potential optimization:** Since SHA-256 coefficients are mostly {-1, 0, 1, 2, -2},
batch-sort entries by coefficient value and do vectorized accumulation per group:
- coeff=1 entries: `buffer[col] += rx_row` (no multiply)
- coeff=-1 entries: `buffer[col] -= rx_row`
- coeff=2 entries: `buffer[col] += rx_row_doubled`
- rare general entries: `buffer[col] += rx_row * from_i32(coeff)`

This eliminates the per-entry match/branch.

---

## PCS: commit_bits for binary witnesses

The Hyrax PCS commit does a multi-scalar multiplication (MSM): `∑ scalar_i × G_i`.
When all scalars are 0 or 1, this reduces to a subset-sum of generators: `∑_{i: bit=1} G_i`.
No scalar multiplication needed — just point addition.

**Expected speedup:** MSM is the dominant PCS cost. Replacing scalar-multiply with
conditional point-addition should be ~4-8× faster for the commit step.

---

## Shape extraction caching (DONE)

`setup_small` now caches the `SplitR1CSShape<E, i32>` in `SpartanProverKey<E, i32>`,
computed once during setup. This moves shape extraction from per-proof to per-circuit cost.

At 128B: `setup_small` = 25ms (one-time), eliminates 22ms per proof from `prove_int`'s
shape re-extraction. The prep phase drops from 23ms → 10ms (4.1× faster).

---

## Parallel witness synthesis

Currently `SmallSatisfyingAssignment<i8>` runs the SHA-256 circuit sequentially on a single
thread. At 1024B, `synth_pre` takes 49ms — still the largest single phase in prove_int.

The SHA-256 circuit has natural parallelism: each message block's compression rounds are
independent until the final state merge. Within a block, the 64 rounds are sequential, but
bit-decomposition and `addmany` limbed addition across different words can run in parallel.

**Approach 1: Block-level parallelism.** For multi-block messages (≥2 blocks = ≥128B),
synthesize each block's witness independently on separate threads, then merge. Each block
produces ~4K variables. The shared state (chaining values) creates a dependency between
blocks, but the witness values for internal variables within a block are independent.

**Approach 2: Gadget-level parallelism.** Within a single SHA-256 round, the CH/MAJ/Σ
computations and their bit-level XOR/AND operations are independent. A parallel
`SmallSatisfyingAssignment` could batch-allocate variables and fill them concurrently.
This requires careful variable index management but no locking (each gadget writes to
disjoint index ranges known at shape-extraction time).

**Approach 3: Separate shape from witness.** Since the shape (variable count, constraint
structure) is known from `SmallShapeCS`, pre-allocate the witness vector and fill regions
in parallel. The SHA-256 circuit's variable layout is deterministic — region boundaries
can be computed from the message length without running the circuit.

**Expected speedup:** 2-4× on `synth_pre` with block-level parallelism (limited by
sequential dependency between blocks). Gadget-level could go further but is more complex.

---

## Priority Order

1. ~~**Shape caching** (DONE): `setup_small` caches i32 shape in prover key~~
2. **PCS prove_binary** (high impact, moderate): `PCS::prove_binary(&[i8])` for evaluation argument — eliminates the only bulk i8→field conversion remaining
3. **Parallel witness synthesis** (high impact, moderate): multi-threaded i8 witness gen
4. **eval_sparse batching** (moderate impact, moderate): coefficient-grouped accumulation
5. **Transcript absorb_bits** (low impact, easy): absorb i8 public values directly
