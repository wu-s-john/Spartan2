# Small-Value R1CS Pipeline: Remaining Optimizations

## Status

The end-to-end i32/i8 pipeline (`prove_int`) is working and produces valid proofs.
Current benchmark at 128B message (262K constraints):

```
                 synth_pre  commit_pre  r1cs_rest  commit_rest  mat_vec  outer_sc  eval_rx  eval_sparse  inner_sc  pcs   prep   prove   total
int                    6         3         0          1          0        5         0         9           5        10     24      47      39
large                 35         3         1          0          2       11         1         8           6        11     38      43      78
speedup             5.8×      1.0×         -        0.0×          -     2.2×         -      0.9×        1.2×     1.1×   1.6×    0.9×    2.0×
```

At 1024B message (1M constraints):

```
                 synth_pre  commit_pre  r1cs_rest  commit_rest  mat_vec  outer_sc  eval_rx  eval_sparse  inner_sc  pcs   prep   prove   total
int                   49        25         0         13          2       16         0        59           26       14    166     234     204
large                234        18        12          9         16       38         6        50           26       14    258     182     432
speedup             4.8×      0.7×         -        0.7×       8.0×    2.4×         -      0.8×         1.0×     1.0×   1.6×    0.8×    2.1×
```

**Overall speedup: ~2×.** Goal: ≥2.5× (ideally 3×+).

---

## Redundant i8 → Field Conversions

The witness vector is currently converted from `Vec<i8>` to `Vec<E::Scalar>` multiple times.
Each conversion allocates a new `Vec<E::Scalar>` (32 bytes per element vs 1 byte for i8).
At 1M constraints, that's ~32MB per conversion.

### Conversion #1-3: For PCS commits (bellpepper/r1cs.rs)

In `small_r1cs_instance_and_witness`, three `to_field()` calls convert witness slices for commits:

```rust
// Line ~374: shared portion
let field = to_field(&W_i8[..S.num_shared]);
let comm = PCS::<E>::commit(ck, &field, &r, true)?;

// Line ~385: precommitted portion
let field = to_field(&W_i8[S.num_shared..S.num_shared + S.num_precommitted]);
let comm = PCS::<E>::commit(ck, &field, &r, true)?;

// Line ~396: rest portion
let field_rest = to_field(&W_i8[S.num_shared + S.num_precommitted..]);
let comm_W_rest = PCS::<E>::commit(ck, &field_rest, &r_W_rest, true)?;
```

**Fix:** Add `PCS::commit_bits` that accepts `&[i8]` directly.
Since witnesses are 0/1, the MSM can skip the scalar multiplication for 0-entries and use
the generator directly for 1-entries: `if bit { acc += generator[i]; }`.
This avoids both the conversion AND the expensive scalar-multiply in the MSM.

### Conversion #4: Public values for transcript (spartan.rs)

```rust
// Line ~690
let pub_field: Vec<E::Scalar> = pub_i8.iter()
    .map(|&v| if v == 0 { E::Scalar::ZERO } else { E::Scalar::ONE })
    .collect();
transcript.absorb(b"public_values", &pub_field.as_slice());
```

**Fix:** Add `transcript.absorb_bits` or absorb the i8 bytes directly.
Public values are tiny (32 bytes for SHA-256 hash output), so this is low priority.

### Conversion #5: Entire witness for inner sumcheck + PCS (spartan.rs)

```rust
// Lines ~742-767: convert FULL witness Vec<i8> → Vec<E::Scalar>
let W_field: Vec<E::Scalar> = W_i8.W.iter()
    .map(|&v| if v == 0 { E::Scalar::ZERO } else { E::Scalar::ONE })
    .collect();

// Then CLONE it into z_field (another full allocation)
let z_field = [W_field_witness.W.clone(), vec![E::Scalar::ONE], ...].concat();
```

This is the most expensive conversion — two full-size `Vec<E::Scalar>` allocations.

**Fix:** Build `z_field` once, in-place, without the intermediate `W_field`:
```rust
let mut z_field = vec![E::Scalar::ZERO; num_vars * 2];
for (i, &v) in W_i8.W.iter().enumerate() {
    if v != 0 { z_field[i] = E::Scalar::ONE; }
}
z_field[num_vars] = E::Scalar::ONE; // the "1" separator
// copy public values and challenges into their positions
```

If conversion #1-3 is fixed with `commit_bits`, the `W_field` for PCS prove can also
use the in-place `z_field` slice instead of a separate allocation.

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

## Shape extraction caching

`small_r1cs_shape` (24ms at 128B, 166ms at 1024B) re-runs the full SHA-256 circuit
through `SmallShapeCS` every time. The shape only depends on the circuit structure
(message length), not the witness values.

**Fix:** Cache the `SplitR1CSShape<E, i32>` in the prover key or compute it once during setup.
This moves shape extraction from per-proof to per-circuit cost.

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

1. **Conversion #5** (high impact, easy): eliminate redundant allocation in prove_int
2. **Conversion #1-3** (high impact, moderate): `commit_bits` for binary witnesses
3. **Shape caching** (moderate impact, easy): compute i32 shape once during setup
4. **Parallel witness synthesis** (high impact, moderate): multi-threaded i8 witness gen
5. **eval_sparse batching** (moderate impact, moderate): coefficient-grouped accumulation
6. **Conversion #4** (low impact, easy): absorb bits directly in transcript
