# Refactor Plan: SmallPrepSNARK Alignment + PrecommittedState Flattening

Two separate PRs. PR 1 is low-risk and self-contained. PR 2 touches NeutronNova and SpartanZK and should be done carefully.

---

## PR 1 — Align SmallPrepSNARK with the OG SpartanWitness pattern

### Goal

Move `shared_witness` / `precommitted_witness` logic off `SpartanSNARK` and onto
`SmallSatisfyingAssignment`, mirroring how `SatisfyingAssignment` implements `SpartanWitness`.

### Current state

```
SpartanWitness<E> trait  ←  SatisfyingAssignment implements this
  type PrecommittedState
  fn shared_witness(S, ck, circuit, is_small) -> PrecommittedState
  fn precommitted_witness(&mut ps, S, ck, circuit, is_small) -> ()
  fn r1cs_instance_and_witness(&mut ps, S, ck, circuit, is_small, transcript) -> (...)

SmallSatisfyingAssignment<V>  ←  only implements SmallConstraintSystem, no witness logic

SpartanSNARK<E>
  fn prep_prove_small_shared(pk, circuit) -> SmallPrepSNARK   ← wrong home
  fn prep_prove_small_precommitted(pk, circuit, &mut prep)    ← wrong home
  fn prep_prove_small(pk, circuit) -> SmallPrepSNARK          ← thin wrapper
```

### Proposed state

```
SmallSpartanWitness<E, W> trait  ←  NEW, SmallSatisfyingAssignment<W> implements this
  type SmallPrepState
  fn shared_witness(S, ck, circuit) -> SmallPrepState
  fn precommitted_witness(&mut prep, S, ck, circuit) -> ()

SmallSatisfyingAssignment<W>
  implements SmallConstraintSystem<W>   (unchanged)
  implements SmallSpartanWitness<E, W>  (new)

SpartanSNARK<E>
  fn prep_prove_small(pk, circuit) -> SmallPrepSNARK   ← thin wrapper, calls both trait methods
  fn prove_small_value(pk, circuit, prep)              ← unchanged
```

### Changes required

#### 1. New trait `SmallSpartanWitness` — `bellpepper/r1cs.rs`

Define alongside `SpartanWitness`:

```rust
pub trait SmallSpartanWitness<E: Engine, W> {
  type SmallPrepState;

  fn shared_witness<C: SmallSpartanCircuit<E, W>>(
    S: &SplitR1CSShape<E>,
    ck: &CommitmentKey<E>,
    circuit: &C,
  ) -> Result<Self::SmallPrepState, SpartanError>;

  fn precommitted_witness<C: SmallSpartanCircuit<E, W>>(
    prep: &mut Self::SmallPrepState,
    S: &SplitR1CSShape<E>,
    ck: &CommitmentKey<E>,
    circuit: &C,
  ) -> Result<(), SpartanError>;
}
```

Note: no `is_small: bool` parameter — the small-value path always uses `commit_witness`
(bool-converted MSM), so there is no flag needed.

#### 2. Implement `SmallSpartanWitness` for `SmallSatisfyingAssignment<W>` — `small_constraint_system/mod.rs` or `bellpepper/r1cs.rs`

The impl body is a direct lift of the logic currently in `prep_prove_small_shared` and
`prep_prove_small_precommitted` on `SpartanSNARK`.

New imports needed in the impl's file:
- `crate::{Blind, Commitment, CommitmentKey, PCS}`
- `crate::r1cs::SplitR1CSShape`
- `crate::bellpepper::r1cs::{SmallPrepSNARK, WitnessCommitment}`
- `crate::traits::Engine`
- `crate::errors::SpartanError`
- `crate::start_span!`

Decision: the impl could live in `bellpepper/r1cs.rs` (keeps Engine-aware code out of
`small_constraint_system/`) or in `small_constraint_system/mod.rs` (co-located with the type).
Recommend `bellpepper/r1cs.rs` to keep `small_constraint_system` purely algebraic.

#### 3. Update `SpartanSNARK::prep_prove_small` — `spartan.rs`

Replace the three current functions (`prep_prove_small_shared`,
`prep_prove_small_precommitted`, `prep_prove_small`) with a single thin wrapper:

```rust
pub fn prep_prove_small<C, Coeff, W>(
  pk: &SpartanProverKey<E, Coeff>,
  circuit: &C,
) -> Result<SmallPrepSNARK<E, W>, SpartanError>
{
  let mut prep = SmallSatisfyingAssignment::shared_witness(&pk.S, &pk.ck, circuit)?;
  SmallSatisfyingAssignment::precommitted_witness(&mut prep, &pk.S, &pk.ck, circuit)?;
  Ok(prep)
}
```

#### 4. Also apply `WitnessCommitment` to `PrecommittedState` — `bellpepper/r1cs.rs`

While touching this file, replace the 4-option fields in `PrecommittedState` with
2 `Option<WitnessCommitment<E>>` fields, consistent with `SmallPrepSNARK`. This also
simplifies `rerandomize` and `rerandomize_with_shared`.

**Field accesses that change:**
- `ps.comm_W_shared` / `ps.r_W_shared` → `ps.comm_shared.as_ref().map(|wc| &wc.comm)` etc.
- `ps_core.comm_W_shared` / `ps_core.r_W_shared` (neutronnova_zk.rs lines 1776-1777)

### Files touched

| File | Change |
|---|---|
| `bellpepper/r1cs.rs` | Add `SmallSpartanWitness` trait + impl; apply `WitnessCommitment` to `PrecommittedState` |
| `small_constraint_system/mod.rs` | No change (keeps purely algebraic) |
| `spartan.rs` | Replace 3 prep functions with 1 thin wrapper |
| `neutronnova_zk.rs` | Update `comm_W_shared` / `r_W_shared` field accesses if `WitnessCommitment` applied |

---

## PR 2 — Flatten PrecommittedState into SpartanPrepSNARK

### Goal

Remove the `SpartanPrepSNARK { ps: PrecommittedState<E> }` wrapper. `SpartanPrepSNARK`
becomes the state struct directly, eliminating the indirection `prep_snark.ps.*`.

Same applies to `SpartanPrepZkSNARK` and `NeutronNovaPrepZkSNARK`.

### Current state

```rust
// Three thin wrappers around PrecommittedState:
pub struct SpartanPrepSNARK<E>    { ps: PrecommittedState<E> }
pub struct SpartanPrepZkSNARK<E>  { ps: PrecommittedState<E> }
pub struct NeutronNovaPrepZkSNARK<E> {
  ps_step: Vec<PrecommittedState<E>>,
  ps_core: PrecommittedState<E>,
}
```

### Proposed state

```rust
// PrecommittedState fields promoted directly into the prep structs:
pub struct SpartanPrepSNARK<E> {
  cs: SatisfyingAssignment<E>,
  shared: Vec<AllocatedNum<E::Scalar>>,
  precommitted: Vec<AllocatedNum<E::Scalar>>,
  comm_shared: Option<WitnessCommitment<E>>,
  comm_precommitted: Option<WitnessCommitment<E>>,
  W: Vec<E::Scalar>,
}
// same for SpartanPrepZkSNARK

pub struct NeutronNovaPrepZkSNARK<E> {
  ps_step: Vec<SpartanPrepSNARK<E>>,  // was Vec<PrecommittedState<E>>
  ps_core: SpartanPrepSNARK<E>,       // was PrecommittedState<E>
}
```

`PrecommittedState<E>` is deleted entirely.

### Blast radius

#### `bellpepper/r1cs.rs`
- Delete `PrecommittedState<E>` struct definition (lines 375-386)
- `SpartanWitness::type PrecommittedState` becomes `type PrecommittedState = SpartanPrepSNARK<E>`
  — but this creates a circular dependency (bellpepper depends on spartan types)
- **Alternative:** keep `SpartanWitness::PrecommittedState` as an associated type, implement
  it with the promoted fields struct. `PrecommittedState` is deleted but the trait associated
  type still exists, now pointing to `SpartanPrepSNARK`.
- `RerandomizationTrait` impl moves from `PrecommittedState` to `SpartanPrepSNARK`
- `shared_witness`, `precommitted_witness`, `r1cs_instance_and_witness` impls updated to
  construct `SpartanPrepSNARK` directly instead of `PrecommittedState`
- 3 construction sites (lines 432, 599, 633)
- 11 field accesses updated to use `WitnessCommitment` pattern

#### `spartan.rs`
- `SpartanPrepSNARK` gains the 6 fields (no longer wraps `ps`)
- `prep_snark.ps` → `prep_snark` at lines 478, 589, 1114
- `SpartanWitness::type PrecommittedState = SpartanPrepSNARK<E>`
- Serialization: `SpartanPrepSNARK` already derives `Serialize/Deserialize`; the 6 fields
  must derive the same

#### `spartan_zk.rs`
- `SpartanPrepZkSNARK` gains the 6 fields
- Line 215: `prep_snark.ps.rerandomize(...)` → `prep_snark.rerandomize(...)`
- Lines 230-237: `&mut ps` → `&mut prep_snark` (now directly the state)

#### `neutronnova_zk.rs`
- `NeutronNovaPrepZkSNARK.ps_step: Vec<PrecommittedState<E>>` → `Vec<SpartanPrepSNARK<E>>`
  (or a dedicated `StepPrepSNARK` if step and core have different shapes)
- `NeutronNovaPrepZkSNARK.ps_core: PrecommittedState<E>` → `SpartanPrepSNARK<E>`
- Line 1768: `prep_snark.ps_core.rerandomize(...)` → `prep_snark.ps_core.rerandomize(...)`
  (field name stays, type changes — minimal diff)
- Lines 1776-1777: `ps_core.comm_W_shared` / `ps_core.r_W_shared`
  → `ps_core.comm_shared.as_ref().map(|wc| &wc.comm)` etc. (if WitnessCommitment applied in PR 1)
- Line 1850: `&mut ps_core` usage unchanged structurally

### Serialization note

`SpartanPrepSNARK` is the `PrepSNARK` associated type of `R1CSSNARKTrait`, which requires
`Serialize + Deserialize`. The promoted fields (`SatisfyingAssignment`, `AllocatedNum`,
`WitnessCommitment`) must all be serializable. `SatisfyingAssignment` and `AllocatedNum`
already derive these in the current code. `WitnessCommitment` (added in PR 1) will need
`Serialize/Deserialize` derives added.

### Files touched

| File | Change |
|---|---|
| `bellpepper/r1cs.rs` | Delete `PrecommittedState`; update `SpartanWitness` impl; move `RerandomizationTrait` impl |
| `spartan.rs` | Promote fields into `SpartanPrepSNARK`; remove `.ps` indirection (3 sites) |
| `spartan_zk.rs` | Promote fields into `SpartanPrepZkSNARK`; remove `.ps` indirection (2 sites) |
| `neutronnova_zk.rs` | Update `ps_step`/`ps_core` types; update field accesses (lines 1776-1777) |

### Risk

Medium. NeutronNova is complex and the parallel proving path (`par_iter` over `ps_step`)
must be preserved exactly. Recommend running the full test suite after each file change
rather than all at once.
