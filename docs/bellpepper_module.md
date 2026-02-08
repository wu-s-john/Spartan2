# Bellpepper Module Documentation

This document explains how the `src/bellpepper/` module bridges Bellpepper circuits to Spartan's proving system.

## Overview

The bellpepper module provides an **adapter/bridge** layer that:
1. Accepts circuits written using Bellpepper's `ConstraintSystem` API
2. Collects constraints and witness values
3. Converts them to Spartan's native R1CS types

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  BELLPEPPER (third-party)              SPARTAN2 (this codebase)             │
│                                                                             │
│  ConstraintSystem trait  ◄─────────────  ShapeCS implements                 │
│  LinearCombination                       SatisfyingAssignment implements    │
│  Variable, Index                                                            │
│                                                    │                        │
│                                                    ▼                        │
│                                          SplitR1CSShape (matrices A,B,C)    │
│                                          R1CSWitness (witness vector W)     │
│                                          SplitR1CSInstance (public data)    │
└─────────────────────────────────────────────────────────────────────────────┘
```

## What Comes From Where

### From Bellpepper (third-party crate)

```rust
use bellpepper_core::{
    ConstraintSystem,      // Trait - the interface circuits use
    LinearCombination,     // Sum of (Variable, coefficient) pairs
    Variable,              // Handle to a variable
    Index,                 // Index::Aux(i) or Index::Input(i)
    SynthesisError,        // Error type
    boolean::Boolean,      // Boolean gadget
    num::AllocatedNum,     // Field element gadget
};
```

### Custom in Spartan2 (`src/bellpepper/`)

| File | Contents |
|------|----------|
| `shape_cs.rs` | `ShapeCS` - collects constraint structure |
| `solver.rs` | `SatisfyingAssignment` - collects witness values |
| `r1cs.rs` | Traits + conversion to Spartan types |
| `test_shape_cs.rs` | `TestShapeCS` - extended version for testing |
| `test_r1cs.rs` | Test utilities |

## Key Structs

### `ShapeCS<E>` — Collects Constraint Structure

```rust
// src/bellpepper/shape_cs.rs
pub struct ShapeCS<E: Engine> {
    pub constraints: Vec<(
        LinearCombination<E::Scalar>,  // a
        LinearCombination<E::Scalar>,  // b
        LinearCombination<E::Scalar>,  // c
        String,                         // name/path
    )>,
    inputs: Vec<String>,   // input variable names
    aux: Vec<String>,      // witness variable names
}
```

**Behavior:**
- `alloc()` → records variable name, returns index
- `enforce()` → stores `(a, b, c)` LinearCombination tuple
- Does NOT evaluate or check satisfiability

### `SatisfyingAssignment<E>` — Collects Witness Values

```rust
// src/bellpepper/solver.rs
pub struct SatisfyingAssignment<E: Engine> {
    pub(crate) input_assignment: Vec<E::Scalar>,  // public input values
    pub(crate) aux_assignment: Vec<E::Scalar>,    // witness values
}
```

**Behavior:**
- `alloc()` → calls the value closure `f()`, stores the result
- `enforce()` → does nothing (empty function body)
- Does NOT store constraint structure

### Comparison

| Method | `ShapeCS` | `SatisfyingAssignment` |
|--------|-----------|------------------------|
| `alloc(name, \|\| value)` | Stores `name` | Stores `value` |
| `enforce(a, b, c)` | Stores constraint | **Ignored** |
| Used in | `setup()` | `prove()` |
| Produces | Matrices (A, B, C) | Witness (W) |

## Key Traits

### `SpartanShape` — Get R1CS matrices from circuit

```rust
// src/bellpepper/r1cs.rs:30-34
pub trait SpartanShape<E: Engine> {
    fn r1cs_shape<C: SpartanCircuit<E>>(circuit: &C) -> Result<SplitR1CSShape<E>, SpartanError>;
}

// Implemented by: ShapeCS
```

### `SpartanWitness` — Get witness from circuit

```rust
// src/bellpepper/r1cs.rs:55-90
pub trait SpartanWitness<E: Engine> {
    type PrecommittedState;

    fn shared_witness(...) -> Result<PrecommittedState, SpartanError>;
    fn precommitted_witness(...) -> Result<(), SpartanError>;
    fn r1cs_instance_and_witness(...) -> Result<(SplitR1CSInstance<E>, R1CSWitness<E>), SpartanError>;
}

// Implemented by: SatisfyingAssignment
```

### Multi-Round Variants

- `MultiRoundSpartanShape` — for `MultiRoundCircuit`
- `MultiRoundSpartanWitness` — for multi-round witness generation

## The Adapter/Bridge Pattern

`ShapeCS` and `SatisfyingAssignment` serve **dual roles**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   ADAPTER SIDE                              BRIDGE SIDE                     │
│   (implements Bellpepper's                  (implements Spartan's           │
│    ConstraintSystem trait)                   SpartanShape/Witness traits)   │
│                                                                             │
│   fn alloc() { collect }                    fn r1cs_shape() {               │
│   fn enforce() { collect }                      convert to matrices         │
│                                             }                               │
│                                                                             │
│   Circuit calls ──► ShapeCS stores ──► SplitR1CSShape                       │
│   (Bellpepper API)   (collected data)   (Spartan type)                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Role | Trait Implemented | Purpose |
|------|-------------------|---------|
| **Adapter** | Bellpepper's `ConstraintSystem` | Accept circuit's `alloc()`, `enforce()` calls |
| **Bridge** | Spartan's `SpartanShape`/`SpartanWitness` | Convert collected data → Spartan types |

## How Constraints Flow

### Step 1: Circuit calls `cs.enforce()`

```rust
cs.enforce(
    || "x * y = z",
    |lc| lc + x,      // a = LinearCombination([(Aux(0), 1)])
    |lc| lc + y,      // b = LinearCombination([(Aux(1), 1)])
    |lc| lc + z,      // c = LinearCombination([(Aux(2), 1)])
);
```

### Step 2: `ShapeCS` collects

```rust
self.constraints.push((
    LinearCombination([(Aux(0), 1)]),   // a
    LinearCombination([(Aux(1), 1)]),   // b
    LinearCombination([(Aux(2), 1)]),   // c
));
```

### Step 3: `add_constraint()` converts to matrix rows

```rust
// src/bellpepper/r1cs.rs:234-287
for (variable, coeff) in a_lc.iter() {
    match variable.0 {
        Index::Input(idx) => {
            // Column = idx + num_vars (inputs after witness)
            A.data.push(*coeff);
            A.indices.push(idx + num_vars);
        }
        Index::Aux(idx) => {
            // Column = idx
            A.data.push(*coeff);
            A.indices.push(idx);
        }
    }
}
```

### Step 4: Build `SplitR1CSShape`

```rust
SplitR1CSShape::new(
    num_constraints,
    num_shared,
    num_precommitted,
    num_rest,
    num_public,
    num_challenges,
    A, B, C,  // Sparse matrices
)
```

## Circuit Requirements

All circuits must implement `SpartanCircuit<E>`:

```rust
pub trait SpartanCircuit<E: Engine>: Send + Sync + Clone {
    /// Public outputs (computed outside constraint system)
    fn public_values(&self) -> Result<Vec<E::Scalar>, SynthesisError>;

    /// Variables shared with other circuits (for folding)
    fn shared<CS: ConstraintSystem<E::Scalar>>(
        &self, cs: &mut CS
    ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError>;

    /// Variables committed before challenges
    fn precommitted<CS: ConstraintSystem<E::Scalar>>(
        &self, cs: &mut CS, shared: &[AllocatedNum<E::Scalar>]
    ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError>;

    /// Number of verifier challenges
    fn num_challenges(&self) -> usize;

    /// Main circuit logic (can use challenges)
    fn synthesize<CS: ConstraintSystem<E::Scalar>>(
        &self, cs: &mut CS, shared: &[...], precommitted: &[...], challenges: Option<&[E::Scalar]>
    ) -> Result<(), SynthesisError>;
}
```

## Complete Flow: Circuit → Proof

```rust
use spartan2::{
    provider::Bn254Engine,
    spartan::SpartanSNARK,
    traits::snark::R1CSSNARKTrait,
};

// 1. Define circuit implementing SpartanCircuit<E>
let circuit = MyCircuit { ... };

// 2. Setup: ShapeCS collects structure → SplitR1CSShape
let (pk, vk) = SpartanSNARK::<Bn254Engine>::setup(circuit.clone())?;
//              └── internally calls ShapeCS::r1cs_shape()

// 3. Prep: SatisfyingAssignment collects early witness values
let prep = SpartanSNARK::prep_prove(&pk, circuit.clone(), is_small)?;
//         └── internally uses SatisfyingAssignment

// 4. Prove: Complete witness, run Spartan prover
let proof = SpartanSNARK::prove(&pk, circuit, &prep, is_small)?;

// 5. Verify
proof.verify(&vk)?;
```

## Three Layers of Traits

| Layer | Trait | Implementor | Purpose |
|-------|-------|-------------|---------|
| **Circuit** | `SpartanCircuit<E>` | Your circuit | Define the computation |
| **Adapter** | `ConstraintSystem` (Bellpepper) | `ShapeCS`, `SatisfyingAssignment` | Collect constraints/values |
| **Bridge** | `SpartanShape`, `SpartanWitness` | `ShapeCS`, `SatisfyingAssignment` | Convert to Spartan types |

## Output Types

After conversion, Spartan works with its native types:

| Type | Contents | From |
|------|----------|------|
| `SplitR1CSShape<E>` | Matrices A, B, C + dimensions | `ShapeCS::r1cs_shape()` |
| `R1CSWitness<E>` | Witness vector W + blinding | `SatisfyingAssignment::r1cs_instance_and_witness()` |
| `SplitR1CSInstance<E>` | Commitments + public values | (same as above) |

The R1CS equation verified is: **A·z ⊙ B·z = C·z**

Where z = `[W_shared | W_precommitted | W_rest | 1 | public_io | challenges]`
