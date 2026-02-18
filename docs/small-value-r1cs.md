# Small-Value R1CS Design

This document describes the design for a generic R1CS system that supports small integer types (i8, i16, i32, i64) instead of field elements, enabling significant performance improvements.

## Motivation

Field arithmetic is expensive:

| Operation | Field (current) | Small Value (goal) |
|-----------|----------------|-------------------|
| Witness alloc | Montgomery conversion | Native store |
| Multiplication | ~50 cycles (modular) | ~3 cycles (native) |
| Memory per value | 32 bytes | 4 bytes |
| Cache efficiency | Poor | Excellent |
| Matrix-vector (sumcheck) | Field ops | i64 accumulate |

Most R1CS circuits have:
- Coefficients that are small: 0, 1, -1, occasionally 2 or small constants
- Witness values that fit in 32 bits

We're paying field arithmetic costs throughout the entire prover when the actual values fit in machine integers.

## Architecture Overview

### Type Parameters

| Type | Purpose | Concrete (SHA256) | Size |
|------|---------|-------------------|------|
| **C** (Coefficient) | Matrix entries, LC terms | i32 | 4 bytes |
| **W** (Witness) | Variable values | i32 | 4 bytes |
| **Acc** (Accumulator) | LC evaluation = Σ C×W | i64 | 8 bytes |
| **Product** | Constraint check Acc×Acc | i128 | 16 bytes |

**Key relationships:**
```
LinearCombination<C> = Vec<(Variable, C)>     // stores i32 coefficients
Witness vector z: Vec<W>                       // stores i32 values
LC evaluation: Σ C×W → Acc                     // i32 × i32 → i64
Constraint: Az × Bz = Cz → Acc × Acc          // i64 × i64 → i128
```

### Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│  1. SYNTHESIS (small values)                                │
│                                                             │
│     SmallCS<W=i32, C=i32>                                   │
│       - witness values: Vec<W>  (i32)                       │
│       - matrix coeffs: Vec<C>   (i32)                       │
│                                                             │
│     No field conversion needed here                         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  2. MATRIX-VECTOR MULTIPLY (sumcheck hot path)              │
│                                                             │
│     Az, Bz, Cz where:                                       │
│       - A, B, C have C=i32 coefficients                     │
│       - z has W=i32 witness values                          │
│       - result is Acc=i64 (accumulator)                     │
│                                                             │
│     Σ (C × W) → Acc  (i32 × i32 → i64)                      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  3. CONSTRAINT CHECK                                        │
│                                                             │
│     Az[i] × Bz[i] = Cz[i]                                   │
│     Acc × Acc → i128 (for overflow-safe comparison)         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  4. COMMITMENT (crypto boundary) ← CONVERSION HERE          │
│                                                             │
│     PCS::commit(witness, ...) needs Vec<F>                  │
│                                                             │
│     witness: Vec<W=i32> → Vec<F>                            │
│                       ↑                                     │
│                  signed_to_field()                          │
└─────────────────────────────────────────────────────────────┘
```

### Conversion Points

| Location | From | To | When |
|----------|------|-----|------|
| Witness commit | `W=i32` | `F` | Once per proof |
| Sumcheck finalize | `Acc=i64` | `F` | Once per sumcheck |

**Note**: Matrix coefficients (C=i32) never need field conversion. The shape stays `R1CSShape<E, C=i32>` throughout.

## Overflow Analysis

For `Az` where:
- `A` has coefficients in `[-2³¹, 2³¹-1]` (C=i32)
- `z` has values in `[-2³¹, 2³¹-1]` (W=i32)
- Each row has at most `N` non-zeros

Each product: `|C × W| ≤ 2³¹ × 2³¹ = 2⁶²` → fits in i64
Row sum: `|Σ C × W| ≤ N × 2⁶²`

For N = 1000 non-zeros per row: `≤ 2⁷²` → **overflows i64!**

**Practical bounds for SHA-256:**
- Coefficients are small: 0, 1, -1, powers of 2 up to 2¹⁶
- Max coefficient in 2-limb addition: 2¹⁶ = 65,536
- Witnesses are bits (0 or 1) or small integers
- Row sum: safe within i64

**2-Limb Addition (key technique):**

To keep coefficients within i32 bounds, 32-bit addition is split into two 16-bit limbs:
```
Lower limb: Σ operand_lower = result_lower + carry × 2¹⁶
Upper limb: Σ operand_upper + carry = result_upper + overflow × 2¹⁶
```
Max coefficient: 2¹⁶ = 65,536 (fits comfortably in i32).

For addmany(4 operands): **38 constraints** (32 result bits + 2 carry + 2 overflow + 2 equality).

See [small-value-sha256.md](small-value-sha256.md) for full gadget implementations.

The constraint check `Az × Bz = Cz`:
- `Az[i]`, `Bz[i]` are Acc=i64
- Product `Acc × Acc` could be `2¹²⁶` → needs `i128`

---

## Implementation

### Core Traits

```rust
// src/r1cs/traits.rs

use std::ops::{Add, Sub, Mul, Neg};
use num_traits::{Zero, One};

/// Trait for matrix coefficients (C = i32 for SHA-256)
pub trait Coefficient:
    Copy + Clone + Default + Send + Sync + Debug + PartialEq + Eq
    + Zero + One + Neg<Output = Self>
{}

impl<T> Coefficient for T where
    T: Copy + Clone + Default + Send + Sync + Debug + PartialEq + Eq
       + Zero + One + Neg<Output = Self>
{}

/// Trait for witness values (W = i32 for SHA-256)
pub trait Witness:
    Copy + Clone + Default + Send + Sync + Debug
    + Zero + One
    + Add<Output = Self> + Sub<Output = Self>
    + Mul<Output = Self> + Neg<Output = Self>
{}

impl<T> Witness for T where
    T: Copy + Clone + Default + Send + Sync + Debug
       + Zero + One
       + Add<Output = Self> + Sub<Output = Self>
       + Mul<Output = Self> + Neg<Output = Self>
{}

/// Trait for accumulator type (Acc = i64 for SHA-256)
/// Used for LC evaluation: Σ C×W → Acc
pub trait Accumulator:
    Copy + Clone + Default + Send + Sync + Debug
    + Zero + One
    + Add<Output = Self> + Sub<Output = Self>
    + Mul<Output = Self> + Neg<Output = Self>
{}

impl<T> Accumulator for T where
    T: Copy + Clone + Default + Send + Sync + Debug
       + Zero + One
       + Add<Output = Self> + Sub<Output = Self>
       + Mul<Output = Self> + Neg<Output = Self>
{}
```

### Variable and Linear Combination

```rust
// src/r1cs/lc.rs

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Variable {
    One,
    Input(usize),
    Aux(usize),
}

/// Linear combination with C (i32) coefficients
/// Evaluation against W (i32) witnesses produces Acc (i64)
#[derive(Clone, Debug, Default)]
pub struct LinearCombination<C: Coefficient> {
    pub terms: Vec<(Variable, C)>,  // stores C=i32 coefficients
}

impl<C: Coefficient> LinearCombination<C> {
    pub fn zero() -> Self { Self { terms: vec![] } }

    /// Evaluate LC against witnesses: Σ C×W → Acc
    pub fn evaluate<W, Acc>(&self, one: W, inputs: &[W], aux: &[W]) -> Acc
    where
        W: Witness + Into<Acc>,
        C: Into<Acc>,
        Acc: Accumulator,
    {
        self.terms.iter().fold(Acc::zero(), |acc, (var, coeff)| {
            let w: W = match var {
                Variable::One => one,
                Variable::Input(i) => inputs[*i],
                Variable::Aux(i) => aux[*i],
            };
            let c: Acc = (*coeff).into();  // C=i32 → Acc=i64
            let w: Acc = w.into();          // W=i32 → Acc=i64
            acc + c * w                     // Acc=i64 arithmetic
        })
    }
}

impl<C: Coefficient> Add<Variable> for LinearCombination<C> {
    type Output = Self;
    fn add(mut self, var: Variable) -> Self {
        self.terms.push((var, C::one()));
        self
    }
}

impl<C: Coefficient> Sub<Variable> for LinearCombination<C> {
    type Output = Self;
    fn sub(mut self, var: Variable) -> Self {
        self.terms.push((var, -C::one()));
        self
    }
}

impl<C: Coefficient> Add<(C, Variable)> for LinearCombination<C> {
    type Output = Self;
    fn add(mut self, (coeff, var): (C, Variable)) -> Self {
        self.terms.push((var, coeff));
        self
    }
}
```

### Constraint System Trait

```rust
// src/r1cs/cs.rs

/// Constraint system with W (i32) witnesses and C (i32) coefficients
pub trait ConstraintSystem<W: Witness, C: Coefficient> {
    fn alloc<F>(&mut self, f: F) -> Variable
    where F: FnOnce() -> W;      // allocates W=i32 witness

    fn alloc_input<F>(&mut self, f: F) -> Variable
    where F: FnOnce() -> W;      // allocates W=i32 public input

    fn enforce<FA, FB, FC>(&mut self, a: FA, b: FB, c: FC)
    where
        FA: FnOnce(LinearCombination<C>) -> LinearCombination<C>,
        FB: FnOnce(LinearCombination<C>) -> LinearCombination<C>,
        FC: FnOnce(LinearCombination<C>) -> LinearCombination<C>;

    fn one() -> Variable { Variable::One }
}
```

### Generic Sparse Matrix

```rust
// src/r1cs/sparse.rs

/// Sparse matrix with C (i32) coefficients
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparseMatrix<C: Coefficient> {
    pub data: Vec<C>,         // C=i32 coefficients
    pub indices: Vec<usize>,
    pub indptr: Vec<usize>,
    pub cols: usize,
}

impl<C: Coefficient> SparseMatrix<C> {
    pub fn empty() -> Self {
        Self { data: vec![], indices: vec![], indptr: vec![0], cols: 0 }
    }

    /// Matrix-vector multiply: C × W → Acc
    /// Input: z is Vec<W=i32>
    /// Output: Vec<Acc=i64>
    pub fn multiply_vec<W, Acc>(&self, z: &[W]) -> Vec<Acc>
    where
        W: Witness + Into<Acc>,
        C: Into<Acc>,
        Acc: Accumulator,
    {
        self.indptr
            .par_windows(2)
            .map(|ptrs| {
                let mut acc = Acc::zero();
                for i in ptrs[0]..ptrs[1] {
                    let c: Acc = self.data[i].into();      // C=i32 → Acc=i64
                    let w: Acc = z[self.indices[i]].into(); // W=i32 → Acc=i64
                    acc = acc + c * w;                      // Acc=i64 arithmetic
                }
                acc
            })
            .collect()
    }
}
```

### Generic R1CS Shape

```rust
// src/r1cs/mod.rs

/// R1CS shape with C (i32) coefficient matrices
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct R1CSShape<E: Engine, C: Coefficient = E::Scalar> {
    pub num_cons: usize,
    pub num_vars: usize,
    pub num_io: usize,
    pub A: SparseMatrix<C>,   // C=i32 coefficients
    pub B: SparseMatrix<C>,
    pub C: SparseMatrix<C>,
    #[serde(skip)]
    _phantom: PhantomData<E>,
}

impl<E: Engine, C: Coefficient> R1CSShape<E, C> {
    /// Multiply all matrices by witness vector
    /// Input: z is Vec<W=i32>
    /// Output: (Az, Bz, Cz) each Vec<Acc=i64>
    pub fn multiply_vec<W, Acc>(&self, z: &[W]) -> (Vec<Acc>, Vec<Acc>, Vec<Acc>)
    where
        W: Witness + Into<Acc>,
        C: Into<Acc>,
        Acc: Accumulator,
    {
        let (Az, (Bz, Cz)) = rayon::join(
            || self.A.multiply_vec(z),
            || rayon::join(|| self.B.multiply_vec(z), || self.C.multiply_vec(z)),
        );
        (Az, Bz, Cz)
    }

    /// Check constraint satisfaction: Az × Bz = Cz
    /// Uses i128 for product to avoid overflow
    pub fn is_sat<W, Acc>(&self, z: &[W]) -> bool
    where
        W: Witness + Into<Acc>,
        C: Into<Acc>,
        Acc: Accumulator + Into<i128>,
    {
        let (az, bz, cz): (Vec<Acc>, Vec<Acc>, Vec<Acc>) = self.multiply_vec(z);
        az.iter().zip(&bz).zip(&cz).all(|((a, b), c)| {
            let a128: i128 = (*a).into();
            let b128: i128 = (*b).into();
            let c128: i128 = (*c).into();
            a128 * b128 == c128  // i128 multiplication
        })
    }
}
```

### Generic R1CS Witness

```rust
// src/r1cs/mod.rs

/// R1CS witness with W (i32) values
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct R1CSWitness<E: Engine, W: Witness = E::Scalar> {
    pub W: Vec<W>,            // W=i32 witness values
    pub r_W: Blind<E>,
    #[serde(skip)]
    _phantom: PhantomData<E>,
}

impl<E: Engine, W: Witness> R1CSWitness<E, W> {
    pub fn new(witness: Vec<W>, r_W: Blind<E>) -> Self {
        Self { W: witness, r_W, _phantom: PhantomData }
    }

    /// Convert W=i32 to field and commit (only place needing field conversion)
    pub fn commit(&self, ck: &CommitmentKey<E>) -> Result<Commitment<E>, SpartanError>
    where
        W: Into<i64>,
    {
        let field_W: Vec<E::Scalar> = self.W.iter()
            .map(|&w| signed_to_field(w.into()))
            .collect();
        PCS::<E>::commit(ck, &field_W, &self.r_W, true)
    }
}

/// Helper to convert signed integer to field element
pub fn signed_to_field<F: PrimeField>(x: i64) -> F {
    if x >= 0 {
        F::from(x as u64)
    } else {
        -F::from((-x) as u64)
    }
}
```

### Generic Spartan Circuit Trait

```rust
// src/traits/circuit.rs

/// Circuit trait with W (i32) witnesses and C (i32) coefficients
pub trait SpartanCircuit<E: Engine, W: Witness = E::Scalar, C: Coefficient = E::Scalar>:
    Send + Sync + Clone
{
    fn public_values(&self) -> Result<Vec<W>, SynthesisError>;

    fn shared<CS: ConstraintSystem<W, C>>(
        &self,
        cs: &mut CS,
    ) -> Result<Vec<Variable>, SynthesisError>;

    fn precommitted<CS: ConstraintSystem<W, C>>(
        &self,
        cs: &mut CS,
        shared: &[Variable],
    ) -> Result<Vec<Variable>, SynthesisError>;

    fn num_challenges(&self) -> usize;

    fn synthesize<CS: ConstraintSystem<W, C>>(
        &self,
        cs: &mut CS,
        shared: &[Variable],
        precommitted: &[Variable],
        challenges: Option<&[W]>,
    ) -> Result<(), SynthesisError>;
}
```

### SmallCS Implementation

```rust
// src/r1cs/small_cs.rs

/// Constraint system with W (i32) witnesses and C (i32) coefficients
pub struct SmallCS<W: Witness, C: Coefficient> {
    pub inputs: Vec<W>,       // W=i32 public inputs
    pub aux: Vec<W>,          // W=i32 auxiliary witnesses
    pub constraints: Vec<(
        LinearCombination<C>,  // A with C=i32 coefficients
        LinearCombination<C>,  // B
        LinearCombination<C>,  // C
    )>,
}

impl<W: Witness, C: Coefficient> SmallCS<W, C> {
    pub fn new() -> Self {
        Self {
            inputs: vec![W::one()],  // first input is always 1
            aux: vec![],
            constraints: vec![],
        }
    }

    /// Build R1CSShape with C (i32) coefficient matrices
    pub fn to_shape<E: Engine>(&self) -> R1CSShape<E, C> {
        let (A, B, C_mat) = self.build_matrices();
        R1CSShape {
            num_cons: self.constraints.len(),
            num_vars: self.aux.len(),
            num_io: self.inputs.len() - 1,
            A,
            B,
            C: C_mat,
            _phantom: PhantomData,
        }
    }

    /// Build R1CSWitness with W (i32) values
    pub fn to_witness<E: Engine>(&self, r_W: Blind<E>) -> R1CSWitness<E, W> {
        R1CSWitness::new(self.aux.clone(), r_W)
    }

    /// Full witness vector z = [aux | 1 | inputs[1..]]
    pub fn z(&self) -> Vec<W> {
        let mut z = self.aux.clone();
        z.push(W::one());
        z.extend(self.inputs[1..].iter().cloned());
        z
    }

    fn build_matrices(&self) -> (SparseMatrix<C>, SparseMatrix<C>, SparseMatrix<C>) {
        // ... (same implementation, builds C=i32 coefficient matrices)
    }
}

impl<W: Witness, C: Coefficient> ConstraintSystem<W, C> for SmallCS<W, C> {
    fn alloc<F>(&mut self, f: F) -> Variable
    where F: FnOnce() -> W     // allocates W=i32 witness
    {
        self.aux.push(f());
        Variable::Aux(self.aux.len() - 1)
    }

    fn alloc_input<F>(&mut self, f: F) -> Variable
    where F: FnOnce() -> W     // allocates W=i32 public input
    {
        self.inputs.push(f());
        Variable::Input(self.inputs.len() - 1)
    }

    fn enforce<FA, FB, FC>(&mut self, a: FA, b: FB, c: FC)
    where
        FA: FnOnce(LinearCombination<C>) -> LinearCombination<C>,
        FB: FnOnce(LinearCombination<C>) -> LinearCombination<C>,
        FC: FnOnce(LinearCombination<C>) -> LinearCombination<C>,
    {
        self.constraints.push((
            a(LinearCombination::zero()),
            b(LinearCombination::zero()),
            c(LinearCombination::zero()),
        ));
    }
}
```

---

## Usage Example

```rust
#[derive(Clone)]
struct MulCircuit {
    x: i32,
    y: i32,
}

// W=i32 witnesses, C=i32 coefficients
impl<E: Engine> SpartanCircuit<E, i32, i32> for MulCircuit {
    fn public_values(&self) -> Result<Vec<i32>, SynthesisError> {
        Ok(vec![])
    }

    fn shared<CS: ConstraintSystem<i32, i32>>(
        &self,
        _cs: &mut CS,
    ) -> Result<Vec<Variable>, SynthesisError> {
        Ok(vec![])
    }

    fn precommitted<CS: ConstraintSystem<i32, i32>>(
        &self,
        _cs: &mut CS,
        _shared: &[Variable],
    ) -> Result<Vec<Variable>, SynthesisError> {
        Ok(vec![])
    }

    fn num_challenges(&self) -> usize { 0 }

    fn synthesize<CS: ConstraintSystem<i32, i32>>(
        &self,
        cs: &mut CS,
        _shared: &[Variable],
        _precommitted: &[Variable],
        _challenges: Option<&[i32]>,
    ) -> Result<(), SynthesisError> {
        // Native i32 arithmetic - no field ops!
        let a = cs.alloc(|| self.x);      // W=i32
        let b = cs.alloc(|| self.y);      // W=i32
        let c = cs.alloc(|| self.x * self.y);  // W=i32

        // Enforce a * b = c (coefficients are C=i32)
        cs.enforce(
            |lc| lc + a,
            |lc| lc + b,
            |lc| lc + c,
        );

        Ok(())
    }
}

fn main() {
    let circuit = MulCircuit { x: 5, y: 7 };

    // Synthesize with W=i32 witnesses, C=i32 coefficients
    let mut cs: SmallCS<i32, i32> = SmallCS::new();
    circuit.synthesize(&mut cs, &[], &[], None).unwrap();

    // Check: Az × Bz = Cz
    // multiply_vec: C=i32 × W=i32 → Acc=i64
    let z: Vec<i32> = cs.z();
    let shape: R1CSShape<BN254Engine, i32> = cs.to_shape();
    let (az, bz, cz): (Vec<i64>, Vec<i64>, Vec<i64>) = shape.multiply_vec(&z);

    // Constraint check uses i128 for Acc × Acc
    assert!(az.iter().zip(&bz).zip(&cz).all(|((a, b), c)| {
        (*a as i128) * (*b as i128) == (*c as i128)
    }));

    // Only convert W=i32 → Field at commitment time
    let witness: R1CSWitness<BN254Engine, i32> = cs.to_witness(blind);
    let commitment = witness.commit(&ck).unwrap();
}
```

---

## Type Flow Summary

```
Circuit: SpartanCircuit<E, W=i32, C=i32>
            ↓
Synthesis:  SmallCS<W=i32, C=i32>
            ↓
Shape:      R1CSShape<E, C=i32>        ← i32 coefficients
Witness:    R1CSWitness<E, W=i32>      ← i32 values
            ↓
Multiply:   Az, Bz, Cz : Vec<Acc=i64>  ← i64 accumulator
            ↓
Check:      Az × Bz = Cz (i128)        ← constraint satisfaction
            ↓
Commit:     Vec<W=i32> → Vec<F>        ← field conversion here only
```

---

## Constraints That Need Field Operations

Some constraints fundamentally require field properties:

### Non-equality `x ≠ y` (Traditional)
```
allocate w = (x - y)⁻¹
enforce: (x - y) · w = 1
```
This requires field inverse - integers don't have multiplicative inverses.

### Integer-Friendly Alternatives

**For `x ≠ y` (bit decomposition):**
If `|x - y| < 2^k`, decompose `d = x - y` into bits:
```
d = Σ bᵢ · 2ⁱ
bᵢ · (1 - bᵢ) = 0                     // each bit is boolean
(1 - b₀)(1 - b₁)...(1 - bₖ₋₁) = 0    // at least one bit is 1
```

**For conditional/branching:**
Force selector to be boolean instead of using inverse trick:
```
s ∈ {0, 1}          // boolean constraint: s(1-s) = 0
s · (x - y) = 0     // now just integer multiplication
```

---

## File Structure

```
src/r1cs/
├── mod.rs           # R1CSShape<E, C>, R1CSWitness<E, W>
├── sparse.rs        # SparseMatrix<C>
├── traits.rs        # Coefficient, Witness, Accumulator
├── lc.rs            # Variable, LinearCombination<C>
├── cs.rs            # ConstraintSystem<W, C> trait
└── small_cs.rs      # SmallCS<W, C> implementation
```

---

## Implementation Tasks

1. Define `Witness`, `Coefficient`, `Accumulator` traits in `src/r1cs/traits.rs`
2. Create `Variable` and `LinearCombination<C>` in `src/r1cs/lc.rs`
3. Create `ConstraintSystem<W, C>` trait in `src/r1cs/cs.rs`
4. Make `SparseMatrix<C>` generic in `src/r1cs/sparse.rs`
5. Make `R1CSShape<E, C>` and `R1CSWitness<E, W>` generic
6. Create `SmallCS<W, C>` implementation
7. Update `SpartanCircuit<E, W, C>` trait
8. Update `SpartanSNARK` to use generic types
9. Update `NeutronNova` to use generic types
