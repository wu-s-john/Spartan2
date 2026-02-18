//! Small-value R1CS constraint system.
//!
//! This module provides a constraint system that uses native integer types
//! (i32, i64) instead of field elements, deferring field conversion until
//! commitment time for significant performance improvements.
//!
//! # Type System
//!
//! | Type | Purpose | Example |
//! |------|---------|---------|
//! | C (Coefficient) | Matrix entries | i32 |
//! | W (Witness) | Variable values | i32 |
//! | Acc (Accumulator) | LC evaluation | i64 |
//! | Product | Constraint check | i128 |
//!
//! # Example
//!
//! ```ignore
//! use spartan2::small_r1cs::{SmallCS, SmallConstraintSystem};
//!
//! // Create a constraint system with i32 witnesses and i32 coefficients
//! let mut cs = SmallCS::<i32, i32>::new();
//!
//! // Allocate variables
//! let x = cs.alloc(|| 3).unwrap();
//! let y = cs.alloc(|| 4).unwrap();
//! let z = cs.alloc(|| 12).unwrap();
//!
//! // Enforce x * y = z
//! cs.enforce(|lc| lc + x, |lc| lc + y, |lc| lc + z);
//!
//! // Check satisfaction with i64 accumulator
//! assert!(cs.is_satisfied::<i64>());
//! ```

mod cs;
mod lc;
mod small_cs;
mod sparse;
mod traits;

pub use cs::{SmallConstraintSystem, SynthesisError};
pub use lc::LinearCombination;
pub use small_cs::SmallCS;
pub use sparse::SmallSparseMatrix;
pub use traits::{Accumulator, Coefficient, WideningMul, Witness};

use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use crate::traits::Engine;

/// R1CS shape with coefficient type C (i32 for SHA-256).
///
/// This stores the structure of the R1CS constraint system (the A, B, C matrices)
/// with coefficients of type C instead of field elements.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "C: Serialize + for<'a> Deserialize<'a>")]
pub struct SmallR1CSShape<E: Engine, C: Coefficient> {
    /// Number of constraints.
    pub num_cons: usize,
    /// Number of auxiliary variables.
    pub num_vars: usize,
    /// Number of public inputs/outputs.
    pub num_io: usize,
    /// Matrix A with C coefficients.
    pub A: SmallSparseMatrix<C>,
    /// Matrix B with C coefficients.
    pub B: SmallSparseMatrix<C>,
    /// Matrix C with C coefficients.
    #[serde(rename = "C_matrix")]
    pub C_mat: SmallSparseMatrix<C>,
    #[serde(skip)]
    _phantom: PhantomData<E>,
}

impl<E: Engine, C: Coefficient> SmallR1CSShape<E, C> {
    /// Create a new R1CS shape from matrices.
    pub fn new(
        num_cons: usize,
        num_vars: usize,
        num_io: usize,
        A: SmallSparseMatrix<C>,
        B: SmallSparseMatrix<C>,
        C_mat: SmallSparseMatrix<C>,
    ) -> Self {
        Self {
            num_cons,
            num_vars,
            num_io,
            A,
            B,
            C_mat,
            _phantom: PhantomData,
        }
    }

    /// Multiply all matrices by witness vector z.
    ///
    /// Input: z is Vec<W> (e.g., i32 witnesses)
    /// Output: (Az, Bz, Cz) each Vec<Acc> (e.g., i64 accumulators)
    pub fn multiply_vec<W, Acc>(&self, z: &[W]) -> (Vec<Acc>, Vec<Acc>, Vec<Acc>)
    where
        W: Witness + Send + Sync,
        C: WideningMul<W, Acc> + Send + Sync,
        Acc: Accumulator + Send,
    {
        let (az, (bz, cz)) = rayon::join(
            || self.A.multiply_vec_widening_unchecked(z),
            || {
                rayon::join(
                    || self.B.multiply_vec_widening_unchecked(z),
                    || self.C_mat.multiply_vec_widening_unchecked(z),
                )
            },
        );
        (az, bz, cz)
    }

    /// Check constraint satisfaction: Az × Bz = Cz.
    ///
    /// Uses i128 for the product to avoid overflow.
    pub fn is_sat<W, Acc>(&self, z: &[W]) -> bool
    where
        W: Witness + Send + Sync,
        C: WideningMul<W, Acc> + Send + Sync,
        Acc: Accumulator + Send + Into<i128>,
    {
        let (az, bz, cz) = self.multiply_vec::<W, Acc>(z);
        az.iter().zip(&bz).zip(&cz).all(|((a, b), c)| {
            let a128: i128 = (*a).into();
            let b128: i128 = (*b).into();
            let c128: i128 = (*c).into();
            a128 * b128 == c128
        })
    }

    /// Find the first unsatisfied constraint (for debugging).
    pub fn which_is_unsatisfied<W, Acc>(&self, z: &[W]) -> Option<usize>
    where
        W: Witness + Send + Sync,
        C: WideningMul<W, Acc> + Send + Sync,
        Acc: Accumulator + Send + Into<i128>,
    {
        let (az, bz, cz) = self.multiply_vec::<W, Acc>(z);
        az.iter()
            .zip(&bz)
            .zip(&cz)
            .enumerate()
            .find(|(_, ((a, b), c))| {
                let a128: i128 = (**a).into();
                let b128: i128 = (**b).into();
                let c128: i128 = (**c).into();
                a128 * b128 != c128
            })
            .map(|(i, _)| i)
    }
}

/// R1CS witness with witness type W (i32 for SHA-256).
///
/// This stores the witness values as native integers,
/// deferring field conversion until commitment time.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "W: Serialize + for<'a> Deserialize<'a>")]
pub struct SmallR1CSWitness<E: Engine, W: Witness> {
    /// Witness values (auxiliary variables).
    pub W: Vec<W>,
    #[serde(skip)]
    _phantom: PhantomData<E>,
}

impl<E: Engine, W: Witness> SmallR1CSWitness<E, W> {
    /// Create a new witness from values.
    pub fn new(witness: Vec<W>) -> Self {
        Self {
            W: witness,
            _phantom: PhantomData,
        }
    }

    /// Number of witness elements.
    pub fn len(&self) -> usize {
        self.W.len()
    }

    /// Check if witness is empty.
    pub fn is_empty(&self) -> bool {
        self.W.is_empty()
    }
}

/// R1CS instance (public interface) with witness type W.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "W: Serialize + for<'a> Deserialize<'a>")]
pub struct SmallR1CSInstance<E: Engine, W: Witness> {
    /// Public input/output values.
    pub X: Vec<W>,
    #[serde(skip)]
    _phantom: PhantomData<E>,
}

impl<E: Engine, W: Witness> SmallR1CSInstance<E, W> {
    /// Create a new instance from public values.
    pub fn new(public_values: Vec<W>) -> Self {
        Self {
            X: public_values,
            _phantom: PhantomData,
        }
    }
}

/// Helper to convert SmallCS to SmallR1CSShape.
impl<W: Witness, C: Coefficient> SmallCS<W, C> {
    /// Convert to SmallR1CSShape.
    pub fn to_shape<E: Engine>(&self) -> SmallR1CSShape<E, C> {
        let (a, b, c) = self.build_matrices();
        SmallR1CSShape::new(
            self.num_constraints(),
            self.num_aux(),
            self.num_inputs() - 1, // -1 for implicit ONE
            a,
            b,
            c,
        )
    }

    /// Convert to SmallR1CSWitness.
    pub fn to_witness<E: Engine>(&self) -> SmallR1CSWitness<E, W> {
        SmallR1CSWitness::new(self.aux.clone())
    }

    /// Convert to SmallR1CSInstance (public values).
    pub fn to_instance<E: Engine>(&self) -> SmallR1CSInstance<E, W> {
        SmallR1CSInstance::new(self.inputs[1..].to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::PallasHyraxEngine;

    type E = PallasHyraxEngine;

    #[test]
    fn test_small_r1cs_shape() {
        let mut cs = SmallCS::<i32, i32>::new();

        let x = cs.alloc(|| 3).unwrap();
        let y = cs.alloc(|| 4).unwrap();
        let z = cs.alloc(|| 12).unwrap();

        cs.enforce(|lc| lc + x, |lc| lc + y, |lc| lc + z);

        let shape: SmallR1CSShape<E, i32> = cs.to_shape();
        let z_vec = cs.z();

        assert!(shape.is_sat::<i32, i64>(&z_vec));
    }

    #[test]
    fn test_shape_multiply_vec() {
        let mut cs = SmallCS::<i32, i32>::new();

        let x = cs.alloc(|| 2).unwrap();
        let y = cs.alloc(|| 3).unwrap();
        let z = cs.alloc(|| 6).unwrap();

        cs.enforce(|lc| lc + x, |lc| lc + y, |lc| lc + z);

        let shape: SmallR1CSShape<E, i32> = cs.to_shape();
        let z_vec = cs.z();

        let (az, bz, cz) = shape.multiply_vec::<i32, i64>(&z_vec);

        // az[0] should be x = 2
        // bz[0] should be y = 3
        // cz[0] should be z = 6
        assert_eq!(az[0], 2);
        assert_eq!(bz[0], 3);
        assert_eq!(cz[0], 6);

        // Check: az * bz = cz
        assert_eq!(az[0] * bz[0], cz[0]);
    }
}
