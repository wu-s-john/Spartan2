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

mod batching;
mod cs;
mod lc;
mod small_cs;
mod sparse;
mod traits;

pub use batching::BatchingSmallCS;
pub use cs::{SmallConstraintSystem, SynthesisError};
pub use lc::LinearCombination;
pub use small_cs::SmallCS;
pub use sparse::SmallSparseMatrix;
pub use traits::{Accumulator, Coefficient, SmallMultiEqCS, WideningMul, Witness};

use crate::r1cs::{R1CSShape, SplitR1CSShape};
use crate::traits::Engine;

/// Helper to convert SmallCS to R1CS shapes with small coefficients.
impl<W: Witness, C: Coefficient> SmallCS<W, C> {
    /// Convert to R1CSShape with coefficient type C.
    ///
    /// Returns `R1CSShape<E, C>` where C is typically i32 for small-value circuits.
    /// Use `multiply_vec_widening` and `is_sat_widening` methods on the resulting shape.
    pub fn to_r1cs_shape<E: Engine>(&self) -> R1CSShape<E, C> {
        let (a, b, c) = self.build_matrices();
        R1CSShape::new_generic(
            self.num_constraints(),
            self.num_aux(),
            self.num_inputs() - 1, // -1 for implicit ONE
            a.into(),
            b.into(),
            c.into(),
        )
    }

    /// Convert to SplitR1CSShape with coefficient type C.
    ///
    /// Uses the segment boundaries set by `end_shared_phase()` and
    /// `end_precommitted_phase()` to determine shared/precommitted/rest split.
    /// If no phases were marked, all variables go into "rest" for backwards compatibility.
    pub fn to_split_r1cs_shape<E: Engine>(&self) -> SplitR1CSShape<E, C>
    where
        C: Copy + Send + Sync,
    {
        let (a, b, c) = self.build_matrices();
        let num_cons_unpadded = self.num_constraints();
        let num_public = self.num_inputs() - 1; // -1 for implicit ONE

        SplitR1CSShape::new_simple(
            num_cons_unpadded,
            self.num_shared(),
            self.num_precommitted(),
            self.num_rest(),
            num_public,
            0, // num_challenges
            a.into(),
            b.into(),
            c.into(),
        )
    }

    /// Get the witness values (auxiliary variables).
    ///
    /// Returns the raw witness values as Vec<W> for use in small-value proving.
    pub fn witness_values(&self) -> Vec<W> {
        self.aux.clone()
    }

    /// Get the public input values (excluding the implicit ONE).
    ///
    /// Returns the public inputs as Vec<W> for use in small-value proving.
    pub fn public_values(&self) -> Vec<W> {
        self.inputs[1..].to_vec()
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

        let shape: R1CSShape<E, i32> = cs.to_r1cs_shape();
        let z_vec = cs.z();

        assert!(shape.is_sat_widening::<i32, i64>(&z_vec).unwrap());
    }

    #[test]
    fn test_shape_multiply_vec() {
        let mut cs = SmallCS::<i32, i32>::new();

        let x = cs.alloc(|| 2).unwrap();
        let y = cs.alloc(|| 3).unwrap();
        let z = cs.alloc(|| 6).unwrap();

        cs.enforce(|lc| lc + x, |lc| lc + y, |lc| lc + z);

        let shape: R1CSShape<E, i32> = cs.to_r1cs_shape();
        let z_vec = cs.z();

        let (az, bz, cz) = shape.multiply_vec_widening::<i32, i64>(&z_vec).unwrap();

        // az[0] should be x = 2
        // bz[0] should be y = 3
        // cz[0] should be z = 6
        assert_eq!(az[0], 2);
        assert_eq!(bz[0], 3);
        assert_eq!(cz[0], 6);

        // Check: az * bz = cz
        assert_eq!(az[0] * bz[0], cz[0]);
    }

    #[test]
    fn test_split_shape_multiply_vec_widening() {
        // Create a simple circuit: x * y = z where x=5, y=7, z=35
        let mut cs = SmallCS::<i32, i32>::new();

        let x = cs.alloc(|| 5).unwrap();
        let y = cs.alloc(|| 7).unwrap();
        let z = cs.alloc(|| 35).unwrap();

        cs.enforce(|lc| lc + x, |lc| lc + y, |lc| lc + z);

        // Convert to SplitR1CSShape<E, i32>
        let split_shape: SplitR1CSShape<E, i32> = cs.to_split_r1cs_shape();

        // Build z vector with padding for split shape
        // z = [W (padded) | 1 | X]
        let num_rest = split_shape.num_rest;
        let mut z_vec: Vec<i32> = vec![0; num_rest];
        z_vec[0] = 5;  // x
        z_vec[1] = 7;  // y
        z_vec[2] = 35; // z
        z_vec.push(1); // constant ONE
        // No public inputs in this circuit

        let (az, bz, cz) = split_shape.multiply_vec_widening::<i32, i64>(&z_vec).unwrap();

        // First constraint: x * y = z
        // az[0] should be x = 5
        // bz[0] should be y = 7
        // cz[0] should be z = 35
        assert_eq!(az[0], 5);
        assert_eq!(bz[0], 7);
        assert_eq!(cz[0], 35);

        // Verify A·z × B·z = C·z
        assert_eq!(az[0] * bz[0], cz[0]);
    }

    #[test]
    fn test_split_shape_with_public_inputs() {
        // Create a circuit with public inputs
        // Constraint: x * y = z (witness multiplication)
        // Plus: expose x as public input
        let mut cs = SmallCS::<i32, i32>::new();

        let x = cs.alloc(|| 3).unwrap();
        let y = cs.alloc(|| 4).unwrap();
        let z = cs.alloc(|| 12).unwrap();

        // Constraint: x * y = z
        cs.enforce(|lc| lc + x, |lc| lc + y, |lc| lc + z);

        // Expose x as public input
        let x_pub = cs.alloc_input(|| 3).unwrap();
        cs.enforce(
            |lc| lc + x,
            |lc| lc + SmallCS::<i32, i32>::one(),
            |lc| lc + x_pub,
        );

        assert!(cs.is_satisfied::<i64>());

        // Convert to split shape
        let split_shape: SplitR1CSShape<E, i32> = cs.to_split_r1cs_shape();

        // Verify structure
        assert_eq!(split_shape.num_public, 1, "Should have 1 public input");
        assert_eq!(split_shape.num_cons_unpadded, 2, "Should have 2 constraints");
    }
}
