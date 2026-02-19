//! Batching wrapper for SmallCS that accumulates equality constraints.
//!
//! This module provides `BatchingSmallCS<K>` which batches up to K equality
//! constraints into a single R1CS constraint by scaling each LC by 2^j.
//!
//! # Coefficient Bounds
//!
//! With K=17 and max original coefficient 2^14:
//! - Batching multiplier: up to 2^16
//! - Combined: 2^14 × 2^16 = 2^30 < 2^31 (fits i32)

use super::{
    cs::{SmallConstraintSystem, SynthesisError},
    lc::LinearCombination,
    small_cs::SmallCS,
    traits::{Coefficient, SmallMultiEqCS, Witness},
};
use bellpepper_core::Variable;

/// Batching wrapper for SmallCS that accumulates equality constraints.
///
/// Batches up to K equality constraints into one by scaling each by 2^j.
/// Max coefficient after batching: original_coeff × 2^(K-1)
///
/// # Type Parameters
///
/// - `W`: Witness type (e.g., i32)
/// - `C`: Coefficient type (e.g., i32)
/// - `K`: Maximum number of equality constraints to batch (e.g., 17)
///
/// # Example
///
/// ```ignore
/// let mut cs = SmallCS::<i32, i32>::new();
/// {
///     let mut batched = BatchingSmallCS::<i32, i32, 17>::new(&mut cs);
///     // Add equality constraints - they will be batched
///     batched.enforce_equal(&lhs1, &rhs1);
///     batched.enforce_equal(&lhs2, &rhs2);
///     // ...
/// } // Drop flushes remaining constraints
/// ```
pub struct BatchingSmallCS<'a, W: Witness, C: Coefficient + From<i32>, const K: usize> {
    /// The underlying constraint system
    cs: &'a mut SmallCS<W, C>,
    /// Accumulated left-hand side of batched equality constraint
    lhs: LinearCombination<C>,
    /// Accumulated right-hand side of batched equality constraint
    rhs: LinearCombination<C>,
    /// Number of bits used (i.e., how many constraints batched so far)
    bits_used: usize,
    /// Total number of batched operations flushed
    ops: usize,
}

impl<'a, W, C, const K: usize> BatchingSmallCS<'a, W, C, K>
where
    W: Witness,
    C: Coefficient + From<i32>,
{
    /// Create a new batching wrapper around an existing SmallCS.
    pub fn new(cs: &'a mut SmallCS<W, C>) -> Self {
        Self {
            cs,
            lhs: LinearCombination::zero(),
            rhs: LinearCombination::zero(),
            bits_used: 0,
            ops: 0,
        }
    }

    /// Batch an equality constraint: lhs = rhs
    ///
    /// Scales both sides by 2^bits_used and accumulates into the batch.
    /// When K constraints are accumulated, automatically flushes to the
    /// underlying CS.
    ///
    /// # Coefficient Scaling
    ///
    /// If original coefficients are bounded by M, then after batching
    /// they are bounded by M × 2^(K-1). For K=17 and M=2^14, this is
    /// 2^14 × 2^16 = 2^30, which fits in i32.
    pub fn enforce_equal(&mut self, lhs: &LinearCombination<C>, rhs: &LinearCombination<C>) {
        // Flush if we've reached capacity
        if self.bits_used >= K {
            self.flush();
        }

        // Scale both sides by 2^bits_used
        let scale = C::from(1i32 << self.bits_used);
        self.lhs = self.lhs.clone() + lhs.scale(scale);
        self.rhs = self.rhs.clone() + rhs.scale(scale);
        self.bits_used += 1;
    }

    /// Flush accumulated constraints to underlying CS.
    ///
    /// Emits a single R1CS constraint: lhs × 1 = rhs
    pub fn flush(&mut self) {
        if self.bits_used > 0 {
            // Clone the LCs before moving into closures
            let lhs = std::mem::take(&mut self.lhs);
            let rhs = std::mem::take(&mut self.rhs);

            self.cs.enforce(|_| lhs, |lc| lc + Self::one(), |_| rhs);
            self.ops += 1;
        }
        self.lhs = LinearCombination::zero();
        self.rhs = LinearCombination::zero();
        self.bits_used = 0;
    }

    /// Get the number of batched operations flushed so far.
    pub fn num_ops(&self) -> usize {
        self.ops
    }

    /// Get the number of constraints pending in the current batch.
    pub fn pending(&self) -> usize {
        self.bits_used
    }
}

// Implement SmallConstraintSystem by delegating to underlying CS
impl<W, C, const K: usize> SmallConstraintSystem<W, C> for BatchingSmallCS<'_, W, C, K>
where
    W: Witness,
    C: Coefficient + From<i32>,
{
    fn alloc<F>(&mut self, f: F) -> Result<Variable, SynthesisError>
    where
        F: FnOnce() -> W,
    {
        self.cs.alloc(f)
    }

    fn alloc_input<F>(&mut self, f: F) -> Result<Variable, SynthesisError>
    where
        F: FnOnce() -> W,
    {
        self.cs.alloc_input(f)
    }

    fn enforce<FA, FB, FC>(&mut self, a: FA, b: FB, c: FC)
    where
        FA: FnOnce(LinearCombination<C>) -> LinearCombination<C>,
        FB: FnOnce(LinearCombination<C>) -> LinearCombination<C>,
        FC: FnOnce(LinearCombination<C>) -> LinearCombination<C>,
    {
        // For general constraints (not equality), delegate directly
        self.cs.enforce(a, b, c)
    }

    fn push_namespace<N: Into<String>>(&mut self, name: N) {
        self.cs.push_namespace(name)
    }

    fn pop_namespace(&mut self) {
        self.cs.pop_namespace()
    }

    fn num_constraints(&self) -> usize {
        self.cs.num_constraints()
    }

    fn num_aux(&self) -> usize {
        self.cs.num_aux()
    }

    fn num_inputs(&self) -> usize {
        self.cs.num_inputs()
    }
}

impl<W, C, const K: usize> Drop for BatchingSmallCS<'_, W, C, K>
where
    W: Witness,
    C: Coefficient + From<i32>,
{
    fn drop(&mut self) {
        self.flush();
    }
}

/// SmallMultiEqCS implementation for BatchingSmallCS (batched).
///
/// Equality constraints are batched together for reduced constraint count.
impl<W, C, const K: usize> SmallMultiEqCS<W, C> for BatchingSmallCS<'_, W, C, K>
where
    W: Witness,
    C: Coefficient + From<i32>,
{
    fn enforce_equal(&mut self, lhs: &LinearCombination<C>, rhs: &LinearCombination<C>) {
        // Use the existing enforce_equal method
        BatchingSmallCS::enforce_equal(self, lhs, rhs)
    }

    fn flush(&mut self) {
        // Use the existing flush method
        BatchingSmallCS::flush(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batching_basic() {
        let mut cs = SmallCS::<i32, i32>::new();

        // Create some variables
        let x = cs.alloc(|| 5).unwrap();
        let y = cs.alloc(|| 5).unwrap();
        let a = cs.alloc(|| 10).unwrap();
        let b = cs.alloc(|| 10).unwrap();

        {
            let mut batched = BatchingSmallCS::<i32, i32, 17>::new(&mut cs);

            // Two equality constraints: x = y and a = b
            let lhs1 = LinearCombination::from_variable(x);
            let rhs1 = LinearCombination::from_variable(y);
            batched.enforce_equal(&lhs1, &rhs1);

            let lhs2 = LinearCombination::from_variable(a);
            let rhs2 = LinearCombination::from_variable(b);
            batched.enforce_equal(&lhs2, &rhs2);

            assert_eq!(batched.pending(), 2);
        } // Drop flushes

        // Should have 1 batched constraint
        assert_eq!(cs.num_constraints(), 1);
        assert!(cs.is_satisfied::<i64>());
    }

    #[test]
    fn test_batching_capacity() {
        let mut cs = SmallCS::<i32, i32>::new();

        // Allocate pairs of equal variables
        let mut vars = Vec::new();
        for i in 0..20 {
            vars.push(cs.alloc(|| i).unwrap());
            vars.push(cs.alloc(|| i).unwrap()); // Equal value
        }

        {
            let mut batched = BatchingSmallCS::<i32, i32, 4>::new(&mut cs);

            // Add 20 equality constraints with K=4
            // Should produce ceil(20/4) = 5 actual constraints
            for chunk in vars.chunks(2) {
                let lhs = LinearCombination::from_variable(chunk[0]);
                let rhs = LinearCombination::from_variable(chunk[1]);
                batched.enforce_equal(&lhs, &rhs);
            }
        }

        // 20 constraints batched with K=4 → 5 actual constraints
        assert_eq!(cs.num_constraints(), 5);
        assert!(cs.is_satisfied::<i64>());
    }

    #[test]
    fn test_batching_unsatisfied() {
        let mut cs = SmallCS::<i32, i32>::new();

        let x = cs.alloc(|| 5).unwrap();
        let y = cs.alloc(|| 6).unwrap(); // Different!

        {
            let mut batched = BatchingSmallCS::<i32, i32, 17>::new(&mut cs);
            let lhs = LinearCombination::from_variable(x);
            let rhs = LinearCombination::from_variable(y);
            batched.enforce_equal(&lhs, &rhs);
        }

        assert!(!cs.is_satisfied::<i64>());
    }

    #[test]
    fn test_batching_with_coefficients() {
        let mut cs = SmallCS::<i32, i32>::new();

        // 2x + 3 = 13 where x = 5
        let x = cs.alloc(|| 5).unwrap();
        let result = cs.alloc(|| 13).unwrap();

        {
            let mut batched = BatchingSmallCS::<i32, i32, 17>::new(&mut cs);

            // 2x + 3 = result
            let lhs = LinearCombination::zero()
                .add_term(2, x)
                .add_term(3, SmallCS::<i32, i32>::one());
            let rhs = LinearCombination::from_variable(result);
            batched.enforce_equal(&lhs, &rhs);
        }

        assert!(cs.is_satisfied::<i64>());
    }

    #[test]
    fn test_scaling_bounds() {
        // Verify coefficient bounds with K=17 and max coeff 2^14
        let max_coeff: i32 = 1 << 14; // 2^14 = 16384
        let max_scale: i32 = 1 << 16; // 2^16 = 65536 (K-1 = 16)

        // Combined should fit in i32
        let combined = max_coeff.checked_mul(max_scale);
        assert!(combined.is_some());
        assert!(combined.unwrap() > 0); // No overflow

        // 2^14 × 2^16 = 2^30 = 1073741824
        assert_eq!(combined.unwrap(), 1 << 30);
    }
}
