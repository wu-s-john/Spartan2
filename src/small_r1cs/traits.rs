//! Core traits for the small-value R1CS system.
//!
//! These traits abstract over the coefficient, witness, and accumulator types,
//! allowing the constraint system to work with native integers (i32, i64)
//! instead of field elements.

use super::lc::LinearCombination;
use num_traits::{One, Zero};
use std::fmt::Debug;
use std::ops::{Add, Mul, Neg, Sub};

/// Trait for matrix coefficients (C = i32 for SHA-256).
///
/// These are the values stored in the A, B, C matrices of the R1CS system.
/// For SHA-256, coefficients are bounded by 2^16 (from 2-limb addition).
pub trait Coefficient:
    Copy
    + Clone
    + Default
    + Send
    + Sync
    + Debug
    + PartialEq
    + Eq
    + Zero
    + One
    + Neg<Output = Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
{
}

// Blanket implementation for all types satisfying the bounds
impl<T> Coefficient for T where
    T: Copy
        + Clone
        + Default
        + Send
        + Sync
        + Debug
        + PartialEq
        + Eq
        + Zero
        + One
        + Neg<Output = Self>
        + Add<Output = Self>
        + Sub<Output = Self>
        + Mul<Output = Self>
{
}

/// Trait for witness values (W = i32 for SHA-256).
///
/// These are the values assigned to variables during synthesis.
/// For SHA-256, witnesses are bits (0 or 1) or small integers.
pub trait Witness:
    Copy
    + Clone
    + Default
    + Send
    + Sync
    + Debug
    + Zero
    + One
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
{
}

impl<T> Witness for T where
    T: Copy
        + Clone
        + Default
        + Send
        + Sync
        + Debug
        + Zero
        + One
        + Add<Output = Self>
        + Sub<Output = Self>
        + Mul<Output = Self>
        + Neg<Output = Self>
{
}

/// Trait for accumulator type (Acc = i64 for SHA-256).
///
/// Used for LC evaluation: Σ C×W → Acc.
/// The accumulator must be wide enough to hold the sum of products
/// without overflow.
pub trait Accumulator:
    Copy
    + Clone
    + Default
    + Send
    + Sync
    + Debug
    + Zero
    + One
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
{
}

impl<T> Accumulator for T where
    T: Copy
        + Clone
        + Default
        + Send
        + Sync
        + Debug
        + Zero
        + One
        + Add<Output = Self>
        + Sub<Output = Self>
        + Mul<Output = Self>
        + Neg<Output = Self>
{
}

/// Trait for widening multiplication: C × W → Acc.
///
/// This is used when evaluating linear combinations, where we need
/// to multiply a coefficient by a witness and accumulate into a wider type.
pub trait WideningMul<W, Acc> {
    /// Multiply self by w, widening to type Acc.
    fn wide_mul(self, w: W) -> Acc;
}

// Implementation for i32 × i32 → i64
impl WideningMul<i32, i64> for i32 {
    #[inline]
    fn wide_mul(self, w: i32) -> i64 {
        (self as i64) * (w as i64)
    }
}

// Implementation for i64 × i64 → i128
impl WideningMul<i64, i128> for i64 {
    #[inline]
    fn wide_mul(self, w: i64) -> i128 {
        (self as i128) * (w as i128)
    }
}

// Implementation for i16 × i16 → i32
impl WideningMul<i16, i32> for i16 {
    #[inline]
    fn wide_mul(self, w: i16) -> i32 {
        (self as i32) * (w as i32)
    }
}

// Identity implementations (when C = W = Acc)
impl WideningMul<i32, i32> for i32 {
    #[inline]
    fn wide_mul(self, w: i32) -> i32 {
        self * w
    }
}

impl WideningMul<i64, i64> for i64 {
    #[inline]
    fn wide_mul(self, w: i64) -> i64 {
        self * w
    }
}

// Implementation for i32 × i64 → i64 (small coefficient × wider witness)
impl WideningMul<i64, i64> for i32 {
    #[inline]
    fn wide_mul(self, w: i64) -> i64 {
        (self as i64) * w
    }
}

// Implementation for i64 × i32 → i64 (wider coefficient × small witness)
impl WideningMul<i32, i64> for i64 {
    #[inline]
    fn wide_mul(self, w: i32) -> i64 {
        self * (w as i64)
    }
}

/// Trait for constraint systems that support batched equality constraints.
///
/// This trait extends the basic constraint system operations with the ability
/// to enforce equality constraints (`lhs = rhs`) that may be batched together
/// for reduced constraint count.
///
/// # Implementations
///
/// - `SmallCS<W, C>`: Direct enforcement (no batching)
/// - `BatchingSmallCS<W, C, K>`: Batches up to K constraints into one
///
/// # Example
///
/// ```ignore
/// fn my_gadget<W, C, CS>(cs: &mut CS, a: Variable, b: Variable)
/// where
///     CS: SmallMultiEqCS<W, C>,
/// {
///     let lhs = LinearCombination::from_variable(a);
///     let rhs = LinearCombination::from_variable(b);
///     cs.enforce_equal(&lhs, &rhs);  // May be batched
/// }
/// ```
pub trait SmallMultiEqCS<W: Witness, C: Coefficient> {
    /// Enforce an equality constraint: lhs = rhs.
    ///
    /// This may be batched with other equality constraints to reduce
    /// the total number of R1CS constraints.
    fn enforce_equal(&mut self, lhs: &LinearCombination<C>, rhs: &LinearCombination<C>);

    /// Flush any pending batched constraints.
    ///
    /// For non-batching implementations, this is a no-op.
    fn flush(&mut self);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wide_mul_i32_i64() {
        let c: i32 = 65536; // 2^16
        let w: i32 = 65536;
        let acc: i64 = c.wide_mul(w);
        assert_eq!(acc, 4294967296); // 2^32
    }

    #[test]
    fn test_wide_mul_negative() {
        let c: i32 = -100;
        let w: i32 = 50;
        let acc: i64 = c.wide_mul(w);
        assert_eq!(acc, -5000);
    }

    #[test]
    fn test_trait_bounds() {
        fn accepts_coefficient<C: Coefficient>(_c: C) {}
        fn accepts_witness<W: Witness>(_w: W) {}
        fn accepts_accumulator<A: Accumulator>(_a: A) {}

        accepts_coefficient(0i32);
        accepts_witness(0i32);
        accepts_accumulator(0i64);
    }
}
