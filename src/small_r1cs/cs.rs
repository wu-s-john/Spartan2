//! Constraint system trait for small-value R1CS.
//!
//! This defines the `SmallConstraintSystem` trait which is our own
//! constraint system interface using generic witness (W) and coefficient (C) types.

use super::{
    lc::LinearCombination,
    traits::{Coefficient, Witness},
};
use bellpepper_core::{Index, Variable};

/// Error type for constraint system operations.
#[derive(Debug, Clone)]
pub enum SynthesisError {
    /// Variable assignment is missing
    AssignmentMissing,
    /// Division by zero
    DivisionByZero,
    /// Unsatisfied constraint
    Unsatisfied,
    /// Invalid constraint
    InvalidConstraint(String),
}

impl std::fmt::Display for SynthesisError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SynthesisError::AssignmentMissing => write!(f, "assignment missing"),
            SynthesisError::DivisionByZero => write!(f, "division by zero"),
            SynthesisError::Unsatisfied => write!(f, "unsatisfied constraint"),
            SynthesisError::InvalidConstraint(msg) => write!(f, "invalid constraint: {}", msg),
        }
    }
}

impl std::error::Error for SynthesisError {}

/// A constraint system with witness type W and coefficient type C.
///
/// This is our own trait, independent of Bellpepper's field-based ConstraintSystem.
/// It allows working with native integer types during synthesis.
pub trait SmallConstraintSystem<W: Witness, C: Coefficient>: Sized {
    /// Returns the "one" variable (constant 1).
    fn one() -> Variable {
        Variable::new_unchecked(Index::Input(0))
    }

    /// Allocate a private auxiliary variable.
    ///
    /// The closure `f` provides the witness value to assign.
    fn alloc<F>(&mut self, f: F) -> Result<Variable, SynthesisError>
    where
        F: FnOnce() -> W;

    /// Allocate a public input variable.
    ///
    /// The closure `f` provides the witness value to assign.
    fn alloc_input<F>(&mut self, f: F) -> Result<Variable, SynthesisError>
    where
        F: FnOnce() -> W;

    /// Enforce a constraint: A × B = C.
    ///
    /// Each of `a`, `b`, `c` is a closure that builds a linear combination
    /// from a zero starting point.
    fn enforce<FA, FB, FC>(&mut self, a: FA, b: FB, c: FC)
    where
        FA: FnOnce(LinearCombination<C>) -> LinearCombination<C>,
        FB: FnOnce(LinearCombination<C>) -> LinearCombination<C>,
        FC: FnOnce(LinearCombination<C>) -> LinearCombination<C>;

    /// Push a namespace for debugging/organization.
    fn push_namespace<N: Into<String>>(&mut self, _name: N) {
        // Default implementation does nothing
    }

    /// Pop the current namespace.
    fn pop_namespace(&mut self) {
        // Default implementation does nothing
    }

    /// Get the number of constraints.
    fn num_constraints(&self) -> usize;

    /// Get the number of auxiliary variables.
    fn num_aux(&self) -> usize;

    /// Get the number of input variables (including the implicit ONE).
    fn num_inputs(&self) -> usize;
}
