//! Concrete implementation of SmallConstraintSystem.
//!
//! `SmallCS<W, C>` is a constraint system that stores witnesses of type W (e.g., i32)
//! and coefficients of type C (e.g., i32), avoiding field arithmetic during synthesis.

use super::{
    cs::{SmallConstraintSystem, SynthesisError},
    lc::LinearCombination,
    sparse::SmallSparseMatrix,
    traits::{Accumulator, Coefficient, SmallMultiEqCS, WideningMul, Witness},
};
use bellpepper_core::{Index, Variable};

/// A constraint system implementation with W (i32) witnesses and C (i32) coefficients.
///
/// This stores all constraints and witnesses using native integer types,
/// deferring field conversion until commitment time.
#[derive(Clone, Debug)]
pub struct SmallCS<W: Witness, C: Coefficient> {
    /// Public inputs (includes implicit "1" at index 0).
    pub inputs: Vec<W>,
    /// Private auxiliary witnesses.
    pub aux: Vec<W>,
    /// Constraints: (A, B, C) where A × B = C.
    pub constraints: Vec<(LinearCombination<C>, LinearCombination<C>, LinearCombination<C>)>,
    /// Namespace stack for debugging.
    namespace: Vec<String>,
}

impl<W: Witness, C: Coefficient> Default for SmallCS<W, C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<W: Witness, C: Coefficient> SmallCS<W, C> {
    /// Create a new constraint system.
    ///
    /// The first input is always the constant 1.
    pub fn new() -> Self {
        Self {
            inputs: vec![W::one()], // First input is always 1
            aux: vec![],
            constraints: vec![],
            namespace: vec![],
        }
    }

    /// Build the full witness vector z = [aux | 1 | inputs[1..]].
    ///
    /// This is the standard R1CS witness layout.
    pub fn z(&self) -> Vec<W> {
        let mut z = self.aux.clone();
        z.push(W::one());
        z.extend(self.inputs[1..].iter().cloned());
        z
    }

    /// Check constraint satisfaction using accumulator type Acc.
    ///
    /// For each constraint (A, B, C): verify that A·z × B·z = C·z
    /// Uses i128 for the final product to avoid overflow.
    pub fn is_satisfied<Acc>(&self) -> bool
    where
        C: WideningMul<W, Acc>,
        Acc: Accumulator + Into<i128>,
    {
        let one = W::one();
        for (a_lc, b_lc, c_lc) in &self.constraints {
            let a: Acc = a_lc.evaluate(one, &self.inputs, &self.aux);
            let b: Acc = b_lc.evaluate(one, &self.inputs, &self.aux);
            let c: Acc = c_lc.evaluate(one, &self.inputs, &self.aux);

            // Use i128 for product to avoid overflow
            let a128: i128 = a.into();
            let b128: i128 = b.into();
            let c128: i128 = c.into();

            if a128 * b128 != c128 {
                return false;
            }
        }
        true
    }

    /// Find the first unsatisfied constraint (for debugging).
    pub fn which_is_unsatisfied<Acc>(&self) -> Option<usize>
    where
        C: WideningMul<W, Acc>,
        Acc: Accumulator + Into<i128>,
    {
        let one = W::one();
        for (i, (a_lc, b_lc, c_lc)) in self.constraints.iter().enumerate() {
            let a: Acc = a_lc.evaluate(one, &self.inputs, &self.aux);
            let b: Acc = b_lc.evaluate(one, &self.inputs, &self.aux);
            let c: Acc = c_lc.evaluate(one, &self.inputs, &self.aux);

            let a128: i128 = a.into();
            let b128: i128 = b.into();
            let c128: i128 = c.into();

            if a128 * b128 != c128 {
                return Some(i);
            }
        }
        None
    }

    /// Build sparse matrices from constraints.
    ///
    /// Returns (A, B, C) matrices in CSR format.
    pub fn build_matrices(&self) -> (SmallSparseMatrix<C>, SmallSparseMatrix<C>, SmallSparseMatrix<C>) {
        let num_cols = self.aux.len() + 1 + self.inputs.len() - 1; // aux + 1 + public_inputs

        let mut a_data = Vec::new();
        let mut a_indices = Vec::new();
        let mut a_indptr = vec![0usize];

        let mut b_data = Vec::new();
        let mut b_indices = Vec::new();
        let mut b_indptr = vec![0usize];

        let mut c_data = Vec::new();
        let mut c_indices = Vec::new();
        let mut c_indptr = vec![0usize];

        for (a_lc, b_lc, c_lc) in &self.constraints {
            // Process A
            for (var, coeff) in &a_lc.terms {
                let col = self.var_to_col(*var);
                a_data.push(*coeff);
                a_indices.push(col);
            }
            a_indptr.push(a_data.len());

            // Process B
            for (var, coeff) in &b_lc.terms {
                let col = self.var_to_col(*var);
                b_data.push(*coeff);
                b_indices.push(col);
            }
            b_indptr.push(b_data.len());

            // Process C
            for (var, coeff) in &c_lc.terms {
                let col = self.var_to_col(*var);
                c_data.push(*coeff);
                c_indices.push(col);
            }
            c_indptr.push(c_data.len());
        }

        (
            SmallSparseMatrix {
                data: a_data,
                indices: a_indices,
                indptr: a_indptr,
                cols: num_cols,
            },
            SmallSparseMatrix {
                data: b_data,
                indices: b_indices,
                indptr: b_indptr,
                cols: num_cols,
            },
            SmallSparseMatrix {
                data: c_data,
                indices: c_indices,
                indptr: c_indptr,
                cols: num_cols,
            },
        )
    }

    /// Convert a variable to its column index in the z vector.
    ///
    /// z = [aux | 1 | inputs[1..]]
    fn var_to_col(&self, var: Variable) -> usize {
        match var.get_unchecked() {
            Index::Aux(i) => i,
            Index::Input(0) => self.aux.len(), // The ONE
            Index::Input(i) => self.aux.len() + i,
        }
    }
}

impl<W: Witness, C: Coefficient> SmallConstraintSystem<W, C> for SmallCS<W, C> {
    fn alloc<F>(&mut self, f: F) -> Result<Variable, SynthesisError>
    where
        F: FnOnce() -> W,
    {
        let value = f();
        self.aux.push(value);
        Ok(Variable::new_unchecked(Index::Aux(self.aux.len() - 1)))
    }

    fn alloc_input<F>(&mut self, f: F) -> Result<Variable, SynthesisError>
    where
        F: FnOnce() -> W,
    {
        let value = f();
        self.inputs.push(value);
        Ok(Variable::new_unchecked(Index::Input(self.inputs.len() - 1)))
    }

    fn enforce<FA, FB, FC>(&mut self, a: FA, b: FB, c: FC)
    where
        FA: FnOnce(LinearCombination<C>) -> LinearCombination<C>,
        FB: FnOnce(LinearCombination<C>) -> LinearCombination<C>,
        FC: FnOnce(LinearCombination<C>) -> LinearCombination<C>,
    {
        let a_lc = a(LinearCombination::zero());
        let b_lc = b(LinearCombination::zero());
        let c_lc = c(LinearCombination::zero());
        self.constraints.push((a_lc, b_lc, c_lc));
    }

    fn push_namespace<N: Into<String>>(&mut self, name: N) {
        self.namespace.push(name.into());
    }

    fn pop_namespace(&mut self) {
        self.namespace.pop();
    }

    fn num_constraints(&self) -> usize {
        self.constraints.len()
    }

    fn num_aux(&self) -> usize {
        self.aux.len()
    }

    fn num_inputs(&self) -> usize {
        self.inputs.len()
    }
}

/// SmallMultiEqCS implementation for SmallCS (non-batched).
///
/// Each equality constraint is directly enforced as `lhs × 1 = rhs`.
impl<W: Witness, C: Coefficient> SmallMultiEqCS<W, C> for SmallCS<W, C> {
    fn enforce_equal(&mut self, lhs: &LinearCombination<C>, rhs: &LinearCombination<C>) {
        // Clone the LCs so we can move them into closures
        let lhs = lhs.clone();
        let rhs = rhs.clone();
        self.enforce(|_| lhs, |lc| lc + Self::one(), |_| rhs);
    }

    fn flush(&mut self) {
        // No-op for non-batching implementation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mul_circuit() {
        // Simple circuit: x * y = z where x=3, y=4, z=12
        let mut cs = SmallCS::<i32, i32>::new();

        let x = cs.alloc(|| 3).unwrap();
        let y = cs.alloc(|| 4).unwrap();
        let z = cs.alloc(|| 12).unwrap();

        // Constraint: x * y = z
        cs.enforce(|lc| lc + x, |lc| lc + y, |lc| lc + z);

        assert!(cs.is_satisfied::<i64>());
        assert_eq!(cs.num_constraints(), 1);
        assert_eq!(cs.num_aux(), 3);
    }

    #[test]
    fn test_unsatisfied_circuit() {
        let mut cs = SmallCS::<i32, i32>::new();

        let x = cs.alloc(|| 3).unwrap();
        let y = cs.alloc(|| 4).unwrap();
        let z = cs.alloc(|| 11).unwrap(); // Wrong! Should be 12

        cs.enforce(|lc| lc + x, |lc| lc + y, |lc| lc + z);

        assert!(!cs.is_satisfied::<i64>());
        assert_eq!(cs.which_is_unsatisfied::<i64>(), Some(0));
    }

    #[test]
    fn test_linear_combination_constraint() {
        // Circuit: (2x + 3) * 1 = y where x=5, y=13
        let mut cs = SmallCS::<i32, i32>::new();

        let x = cs.alloc(|| 5).unwrap();
        let y = cs.alloc(|| 13).unwrap();

        // Constraint: (2x + 3) * 1 = y
        cs.enforce(
            |lc| lc + (2, x) + (3, SmallCS::<i32, i32>::one()),
            |lc| lc + SmallCS::<i32, i32>::one(),
            |lc| lc + y,
        );

        assert!(cs.is_satisfied::<i64>());
    }

    #[test]
    fn test_build_matrices() {
        let mut cs = SmallCS::<i32, i32>::new();

        let x = cs.alloc(|| 3).unwrap();
        let y = cs.alloc(|| 4).unwrap();
        let z = cs.alloc(|| 12).unwrap();

        cs.enforce(|lc| lc + x, |lc| lc + y, |lc| lc + z);

        let (a, b, c) = cs.build_matrices();

        // Verify matrix dimensions
        assert_eq!(a.indptr.len(), 2); // 1 constraint + 1
        assert_eq!(b.indptr.len(), 2);
        assert_eq!(c.indptr.len(), 2);

        // Verify z vector
        let z_vec = cs.z();
        assert_eq!(z_vec, vec![3, 4, 12, 1]); // [aux | 1]
    }
}
