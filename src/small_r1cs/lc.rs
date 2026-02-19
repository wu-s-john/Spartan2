//! Linear combination with generic coefficient type.
//!
//! This module defines `LinearCombination<C>` which stores coefficients of type C
//! (e.g., i32) instead of field elements. We reuse Bellpepper's `Variable` type
//! since it's just an index with no field dependency.

use super::traits::{Accumulator, Coefficient, WideningMul, Witness};
use bellpepper_core::{Index, Variable};
use std::ops::{Add, Sub};

/// A linear combination with coefficients of type C.
///
/// Represents: LC = Σ cᵢ × varᵢ
///
/// For SHA-256 with C=i32, each term is a 4-byte coefficient paired with a variable.
/// When evaluated against witnesses W=i32, produces Acc=i64.
#[derive(Clone, Debug, Default)]
pub struct LinearCombination<C: Coefficient> {
    pub terms: Vec<(Variable, C)>,
}

impl<C: Coefficient> LinearCombination<C> {
    /// Create an empty linear combination (zero).
    pub fn zero() -> Self {
        Self { terms: vec![] }
    }

    /// Create a linear combination representing the constant 1.
    pub fn one() -> Self {
        Self {
            terms: vec![(Variable::new_unchecked(Index::Input(0)), C::one())],
        }
    }

    /// Create a linear combination from a single variable with coefficient 1.
    pub fn from_variable(var: Variable) -> Self {
        Self {
            terms: vec![(var, C::one())],
        }
    }

    /// Create a linear combination from a single variable with given coefficient.
    pub fn from_variable_scaled(var: Variable, coeff: C) -> Self {
        Self {
            terms: vec![(var, coeff)],
        }
    }

    /// Add a term with coefficient 1.
    pub fn add_variable(mut self, var: Variable) -> Self {
        self.terms.push((var, C::one()));
        self
    }

    /// Subtract a variable (add with coefficient -1).
    pub fn sub_variable(mut self, var: Variable) -> Self {
        self.terms.push((var, -C::one()));
        self
    }

    /// Add a term with given coefficient.
    pub fn add_term(mut self, coeff: C, var: Variable) -> Self {
        self.terms.push((var, coeff));
        self
    }

    /// Evaluate the LC against witness values, producing an accumulator value.
    ///
    /// The witness vector layout is: z = [aux | 1 | inputs[1..]]
    /// - `one_value`: The witness value for the constant ONE (typically W::one())
    /// - `inputs`: Public input values (inputs[0] is the implicit ONE)
    /// - `aux`: Auxiliary (private) witness values
    pub fn evaluate<W, Acc>(&self, one_value: W, inputs: &[W], aux: &[W]) -> Acc
    where
        W: Witness,
        C: WideningMul<W, Acc>,
        Acc: Accumulator,
    {
        self.terms.iter().fold(Acc::zero(), |acc, (var, coeff)| {
            let w: W = match var.get_unchecked() {
                Index::Input(0) => one_value,
                Index::Input(i) => inputs[i],
                Index::Aux(i) => aux[i],
            };
            acc + coeff.wide_mul(w)
        })
    }

    /// Number of terms in the linear combination.
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    /// Check if the linear combination is empty (zero).
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    /// Scale all coefficients by a factor.
    ///
    /// Returns a new LC where each term's coefficient is multiplied by `factor`.
    /// Used by BatchingSmallCS to scale LCs before combining them.
    pub fn scale(&self, factor: C) -> Self {
        LinearCombination {
            terms: self
                .terms
                .iter()
                .map(|(var, coeff)| (*var, *coeff * factor))
                .collect(),
        }
    }
}

// Operator overloads for ergonomic constraint building

impl<C: Coefficient> Add<Variable> for LinearCombination<C> {
    type Output = Self;
    fn add(self, var: Variable) -> Self {
        self.add_variable(var)
    }
}

impl<C: Coefficient> Sub<Variable> for LinearCombination<C> {
    type Output = Self;
    fn sub(self, var: Variable) -> Self {
        self.sub_variable(var)
    }
}

impl<C: Coefficient> Add<(C, Variable)> for LinearCombination<C> {
    type Output = Self;
    fn add(self, (coeff, var): (C, Variable)) -> Self {
        self.add_term(coeff, var)
    }
}

impl<C: Coefficient> Sub<(C, Variable)> for LinearCombination<C> {
    type Output = Self;
    fn sub(self, (coeff, var): (C, Variable)) -> Self {
        self.add_term(-coeff, var)
    }
}

// Add two linear combinations
impl<C: Coefficient> Add<LinearCombination<C>> for LinearCombination<C> {
    type Output = Self;
    fn add(mut self, other: Self) -> Self {
        self.terms.extend(other.terms);
        self
    }
}

// Subtract two linear combinations
impl<C: Coefficient> Sub<LinearCombination<C>> for LinearCombination<C> {
    type Output = Self;
    fn sub(mut self, other: Self) -> Self {
        for (var, coeff) in other.terms {
            self.terms.push((var, -coeff));
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lc_evaluate() {
        // Create LC: 2*x + 3*y where x=5, y=7
        let x = Variable::new_unchecked(Index::Aux(0));
        let y = Variable::new_unchecked(Index::Aux(1));

        let lc: LinearCombination<i32> =
            LinearCombination::zero().add_term(2, x).add_term(3, y);

        let inputs: Vec<i32> = vec![1]; // Just the ONE
        let aux: Vec<i32> = vec![5, 7]; // x=5, y=7

        let result: i64 = lc.evaluate(1, &inputs, &aux);
        assert_eq!(result, 2 * 5 + 3 * 7); // 10 + 21 = 31
    }

    #[test]
    fn test_lc_with_one() {
        // Create LC: x + 5 (using the ONE variable)
        let x = Variable::new_unchecked(Index::Aux(0));
        let one = Variable::new_unchecked(Index::Input(0));

        let lc: LinearCombination<i32> =
            LinearCombination::zero().add_variable(x).add_term(5, one);

        let inputs: Vec<i32> = vec![1];
        let aux: Vec<i32> = vec![10]; // x=10

        let result: i64 = lc.evaluate(1, &inputs, &aux);
        assert_eq!(result, 10 + 5); // 15
    }

    #[test]
    fn test_lc_operators() {
        let x = Variable::new_unchecked(Index::Aux(0));
        let y = Variable::new_unchecked(Index::Aux(1));

        // Using operator syntax: lc + x + (2, y)
        let lc: LinearCombination<i32> = LinearCombination::zero() + x + (2i32, y);

        assert_eq!(lc.len(), 2);
    }

    #[test]
    fn test_lc_subtraction() {
        let x = Variable::new_unchecked(Index::Aux(0));

        let lc: LinearCombination<i32> = LinearCombination::zero() + x - x;

        let inputs: Vec<i32> = vec![1];
        let aux: Vec<i32> = vec![42];

        let result: i64 = lc.evaluate(1, &inputs, &aux);
        assert_eq!(result, 0); // x - x = 0
    }
}
