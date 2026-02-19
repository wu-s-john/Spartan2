//! Boolean gadget for small-value R1CS.
//!
//! `Boolean<W, C>` represents a boolean value in the constraint system,
//! supporting XOR, AND, and NOT operations.

use crate::small_r1cs::{
    Coefficient, LinearCombination, SmallConstraintSystem, SynthesisError, Witness,
};
use bellpepper_core::Variable;
use std::marker::PhantomData;

/// A boolean value in the constraint system.
///
/// Can be either a constant (no variable) or an allocated variable.
/// Supports XOR, AND, and NOT operations.
#[derive(Clone, Debug)]
pub struct Boolean<W: Witness, C: Coefficient> {
    /// The variable, if allocated. None means this is a constant.
    pub var: Option<Variable>,
    /// The known value, if any.
    pub value: Option<bool>,
    /// Whether this is negated (for NOT without allocating).
    negated: bool,
    _phantom: PhantomData<(W, C)>,
}

impl<W: Witness, C: Coefficient> Boolean<W, C> {
    /// Create a constant boolean (no constraint needed).
    pub fn constant(b: bool) -> Self {
        Boolean {
            var: None,
            value: Some(b),
            negated: false,
            _phantom: PhantomData,
        }
    }

    /// Allocate a boolean variable and constrain it to be 0 or 1.
    ///
    /// Constraint: b × b = b (equivalent to b × (1-b) = 0)
    pub fn alloc<CS: SmallConstraintSystem<W, C>>(
        cs: &mut CS,
        value: Option<bool>,
    ) -> Result<Self, SynthesisError> {
        let var = cs.alloc(|| {
            if value.unwrap_or(false) {
                W::one()
            } else {
                W::zero()
            }
        })?;

        // Enforce boolean constraint: b × b = b
        // This ensures b ∈ {0, 1}
        cs.enforce(
            |lc| lc + var,
            |lc| lc + var,
            |lc| lc + var,
        );

        Ok(Boolean {
            var: Some(var),
            value,
            negated: false,
            _phantom: PhantomData,
        })
    }

    /// Allocate a boolean without the boolean constraint.
    ///
    /// Use this when you know the value is boolean from other constraints.
    pub fn alloc_unconstrained<CS: SmallConstraintSystem<W, C>>(
        cs: &mut CS,
        value: Option<bool>,
    ) -> Result<Self, SynthesisError> {
        let var = cs.alloc(|| {
            if value.unwrap_or(false) {
                W::one()
            } else {
                W::zero()
            }
        })?;

        Ok(Boolean {
            var: Some(var),
            value,
            negated: false,
            _phantom: PhantomData,
        })
    }

    /// Get the effective value (accounting for negation).
    pub fn get_value(&self) -> Option<bool> {
        self.value.map(|v| if self.negated { !v } else { v })
    }

    /// Check if this boolean is negated.
    pub fn is_negated(&self) -> bool {
        self.negated
    }

    /// Negate the boolean (NOT operation).
    ///
    /// This is free - no new constraint or variable needed.
    pub fn not(&self) -> Self {
        Boolean {
            var: self.var,
            value: self.value,
            negated: !self.negated,
            _phantom: PhantomData,
        }
    }

    /// XOR of two booleans.
    ///
    /// XOR(a, b) = a + b - 2ab
    /// Constraint: 2a × b = a + b - c
    ///
    /// Special cases:
    /// - constant XOR constant: no constraint
    /// - constant XOR variable: just negate if constant is true
    pub fn xor<CS: SmallConstraintSystem<W, C>>(
        cs: &mut CS,
        a: &Self,
        b: &Self,
    ) -> Result<Self, SynthesisError> {
        // Get effective values
        let a_val = a.get_value();
        let b_val = b.get_value();

        match (a.var, a.negated, b.var, b.negated) {
            // Both constants
            (None, _, None, _) => {
                let result = a_val.unwrap() ^ b_val.unwrap();
                Ok(Boolean::constant(result))
            }
            // a is constant 0 (or negated 1)
            (None, _, Some(_), _) if !a_val.unwrap() => Ok(b.clone()),
            // a is constant 1 (or negated 0)
            (None, _, Some(_), _) => Ok(b.not()),
            // b is constant
            (Some(_), _, None, _) if !b_val.unwrap() => Ok(a.clone()),
            (Some(_), _, None, _) => Ok(a.not()),
            // Both are variables
            (Some(a_var), a_neg, Some(b_var), b_neg) => {
                let result_val = match (a_val, b_val) {
                    (Some(a), Some(b)) => Some(a ^ b),
                    _ => None,
                };

                let c_var = cs.alloc(|| {
                    if result_val.unwrap_or(false) {
                        W::one()
                    } else {
                        W::zero()
                    }
                })?;

                // Build the constraint based on negation flags
                // XOR(a, b) = a + b - 2ab when neither is negated
                // XOR(NOT a, b) = (1-a) + b - 2(1-a)b = 1 - a + b - 2b + 2ab = 1 - a - b + 2ab
                // etc.

                let two = C::one() + C::one();
                let one_var = CS::one();

                match (a_neg, b_neg) {
                    (false, false) => {
                        // 2a × b = a + b - c
                        cs.enforce(
                            |lc| lc + (two, a_var),
                            |lc| lc + b_var,
                            |lc| lc + a_var + b_var - c_var,
                        );
                    }
                    (true, false) => {
                        // XOR(1-a, b) = 1 - a + b - 2b + 2ab = 1 - a - b + 2ab
                        // 2a × b = a + b - 1 + c
                        cs.enforce(
                            |lc| lc + (two, a_var),
                            |lc| lc + b_var,
                            |lc| lc + a_var + b_var - one_var + c_var,
                        );
                    }
                    (false, true) => {
                        // Same as (true, false) by symmetry
                        cs.enforce(
                            |lc| lc + (two, a_var),
                            |lc| lc + b_var,
                            |lc| lc + a_var + b_var - one_var + c_var,
                        );
                    }
                    (true, true) => {
                        // XOR(1-a, 1-b) = a XOR b
                        cs.enforce(
                            |lc| lc + (two, a_var),
                            |lc| lc + b_var,
                            |lc| lc + a_var + b_var - c_var,
                        );
                    }
                }

                Ok(Boolean {
                    var: Some(c_var),
                    value: result_val,
                    negated: false,
                    _phantom: PhantomData,
                })
            }
        }
    }

    /// AND of two booleans.
    ///
    /// AND(a, b) = a × b
    ///
    /// Special cases:
    /// - constant AND constant: no constraint
    /// - constant 0 AND variable: result is 0
    /// - constant 1 AND variable: result is variable
    pub fn and<CS: SmallConstraintSystem<W, C>>(
        cs: &mut CS,
        a: &Self,
        b: &Self,
    ) -> Result<Self, SynthesisError> {
        let a_val = a.get_value();
        let b_val = b.get_value();

        match (a.var, a.negated, b.var, b.negated) {
            // Both constants
            (None, _, None, _) => {
                let result = a_val.unwrap() && b_val.unwrap();
                Ok(Boolean::constant(result))
            }
            // a is constant false
            (None, _, _, _) if !a_val.unwrap() => Ok(Boolean::constant(false)),
            // a is constant true
            (None, _, Some(_), _) => Ok(b.clone()),
            // b is constant false
            (_, _, None, _) if !b_val.unwrap() => Ok(Boolean::constant(false)),
            // b is constant true
            (Some(_), _, None, _) => Ok(a.clone()),
            // Both are variables
            (Some(a_var), a_neg, Some(b_var), b_neg) => {
                let result_val = match (a_val, b_val) {
                    (Some(a), Some(b)) => Some(a && b),
                    _ => None,
                };

                let c_var = cs.alloc(|| {
                    if result_val.unwrap_or(false) {
                        W::one()
                    } else {
                        W::zero()
                    }
                })?;

                let one_var = CS::one();

                match (a_neg, b_neg) {
                    (false, false) => {
                        // a × b = c
                        cs.enforce(|lc| lc + a_var, |lc| lc + b_var, |lc| lc + c_var);
                    }
                    (true, false) => {
                        // (1-a) × b = c => b - ab = c => ab = b - c
                        cs.enforce(|lc| lc + a_var, |lc| lc + b_var, |lc| lc + b_var - c_var);
                    }
                    (false, true) => {
                        // a × (1-b) = c => a - ab = c => ab = a - c
                        cs.enforce(|lc| lc + a_var, |lc| lc + b_var, |lc| lc + a_var - c_var);
                    }
                    (true, true) => {
                        // (1-a) × (1-b) = c => 1 - a - b + ab = c => ab = a + b - 1 + c
                        cs.enforce(
                            |lc| lc + a_var,
                            |lc| lc + b_var,
                            |lc| lc + a_var + b_var - one_var + c_var,
                        );
                    }
                }

                Ok(Boolean {
                    var: Some(c_var),
                    value: result_val,
                    negated: false,
                    _phantom: PhantomData,
                })
            }
        }
    }

    /// Convert to a linear combination representing this boolean's value.
    ///
    /// If negated, returns (1 - var). Otherwise returns var.
    pub fn to_lc(&self) -> LinearCombination<C> {
        match (self.var, self.negated) {
            (None, false) if self.value == Some(true) => LinearCombination::one(),
            (None, false) => LinearCombination::zero(),
            (None, true) if self.value == Some(false) => LinearCombination::one(),
            (None, true) => LinearCombination::zero(),
            (Some(var), false) => LinearCombination::from_variable(var),
            (Some(var), true) => {
                // 1 - var
                let one = LinearCombination::<C>::one();
                one - var
            }
        }
    }

    /// SHA-256 Ch function: (a ∧ b) ⊕ (¬a ∧ c)
    ///
    /// Uses a single constraint: a × (b - c) = ch - c
    /// This is mathematically equivalent to: ch = a×b + c×(1-a)
    ///
    /// This is more efficient than computing XOR(AND(a,b), AND(NOT(a),c))
    /// which would require 3 constraints per bit.
    pub fn sha256_ch<CS: SmallConstraintSystem<W, C>>(
        cs: &mut CS,
        a: &Self,
        b: &Self,
        c: &Self,
    ) -> Result<Self, SynthesisError> {
        // Compute expected value
        let ch_val = match (a.get_value(), b.get_value(), c.get_value()) {
            (Some(a), Some(b), Some(c)) => Some((a && b) ^ ((!a) && c)),
            _ => None,
        };

        // Handle constant cases
        match (a.var, b.var, c.var) {
            // All constants
            (None, None, None) => {
                return Ok(Boolean::constant(ch_val.unwrap()));
            }
            // a is constant false: Ch(0,b,c) = c
            (None, _, _) if !a.get_value().unwrap() => {
                return Ok(c.clone());
            }
            // a is constant true: Ch(1,b,c) = b
            (None, _, _) => {
                return Ok(b.clone());
            }
            // b is constant false: Ch(a,0,c) = (NOT a) AND c
            (_, None, _) if !b.get_value().unwrap() => {
                return Self::and(cs, &a.not(), c);
            }
            // c is constant false: Ch(a,b,0) = a AND b
            (_, _, None) if !c.get_value().unwrap() => {
                return Self::and(cs, a, b);
            }
            // c is constant true: Ch(a,b,1) = NOT(a AND NOT(b))
            (_, _, None) => {
                return Ok(Self::and(cs, a, &b.not())?.not());
            }
            // b is constant true: Ch(a,1,c) = NOT(NOT(a) AND NOT(c))
            (_, None, _) => {
                return Ok(Self::and(cs, &a.not(), &c.not())?.not());
            }
            _ => {}
        }

        // All three are variables - use single-constraint formula
        // a × (b - c) = ch - c
        let ch_var = cs.alloc(|| {
            if ch_val.unwrap_or(false) {
                W::one()
            } else {
                W::zero()
            }
        })?;

        let (a_var, a_neg) = (a.var.unwrap(), a.negated);
        let (b_var, b_neg) = (b.var.unwrap(), b.negated);
        let (c_var, c_neg) = (c.var.unwrap(), c.negated);
        let one = CS::one();

        // Build linear combinations for a, b, c accounting for negation
        // If negated, use (1 - var) instead of var

        // For the formula: a × (b - c) = ch - c
        // We need to handle negation carefully

        match (a_neg, b_neg, c_neg) {
            (false, false, false) => {
                // a × (b - c) = ch - c
                cs.enforce(
                    |lc| lc + b_var - c_var,
                    |lc| lc + a_var,
                    |lc| lc + ch_var - c_var,
                );
            }
            (true, false, false) => {
                // (1-a) × (b - c) = ch - c
                // Expand: b - c - ab + ac = ch - c
                // Rearrange: a × (c - b) = ch - b
                cs.enforce(
                    |lc| lc + c_var - b_var,
                    |lc| lc + a_var,
                    |lc| lc + ch_var - b_var,
                );
            }
            (false, true, false) => {
                // a × ((1-b) - c) = ch - c
                // a × (1 - b - c) = ch - c
                cs.enforce(
                    |lc| lc + one - b_var - c_var,
                    |lc| lc + a_var,
                    |lc| lc + ch_var - c_var,
                );
            }
            (false, false, true) => {
                // a × (b - (1-c)) = ch - (1-c)
                // a × (b - 1 + c) = ch - 1 + c
                cs.enforce(
                    |lc| lc + b_var - one + c_var,
                    |lc| lc + a_var,
                    |lc| lc + ch_var - one + c_var,
                );
            }
            (true, true, false) => {
                // (1-a) × ((1-b) - c) = ch - c
                // (1-a) × (1 - b - c) = ch - c
                // Expand: 1 - b - c - a + ab + ac = ch - c
                // Rearrange: a × (b - 1 + c) = ch - 1 + b
                // Actually let me derive this more carefully:
                // ch = (1-a)(1-b) + a×c = 1 - a - b + ab + ac
                // ch - c = 1 - a - b + ab + ac - c
                // Let's verify: a × (b + c - 1) = ... doesn't work well
                // Try different form: ch = (NOT a AND NOT b) XOR (a AND c)
                // For (1-a,1-b,c): Ch = ((1-a) & (1-b)) ^ (a & c)
                // = (1-a)(1-b) ^ ac = (1-a-b+ab) ^ ac
                // When a=0: (1-b) ^ 0 = 1-b
                // When a=1: 0 ^ c = c
                // So ch = (1-a)(1-b) + a×c - 2(1-a)(1-b)×a×c
                // But the last term is 0 since a(1-a)=0
                // So ch = (1-a)(1-b) + ac
                // Hmm this is getting complicated. Let me use a simpler approach.
                cs.enforce(
                    |lc| lc + one - b_var - c_var,
                    |lc| lc + one - a_var,
                    |lc| lc + ch_var - c_var,
                );
            }
            (true, false, true) => {
                // (1-a) × (b - (1-c)) = ch - (1-c)
                // (1-a) × (b - 1 + c) = ch - 1 + c
                cs.enforce(
                    |lc| lc + b_var + c_var - one,
                    |lc| lc + one - a_var,
                    |lc| lc + ch_var + c_var - one,
                );
            }
            (false, true, true) => {
                // a × ((1-b) - (1-c)) = ch - (1-c)
                // a × (c - b) = ch + c - 1
                cs.enforce(
                    |lc| lc + c_var - b_var,
                    |lc| lc + a_var,
                    |lc| lc + ch_var + c_var - one,
                );
            }
            (true, true, true) => {
                // (1-a) × ((1-b) - (1-c)) = ch - (1-c)
                // (1-a) × (c - b) = ch + c - 1
                cs.enforce(
                    |lc| lc + c_var - b_var,
                    |lc| lc + one - a_var,
                    |lc| lc + ch_var + c_var - one,
                );
            }
        }

        Ok(Boolean {
            var: Some(ch_var),
            value: ch_val,
            negated: false,
            _phantom: PhantomData,
        })
    }

    /// SHA-256 Maj function: (a ∧ b) ⊕ (a ∧ c) ⊕ (b ∧ c)
    ///
    /// Uses 2 constraints total:
    /// 1. bc = b × c (AND constraint)
    /// 2. (2bc - b - c) × a = bc - maj
    ///
    /// This is more efficient than computing with multiple XOR/AND ops
    /// which would require 4 constraints per bit.
    pub fn sha256_maj<CS: SmallConstraintSystem<W, C>>(
        cs: &mut CS,
        a: &Self,
        b: &Self,
        c: &Self,
    ) -> Result<Self, SynthesisError> {
        // Compute expected value
        let maj_val = match (a.get_value(), b.get_value(), c.get_value()) {
            (Some(a), Some(b), Some(c)) => Some((a && b) ^ (a && c) ^ (b && c)),
            _ => None,
        };

        // Handle constant cases
        match (a.var, b.var, c.var) {
            // All constants
            (None, None, None) => {
                return Ok(Boolean::constant(maj_val.unwrap()));
            }
            // a is constant false: Maj(0,b,c) = b AND c
            (None, _, _) if !a.get_value().unwrap() => {
                return Self::and(cs, b, c);
            }
            // a is constant true: Maj(1,b,c) = NOT(NOT(b) AND NOT(c))
            (None, _, _) => {
                return Ok(Self::and(cs, &b.not(), &c.not())?.not());
            }
            // b is constant false: Maj(a,0,c) = a AND c
            (_, None, _) if !b.get_value().unwrap() => {
                return Self::and(cs, a, c);
            }
            // b is constant true: Maj(a,1,c) = NOT(NOT(a) AND NOT(c))
            (_, None, _) => {
                return Ok(Self::and(cs, &a.not(), &c.not())?.not());
            }
            // c is constant false: Maj(a,b,0) = a AND b
            (_, _, None) if !c.get_value().unwrap() => {
                return Self::and(cs, a, b);
            }
            // c is constant true: Maj(a,b,1) = NOT(NOT(a) AND NOT(b))
            (_, _, None) => {
                return Ok(Self::and(cs, &a.not(), &b.not())?.not());
            }
            _ => {}
        }

        // All three are variables
        // First compute bc = b AND c (1 constraint)
        let bc = Self::and(cs, b, c)?;

        // Now use the formula: (2bc - b - c) × a = bc - maj
        let maj_var = cs.alloc(|| {
            if maj_val.unwrap_or(false) {
                W::one()
            } else {
                W::zero()
            }
        })?;

        let (a_var, a_neg) = (a.var.unwrap(), a.negated);
        let (b_var, b_neg) = (b.var.unwrap(), b.negated);
        let (c_var, c_neg) = (c.var.unwrap(), c.negated);
        let bc_var = bc.var.unwrap();
        let two = C::one() + C::one();

        // The general formula: (2bc - b - c) × a = bc - maj
        // With negation, we need to substitute (1-x) for negated variables

        match (a_neg, b_neg, c_neg) {
            (false, false, false) => {
                // (2bc - b - c) × a = bc - maj
                cs.enforce(
                    |lc| lc + (two, bc_var) - b_var - c_var,
                    |lc| lc + a_var,
                    |lc| lc + bc_var - maj_var,
                );
            }
            (true, false, false) => {
                // (2bc - b - c) × (1-a) = bc - maj
                // Expand: 2bc - b - c - 2abc + ab + ac = bc - maj
                // (2bc - b - c) - a(2bc - b - c) = bc - maj
                // a(2bc - b - c) = 2bc - b - c - bc + maj = bc - b - c + maj
                cs.enforce(
                    |lc| lc + (two, bc_var) - b_var - c_var,
                    |lc| lc + a_var,
                    |lc| lc + bc_var - b_var - c_var + maj_var,
                );
            }
            _ => {
                // For other negation combinations, fall back to the 4-op version
                // This is rare in SHA-256 where bits are typically not negated
                let x_xor_y = Self::xor(cs, a, b)?;
                let z_and_xxy = Self::and(cs, c, &x_xor_y)?;
                let x_and_y = Self::and(cs, a, b)?;
                let result = Self::xor(cs, &x_and_y, &z_and_xxy)?;
                return Ok(result);
            }
        }

        Ok(Boolean {
            var: Some(maj_var),
            value: maj_val,
            negated: false,
            _phantom: PhantomData,
        })
    }

    /// Expose this boolean as a public input.
    ///
    /// Creates a new public input variable with the boolean's value
    /// and adds an equality constraint between them.
    pub fn inputize<CS: SmallConstraintSystem<W, C>>(
        &self,
        cs: &mut CS,
    ) -> Result<(), SynthesisError> {
        // Allocate a public input with the same value
        let input_var = cs.alloc_input(|| {
            if self.get_value().unwrap_or(false) {
                W::one()
            } else {
                W::zero()
            }
        })?;

        // Enforce equality: (self - input) * 1 = 0
        // Which simplifies to: 1 * 1 = self - input (when rearranged) won't work
        // Better: self * 1 = input
        let one_var = CS::one();
        match (self.var, self.negated) {
            (None, false) if self.value == Some(true) => {
                // Constant true: 1 * 1 = input
                cs.enforce(|lc| lc + one_var, |lc| lc + one_var, |lc| lc + input_var);
            }
            (None, false) => {
                // Constant false: 0 * 1 = input, i.e., 1 * input = 0
                cs.enforce(
                    |lc| lc + one_var,
                    |lc| lc + input_var,
                    |lc| lc, // zero
                );
            }
            (None, true) if self.value == Some(false) => {
                // NOT false = true: 1 * 1 = input
                cs.enforce(|lc| lc + one_var, |lc| lc + one_var, |lc| lc + input_var);
            }
            (None, true) => {
                // NOT true = false: 1 * input = 0
                cs.enforce(
                    |lc| lc + one_var,
                    |lc| lc + input_var,
                    |lc| lc, // zero
                );
            }
            (Some(var), false) => {
                // var * 1 = input
                cs.enforce(|lc| lc + var, |lc| lc + one_var, |lc| lc + input_var);
            }
            (Some(var), true) => {
                // (1 - var) * 1 = input
                // Rearranged: 1 * 1 = var + input
                cs.enforce(
                    |lc| lc + one_var,
                    |lc| lc + one_var,
                    |lc| lc + var + input_var,
                );
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::small_r1cs::SmallCS;

    #[test]
    fn test_boolean_alloc() {
        let mut cs = SmallCS::<i32, i32>::new();

        let b = Boolean::alloc(&mut cs, Some(true)).unwrap();
        assert_eq!(b.get_value(), Some(true));
        assert!(cs.is_satisfied::<i64>());

        let b = Boolean::alloc(&mut cs, Some(false)).unwrap();
        assert_eq!(b.get_value(), Some(false));
        assert!(cs.is_satisfied::<i64>());
    }

    #[test]
    fn test_boolean_xor() {
        let mut cs = SmallCS::<i32, i32>::new();

        let a = Boolean::alloc(&mut cs, Some(true)).unwrap();
        let b = Boolean::alloc(&mut cs, Some(false)).unwrap();
        let c = Boolean::xor(&mut cs, &a, &b).unwrap();

        assert_eq!(c.get_value(), Some(true)); // 1 XOR 0 = 1
        assert!(cs.is_satisfied::<i64>());
    }

    #[test]
    fn test_boolean_xor_same() {
        let mut cs = SmallCS::<i32, i32>::new();

        let a = Boolean::alloc(&mut cs, Some(true)).unwrap();
        let b = Boolean::alloc(&mut cs, Some(true)).unwrap();
        let c = Boolean::xor(&mut cs, &a, &b).unwrap();

        assert_eq!(c.get_value(), Some(false)); // 1 XOR 1 = 0
        assert!(cs.is_satisfied::<i64>());
    }

    #[test]
    fn test_boolean_and() {
        let mut cs = SmallCS::<i32, i32>::new();

        let a = Boolean::alloc(&mut cs, Some(true)).unwrap();
        let b = Boolean::alloc(&mut cs, Some(true)).unwrap();
        let c = Boolean::and(&mut cs, &a, &b).unwrap();

        assert_eq!(c.get_value(), Some(true)); // 1 AND 1 = 1
        assert!(cs.is_satisfied::<i64>());
    }

    #[test]
    fn test_boolean_and_false() {
        let mut cs = SmallCS::<i32, i32>::new();

        let a = Boolean::alloc(&mut cs, Some(true)).unwrap();
        let b = Boolean::alloc(&mut cs, Some(false)).unwrap();
        let c = Boolean::and(&mut cs, &a, &b).unwrap();

        assert_eq!(c.get_value(), Some(false)); // 1 AND 0 = 0
        assert!(cs.is_satisfied::<i64>());
    }

    #[test]
    fn test_boolean_not() {
        let mut cs = SmallCS::<i32, i32>::new();

        let a = Boolean::alloc(&mut cs, Some(true)).unwrap();
        let not_a = a.not();

        assert_eq!(not_a.get_value(), Some(false));
        // NOT doesn't add constraints
        assert_eq!(cs.num_constraints(), 1); // Only the boolean constraint for a
    }

    #[test]
    fn test_boolean_xor_with_not() {
        let mut cs = SmallCS::<i32, i32>::new();

        let a = Boolean::alloc(&mut cs, Some(true)).unwrap();
        let b = Boolean::alloc(&mut cs, Some(true)).unwrap();
        let not_a = a.not();
        let c = Boolean::xor(&mut cs, &not_a, &b).unwrap();

        assert_eq!(c.get_value(), Some(true)); // NOT(1) XOR 1 = 0 XOR 1 = 1
        assert!(cs.is_satisfied::<i64>());
    }

    #[test]
    fn test_constant_xor() {
        let mut cs = SmallCS::<i32, i32>::new();

        let a = Boolean::constant(true);
        let b = Boolean::alloc(&mut cs, Some(false)).unwrap();
        let c = Boolean::xor(&mut cs, &a, &b).unwrap();

        assert_eq!(c.get_value(), Some(true)); // 1 XOR 0 = 1
        // No XOR constraint needed when one is constant
        assert_eq!(cs.num_constraints(), 1); // Only boolean constraint for b
    }
}
