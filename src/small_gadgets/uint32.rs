//! 32-bit unsigned integer gadget for small-value R1CS.
//!
//! `UInt32<W, C>` represents a 32-bit value as 32 boolean bits,
//! supporting SHA-256 operations and efficient multi-operand addition.

use super::boolean::Boolean;
use crate::small_r1cs::{
    Coefficient, LinearCombination, SmallConstraintSystem, SynthesisError, Witness,
};

/// A 32-bit unsigned integer represented as 32 boolean bits.
///
/// Stored in little-endian order (bits[0] is LSB).
/// Supports rotations, shifts, XOR, and multi-operand addition.
#[derive(Clone, Debug)]
pub struct UInt32<W: Witness, C: Coefficient> {
    /// Little-endian bits (bits[0] is LSB).
    pub bits: [Boolean<W, C>; 32],
    /// Cached value for witness generation.
    pub value: Option<u32>,
}

impl<W: Witness, C: Coefficient> UInt32<W, C> {
    /// Create a UInt32 from little-endian bits.
    pub fn from_bits_le(bits: [Boolean<W, C>; 32]) -> Self {
        let value = Self::compute_value(&bits);
        UInt32 { bits, value }
    }

    /// Create a UInt32 from big-endian bits.
    pub fn from_bits_be(bits: &[Boolean<W, C>; 32]) -> Self {
        let le_bits: [Boolean<W, C>; 32] = std::array::from_fn(|i| bits[31 - i].clone());
        Self::from_bits_le(le_bits)
    }

    /// Compute the value from bits.
    fn compute_value(bits: &[Boolean<W, C>; 32]) -> Option<u32> {
        let mut value = 0u32;
        for (i, bit) in bits.iter().enumerate() {
            if let Some(b) = bit.get_value() {
                if b {
                    value |= 1 << i;
                }
            } else {
                return None;
            }
        }
        Some(value)
    }

    /// Create a constant UInt32 (no constraints needed).
    pub fn constant(value: u32) -> Self {
        let bits: [Boolean<W, C>; 32] =
            std::array::from_fn(|i| Boolean::constant((value >> i) & 1 == 1));
        UInt32 {
            bits,
            value: Some(value),
        }
    }

    /// Allocate a UInt32 from a value.
    ///
    /// Creates 32 boolean constraints (one per bit).
    pub fn alloc<CS: SmallConstraintSystem<W, C>>(
        cs: &mut CS,
        value: Option<u32>,
    ) -> Result<Self, SynthesisError> {
        let mut bits: [Option<Boolean<W, C>>; 32] = std::array::from_fn(|_| None);

        for i in 0..32 {
            let bit_val = value.map(|v| (v >> i) & 1 == 1);
            bits[i] = Some(Boolean::alloc(cs, bit_val)?);
        }

        let bits: [Boolean<W, C>; 32] = bits.map(|b| b.unwrap());
        Ok(UInt32 { bits, value })
    }

    /// Rotate right by `by` bits.
    ///
    /// No constraints needed - just reorders the bits.
    pub fn rotr(&self, by: usize) -> Self {
        let by = by % 32;
        let bits: [Boolean<W, C>; 32] = std::array::from_fn(|i| self.bits[(i + by) % 32].clone());
        UInt32 {
            bits,
            value: self.value.map(|v| v.rotate_right(by as u32)),
        }
    }

    /// Shift right by `by` bits.
    ///
    /// No constraints needed - inserts zero constant bits.
    pub fn shr(&self, by: usize) -> Self {
        let bits: [Boolean<W, C>; 32] = std::array::from_fn(|i| {
            if i + by < 32 {
                self.bits[i + by].clone()
            } else {
                Boolean::constant(false)
            }
        });
        UInt32 {
            bits,
            value: self.value.map(|v| v >> by),
        }
    }

    /// XOR two UInt32 values.
    ///
    /// 32 constraints (one XOR per bit).
    pub fn xor<CS: SmallConstraintSystem<W, C>>(
        &self,
        cs: &mut CS,
        other: &Self,
    ) -> Result<Self, SynthesisError> {
        let mut bits: [Option<Boolean<W, C>>; 32] = std::array::from_fn(|_| None);

        for i in 0..32 {
            bits[i] = Some(Boolean::xor(cs, &self.bits[i], &other.bits[i])?);
        }

        let bits: [Boolean<W, C>; 32] = bits.map(|b| b.unwrap());
        Ok(UInt32 {
            bits,
            value: match (self.value, other.value) {
                (Some(a), Some(b)) => Some(a ^ b),
                _ => None,
            },
        })
    }

    /// XOR three UInt32 values.
    ///
    /// 64 constraints (two XORs per bit).
    pub fn xor3<CS: SmallConstraintSystem<W, C>>(
        cs: &mut CS,
        a: &Self,
        b: &Self,
        c: &Self,
    ) -> Result<Self, SynthesisError> {
        let ab = a.xor(cs, b)?;
        ab.xor(cs, c)
    }

    /// Add multiple UInt32 values using 2-limb decomposition.
    ///
    /// Splits each 32-bit value into two 16-bit limbs to keep
    /// coefficients within i32 bounds (max 2^16).
    ///
    /// For n operands with 4 operands: ~38 constraints total.
    /// - 32 result bit constraints
    /// - ~2-3 carry bit constraints
    /// - ~2-3 overflow bit constraints
    /// - 2 equality constraints (lower and upper limb)
    pub fn add_many<CS: SmallConstraintSystem<W, C>>(
        cs: &mut CS,
        operands: &[Self],
    ) -> Result<Self, SynthesisError>
    where
        C: From<i32>,
    {
        assert!(!operands.is_empty(), "add_many requires at least one operand");

        if operands.len() == 1 {
            return Ok(operands[0].clone());
        }

        // Compute the witness value
        let sum_value: Option<u32> = operands
            .iter()
            .try_fold(0u32, |acc, op| op.value.map(|v| acc.wrapping_add(v)));

        // Compute limb sums for witness generation
        let (lower_sum, upper_sum): (Option<u32>, Option<u32>) = operands.iter().fold(
            (Some(0u32), Some(0u32)),
            |(lo, hi), op| match (lo, hi, op.value) {
                (Some(lo), Some(hi), Some(v)) => {
                    let op_lo = v & 0xFFFF;
                    let op_hi = (v >> 16) & 0xFFFF;
                    (Some(lo + op_lo), Some(hi + op_hi))
                }
                _ => (None, None),
            },
        );

        // Carry from lower to upper limb
        let carry_value = lower_sum.map(|s| s >> 16);
        // Overflow from upper limb (discarded for mod 2^32)
        let overflow_value =
            upper_sum.and_then(|u| carry_value.map(|c| (u + c) >> 16));

        // Allocate result bits (32 boolean constraints)
        let mut result_bits: [Option<Boolean<W, C>>; 32] = std::array::from_fn(|_| None);
        for i in 0..32 {
            let bit_val = sum_value.map(|s| (s >> i) & 1 == 1);
            result_bits[i] = Some(Boolean::alloc(cs, bit_val)?);
        }
        let result_bits: [Boolean<W, C>; 32] = result_bits.map(|b| b.unwrap());

        // Allocate carry bits
        // Number of bits needed: ceil(log2(n)) + 1 where n = number of operands
        let num_carry_bits = ((operands.len() as f64).log2().ceil() as usize + 1).max(1);
        let carry_bits = Self::alloc_multi_bits(cs, carry_value, num_carry_bits)?;

        // Allocate overflow bits
        let overflow_bits = Self::alloc_multi_bits(cs, overflow_value, num_carry_bits)?;

        // ========================================
        // Constraint 1: Lower limb equality
        // ========================================
        // Σ operand_lower = result_lower + carry × 2^16
        let mut lhs_lower = LinearCombination::<C>::zero();
        for operand in operands {
            for i in 0..16 {
                let coeff = C::from(1i32 << i);
                if let Some(var) = operand.bits[i].var {
                    if operand.bits[i].is_negated() {
                        // NOT bit: use (1 - var)
                        lhs_lower = lhs_lower + (coeff, CS::one()) - (coeff, var);
                    } else {
                        lhs_lower = lhs_lower + (coeff, var);
                    }
                } else if operand.bits[i].get_value() == Some(true) {
                    lhs_lower = lhs_lower + (coeff, CS::one());
                }
            }
        }

        let mut rhs_lower = LinearCombination::<C>::zero();
        // Result lower 16 bits
        for i in 0..16 {
            let coeff = C::from(1i32 << i);
            if let Some(var) = result_bits[i].var {
                rhs_lower = rhs_lower + (coeff, var);
            }
        }
        // Carry × 2^16
        for (k, carry_bit) in carry_bits.iter().enumerate() {
            if k >= 16 {
                break; // Coefficient would overflow i32
            }
            let coeff = C::from(1i32 << (16 + k));
            if let Some(var) = carry_bit.var {
                rhs_lower = rhs_lower + (coeff, var);
            }
        }

        cs.enforce(|_| lhs_lower.clone(), |lc| lc + CS::one(), |_| rhs_lower.clone());

        // ========================================
        // Constraint 2: Upper limb equality
        // ========================================
        // Σ operand_upper + carry = result_upper + overflow × 2^16
        let mut lhs_upper = LinearCombination::<C>::zero();
        // Upper bits of operands
        for operand in operands {
            for i in 16..32 {
                let coeff = C::from(1i32 << (i - 16));
                if let Some(var) = operand.bits[i].var {
                    if operand.bits[i].is_negated() {
                        lhs_upper = lhs_upper + (coeff, CS::one()) - (coeff, var);
                    } else {
                        lhs_upper = lhs_upper + (coeff, var);
                    }
                } else if operand.bits[i].get_value() == Some(true) {
                    lhs_upper = lhs_upper + (coeff, CS::one());
                }
            }
        }
        // Add carry from lower limb
        for (k, carry_bit) in carry_bits.iter().enumerate() {
            let coeff = C::from(1i32 << k);
            if let Some(var) = carry_bit.var {
                lhs_upper = lhs_upper + (coeff, var);
            }
        }

        let mut rhs_upper = LinearCombination::<C>::zero();
        // Result upper 16 bits
        for i in 16..32 {
            let coeff = C::from(1i32 << (i - 16));
            if let Some(var) = result_bits[i].var {
                rhs_upper = rhs_upper + (coeff, var);
            }
        }
        // Overflow × 2^16
        for (k, overflow_bit) in overflow_bits.iter().enumerate() {
            if k >= 16 {
                break;
            }
            let coeff = C::from(1i32 << (16 + k));
            if let Some(var) = overflow_bit.var {
                rhs_upper = rhs_upper + (coeff, var);
            }
        }

        cs.enforce(|_| lhs_upper.clone(), |lc| lc + CS::one(), |_| rhs_upper.clone());

        Ok(UInt32 {
            bits: result_bits,
            value: sum_value,
        })
    }

    /// Allocate multiple bits for carry/overflow representation.
    fn alloc_multi_bits<CS: SmallConstraintSystem<W, C>>(
        cs: &mut CS,
        value: Option<u32>,
        num_bits: usize,
    ) -> Result<Vec<Boolean<W, C>>, SynthesisError> {
        let mut bits = Vec::with_capacity(num_bits);
        for i in 0..num_bits {
            let bit_val = value.map(|v| (v >> i) & 1 == 1);
            bits.push(Boolean::alloc(cs, bit_val)?);
        }
        Ok(bits)
    }

    // ========================================
    // SHA-256 specific operations
    // ========================================

    /// σ0(x) = ROTR^7(x) ⊕ ROTR^18(x) ⊕ SHR^3(x)
    ///
    /// 64 XOR constraints (2 XORs of 32 bits each).
    pub fn sha256_sigma0<CS: SmallConstraintSystem<W, C>>(
        &self,
        cs: &mut CS,
    ) -> Result<Self, SynthesisError> {
        let rotr7 = self.rotr(7);
        let rotr18 = self.rotr(18);
        let shr3 = self.shr(3);
        let t = rotr7.xor(cs, &rotr18)?;
        t.xor(cs, &shr3)
    }

    /// σ1(x) = ROTR^17(x) ⊕ ROTR^19(x) ⊕ SHR^10(x)
    ///
    /// 64 XOR constraints.
    pub fn sha256_sigma1<CS: SmallConstraintSystem<W, C>>(
        &self,
        cs: &mut CS,
    ) -> Result<Self, SynthesisError> {
        let rotr17 = self.rotr(17);
        let rotr19 = self.rotr(19);
        let shr10 = self.shr(10);
        let t = rotr17.xor(cs, &rotr19)?;
        t.xor(cs, &shr10)
    }

    /// Σ0(x) = ROTR^2(x) ⊕ ROTR^13(x) ⊕ ROTR^22(x)
    ///
    /// 64 XOR constraints.
    pub fn sha256_sum0<CS: SmallConstraintSystem<W, C>>(
        &self,
        cs: &mut CS,
    ) -> Result<Self, SynthesisError> {
        let rotr2 = self.rotr(2);
        let rotr13 = self.rotr(13);
        let rotr22 = self.rotr(22);
        let t = rotr2.xor(cs, &rotr13)?;
        t.xor(cs, &rotr22)
    }

    /// Σ1(x) = ROTR^6(x) ⊕ ROTR^11(x) ⊕ ROTR^25(x)
    ///
    /// 64 XOR constraints.
    pub fn sha256_sum1<CS: SmallConstraintSystem<W, C>>(
        &self,
        cs: &mut CS,
    ) -> Result<Self, SynthesisError> {
        let rotr6 = self.rotr(6);
        let rotr11 = self.rotr(11);
        let rotr25 = self.rotr(25);
        let t = rotr6.xor(cs, &rotr11)?;
        t.xor(cs, &rotr25)
    }

    /// Ch(x, y, z) = (x ∧ y) ⊕ (¬x ∧ z)
    ///
    /// Optimized: ch = z ^ (x & (y ^ z))
    /// 96 constraints (32 × 3 ops: XOR, AND, XOR).
    pub fn sha256_ch<CS: SmallConstraintSystem<W, C>>(
        cs: &mut CS,
        x: &Self,
        y: &Self,
        z: &Self,
    ) -> Result<Self, SynthesisError> {
        let mut bits: [Option<Boolean<W, C>>; 32] = std::array::from_fn(|_| None);

        for i in 0..32 {
            let y_xor_z = Boolean::xor(cs, &y.bits[i], &z.bits[i])?;
            let x_and_yxz = Boolean::and(cs, &x.bits[i], &y_xor_z)?;
            bits[i] = Some(Boolean::xor(cs, &z.bits[i], &x_and_yxz)?);
        }

        let bits: [Boolean<W, C>; 32] = bits.map(|b| b.unwrap());
        let value = match (x.value, y.value, z.value) {
            (Some(x), Some(y), Some(z)) => Some((x & y) ^ ((!x) & z)),
            _ => None,
        };

        Ok(UInt32 { bits, value })
    }

    /// Maj(x, y, z) = (x ∧ y) ⊕ (x ∧ z) ⊕ (y ∧ z)
    ///
    /// Optimized: maj = (x & y) ^ (z & (x ^ y))
    /// 128 constraints (32 × 4 ops: XOR, AND, AND, XOR).
    pub fn sha256_maj<CS: SmallConstraintSystem<W, C>>(
        cs: &mut CS,
        x: &Self,
        y: &Self,
        z: &Self,
    ) -> Result<Self, SynthesisError> {
        let mut bits: [Option<Boolean<W, C>>; 32] = std::array::from_fn(|_| None);

        for i in 0..32 {
            let x_xor_y = Boolean::xor(cs, &x.bits[i], &y.bits[i])?;
            let z_and_xxy = Boolean::and(cs, &z.bits[i], &x_xor_y)?;
            let x_and_y = Boolean::and(cs, &x.bits[i], &y.bits[i])?;
            bits[i] = Some(Boolean::xor(cs, &x_and_y, &z_and_xxy)?);
        }

        let bits: [Boolean<W, C>; 32] = bits.map(|b| b.unwrap());
        let value = match (x.value, y.value, z.value) {
            (Some(x), Some(y), Some(z)) => Some((x & y) ^ (x & z) ^ (y & z)),
            _ => None,
        };

        Ok(UInt32 { bits, value })
    }

    /// Convert to big-endian bits (for output).
    pub fn into_bits_be(self) -> [Boolean<W, C>; 32] {
        std::array::from_fn(|i| self.bits[31 - i].clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::small_r1cs::SmallCS;

    #[test]
    fn test_uint32_constant() {
        let u = UInt32::<i32, i32>::constant(0x12345678);
        assert_eq!(u.value, Some(0x12345678));
    }

    #[test]
    fn test_uint32_alloc() {
        let mut cs = SmallCS::<i32, i32>::new();
        let u = UInt32::alloc(&mut cs, Some(42)).unwrap();
        assert_eq!(u.value, Some(42));
        assert!(cs.is_satisfied::<i64>());
        assert_eq!(cs.num_constraints(), 32); // 32 boolean constraints
    }

    #[test]
    fn test_uint32_rotr() {
        let u = UInt32::<i32, i32>::constant(0x80000001);
        let rotated = u.rotr(1);
        assert_eq!(rotated.value, Some(0xC0000000));
    }

    #[test]
    fn test_uint32_shr() {
        let u = UInt32::<i32, i32>::constant(0x80000001);
        let shifted = u.shr(1);
        assert_eq!(shifted.value, Some(0x40000000));
    }

    #[test]
    fn test_uint32_xor() {
        let mut cs = SmallCS::<i32, i32>::new();

        let a = UInt32::alloc(&mut cs, Some(0xFF00FF00)).unwrap();
        let b = UInt32::alloc(&mut cs, Some(0x0F0F0F0F)).unwrap();
        let c = a.xor(&mut cs, &b).unwrap();

        assert_eq!(c.value, Some(0xFF00FF00 ^ 0x0F0F0F0F));
        assert!(cs.is_satisfied::<i64>());
    }

    #[test]
    fn test_uint32_add_many_two() {
        let mut cs = SmallCS::<i32, i32>::new();

        let a = UInt32::alloc(&mut cs, Some(100)).unwrap();
        let b = UInt32::alloc(&mut cs, Some(200)).unwrap();
        let sum = UInt32::add_many(&mut cs, &[a, b]).unwrap();

        assert_eq!(sum.value, Some(300));
        assert!(cs.is_satisfied::<i64>());
    }

    #[test]
    fn test_uint32_add_many_four() {
        let mut cs = SmallCS::<i32, i32>::new();

        let a = UInt32::alloc(&mut cs, Some(0x10000000)).unwrap();
        let b = UInt32::alloc(&mut cs, Some(0x20000000)).unwrap();
        let c = UInt32::alloc(&mut cs, Some(0x30000000)).unwrap();
        let d = UInt32::alloc(&mut cs, Some(0x40000000)).unwrap();
        let sum = UInt32::add_many(&mut cs, &[a, b, c, d]).unwrap();

        // 0x10000000 + 0x20000000 + 0x30000000 + 0x40000000 = 0xA0000000
        assert_eq!(sum.value, Some(0xA0000000));
        assert!(cs.is_satisfied::<i64>());
    }

    #[test]
    fn test_uint32_add_many_overflow() {
        let mut cs = SmallCS::<i32, i32>::new();

        let a = UInt32::alloc(&mut cs, Some(0xFFFFFFFF)).unwrap();
        let b = UInt32::alloc(&mut cs, Some(1)).unwrap();
        let sum = UInt32::add_many(&mut cs, &[a, b]).unwrap();

        // Wrapping: 0xFFFFFFFF + 1 = 0 (mod 2^32)
        assert_eq!(sum.value, Some(0));
        assert!(cs.is_satisfied::<i64>());
    }

    #[test]
    fn test_sha256_ch() {
        let mut cs = SmallCS::<i32, i32>::new();

        let x = UInt32::alloc(&mut cs, Some(0xFFFF0000)).unwrap();
        let y = UInt32::alloc(&mut cs, Some(0xFF00FF00)).unwrap();
        let z = UInt32::alloc(&mut cs, Some(0xF0F0F0F0)).unwrap();

        let ch = UInt32::sha256_ch(&mut cs, &x, &y, &z).unwrap();

        // Ch(x,y,z) = (x & y) ^ (~x & z)
        let expected = (0xFFFF0000 & 0xFF00FF00) ^ ((!0xFFFF0000) & 0xF0F0F0F0);
        assert_eq!(ch.value, Some(expected));
        assert!(cs.is_satisfied::<i64>());
    }

    #[test]
    fn test_sha256_maj() {
        let mut cs = SmallCS::<i32, i32>::new();

        let x = UInt32::alloc(&mut cs, Some(0xFFFF0000)).unwrap();
        let y = UInt32::alloc(&mut cs, Some(0xFF00FF00)).unwrap();
        let z = UInt32::alloc(&mut cs, Some(0xF0F0F0F0)).unwrap();

        let maj = UInt32::sha256_maj(&mut cs, &x, &y, &z).unwrap();

        // Maj(x,y,z) = (x & y) ^ (x & z) ^ (y & z)
        let expected =
            (0xFFFF0000 & 0xFF00FF00) ^ (0xFFFF0000 & 0xF0F0F0F0) ^ (0xFF00FF00 & 0xF0F0F0F0);
        assert_eq!(maj.value, Some(expected));
        assert!(cs.is_satisfied::<i64>());
    }

    #[test]
    fn test_sha256_sigma0() {
        let mut cs = SmallCS::<i32, i32>::new();

        let x = UInt32::alloc(&mut cs, Some(0x12345678)).unwrap();
        let sigma0 = x.sha256_sigma0(&mut cs).unwrap();

        // σ0(x) = ROTR^7(x) ^ ROTR^18(x) ^ SHR^3(x)
        let expected = 0x12345678u32.rotate_right(7)
            ^ 0x12345678u32.rotate_right(18)
            ^ (0x12345678u32 >> 3);
        assert_eq!(sigma0.value, Some(expected));
        assert!(cs.is_satisfied::<i64>());
    }
}
