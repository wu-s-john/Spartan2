# Small-Value SHA256

This document describes the SHA-256 gadget implementation using small-value R1CS types (i32 coefficients, i64 witnesses) instead of field elements.

## Overview

SHA-256 is compute-intensive, requiring many 32-bit additions. With traditional Bellpepper, coefficients grow exponentially (up to 2^237), requiring `SmallMultiEq` to batch equality constraints. With small-value R1CS, coefficients are bounded by type (i32), eliminating this complexity.

## Type Parameters

| Type | Purpose | Concrete | Size |
|------|---------|----------|------|
| **C** (Coefficient) | Matrix entries, LC terms | i32 | 4 bytes |
| **W** (Witness) | Variable values | i32 | 4 bytes |
| **Acc** (Accumulator) | LC evaluation = Σ C×W | i64 | 8 bytes |

```rust
// Max coefficient for 2-limb addition: 2^16 = 65,536 (fits i32)
// Witnesses are bits (0 or 1) stored as i32
// LC evaluation: Σ (i32 × i32) → i64
```

## Core Types

### Boolean<W, C>

```rust
/// Boolean gadget with W=i32 witnesses and C=i32 coefficients
pub struct Boolean<W, C> {
    /// None = constant, Some = allocated variable
    var: Option<Variable>,
    /// Known value (if any)
    value: Option<bool>,
    _phantom: PhantomData<(W, C)>,
}

impl<W: Witness, C: Coefficient> Boolean<W, C> {
    pub fn constant(b: bool) -> Self {
        Boolean {
            var: None,
            value: Some(b),
            _phantom: PhantomData,
        }
    }

    pub fn alloc<CS: ConstraintSystem<W, C>>(
        cs: &mut CS,
        value: Option<bool>,
    ) -> Result<Self, SynthesisError> {
        // Allocate W=i32 witness (0 or 1)
        let var = cs.alloc(value.map(|b| if b { W::one() } else { W::zero() }))?;

        // Enforce boolean constraint: b × (1 - b) = 0
        cs.enforce(
            LinearCombination::from_variable(var),
            LinearCombination::from_variable(var),
            LinearCombination::from_variable(var),
        );

        Ok(Boolean {
            var: Some(var),
            value,
            _phantom: PhantomData,
        })
    }

    pub fn xor<CS: ConstraintSystem<W, C>>(
        cs: &mut CS,
        a: &Self,
        b: &Self,
    ) -> Result<Self, SynthesisError> {
        // XOR: a + b - 2ab
        // Constraint: 2a × b = a + b - c
        let value = match (a.value, b.value) {
            (Some(a), Some(b)) => Some(a ^ b),
            _ => None,
        };

        let var = cs.alloc(value.map(|b| if b { W::one() } else { W::zero() }))?;

        // 2a × b = a + b - c (coefficients are C=i32)
        let two = C::one() + C::one();
        cs.enforce(
            LinearCombination::from_variable_scaled(a.var.unwrap(), two),
            LinearCombination::from_variable(b.var.unwrap()),
            LinearCombination::zero()
                + (C::one(), a.var.unwrap())
                + (C::one(), b.var.unwrap())
                - (C::one(), var),
        );

        Ok(Boolean { var: Some(var), value, _phantom: PhantomData })
    }

    pub fn and<CS: ConstraintSystem<W, C>>(
        cs: &mut CS,
        a: &Self,
        b: &Self,
    ) -> Result<Self, SynthesisError> {
        // AND: a × b = c
        let value = match (a.value, b.value) {
            (Some(a), Some(b)) => Some(a && b),
            _ => None,
        };

        let var = cs.alloc(value.map(|b| if b { W::one() } else { W::zero() }))?;

        cs.enforce(
            LinearCombination::from_variable(a.var.unwrap()),
            LinearCombination::from_variable(b.var.unwrap()),
            LinearCombination::from_variable(var),
        );

        Ok(Boolean { var: Some(var), value, _phantom: PhantomData })
    }
}
```

### UInt32<W, C>

```rust
/// 32-bit integer gadget with W=i32 witnesses and C=i32 coefficients
/// Stored as 32 boolean bits (needed for XOR, AND, rotations)
pub struct UInt32<W, C> {
    /// Little-endian bits
    bits: [Boolean<W, C>; 32],
    /// Cached value
    value: Option<u32>,
}

impl<W: Witness, C: Coefficient> UInt32<W, C> {
    pub fn from_bits_le(bits: [Boolean<W, C>; 32]) -> Self {
        let value = bits.iter().rev().try_fold(0u32, |acc, bit| {
            bit.value.map(|b| if b { (acc << 1) | 1 } else { acc << 1 })
        });
        UInt32 { bits, value }
    }

    /// Rotation - no constraints (just reorders bits)
    pub fn rotr(&self, by: usize) -> Self {
        let by = by % 32;
        let bits: [Boolean<W, C>; 32] = std::array::from_fn(|i| {
            self.bits[(i + by) % 32].clone()
        });
        UInt32 { bits, value: self.value.map(|v| v.rotate_right(by as u32)) }
    }

    /// Shift - no constraints (inserts zero bits)
    pub fn shr(&self, by: usize) -> Self {
        let bits: [Boolean<W, C>; 32] = std::array::from_fn(|i| {
            if i + by < 32 { self.bits[i + by].clone() }
            else { Boolean::constant(false) }
        });
        UInt32 { bits, value: self.value.map(|v| v >> by) }
    }

    /// XOR - 32 constraints (one per bit)
    pub fn xor<CS: ConstraintSystem<W, C>>(
        &self,
        cs: &mut CS,
        other: &Self,
    ) -> Result<Self, SynthesisError> {
        let mut bits = [const { None }; 32];
        for i in 0..32 {
            bits[i] = Some(Boolean::xor(cs, &self.bits[i], &other.bits[i])?);
        }
        Ok(UInt32::from_bits_le(bits.map(|b| b.unwrap())))
    }
}
```

## 2-Limb Addition (addmany)

The key optimization for SHA-256 is efficient multi-operand addition. We use a 2-limb approach with 16 bits per limb to keep coefficients within i32 bounds.

### Design

```
32-bit value = [lower 16 bits] + [upper 16 bits] × 2^16

For n operands:
- Lower limb: Σ operand_lower = result_lower + carry × 2^16
- Upper limb: Σ operand_upper + carry = result_upper + overflow × 2^16
```

### Coefficient Analysis

| Component | Max Coefficient |
|-----------|-----------------|
| Bit position (2^i for i < 16) | 2^15 = 32,768 |
| Number of operands (n ≤ 7 for SHA-256) | 7 |
| Combined: n × 2^15 | 7 × 32,768 = 229,376 |
| Carry term (2^16) | 65,536 |

All coefficients fit comfortably in i32 (max ≈ 2.1 billion).

### Implementation

```rust
impl<W: Witness, C: Coefficient> UInt32<W, C> {
    /// Add multiple 32-bit values with 2-limb decomposition.
    ///
    /// For 4 operands: 38 constraints total.
    /// Max coefficient: 2^16 = 65,536 (fits i32).
    pub fn add_many<CS: ConstraintSystem<W, C>>(
        cs: &mut CS,
        operands: &[Self],
    ) -> Result<Self, SynthesisError> {
        assert!(!operands.is_empty());

        if operands.len() == 1 {
            return Ok(operands[0].clone());
        }

        // Compute witness values
        let sum_value: Option<u32> = operands
            .iter()
            .try_fold(0u32, |acc, op| op.value.map(|v| acc.wrapping_add(v)));

        // Compute lower and upper limb sums for witness generation
        let (lower_sum, upper_sum): (Option<u32>, Option<u32>) = operands
            .iter()
            .try_fold((Some(0u32), Some(0u32)), |(lo, hi), op| {
                match (lo, hi, op.value) {
                    (Some(lo), Some(hi), Some(v)) => {
                        let op_lo = v & 0xFFFF;
                        let op_hi = (v >> 16) & 0xFFFF;
                        (Some(lo + op_lo), Some(hi + op_hi))
                    }
                    _ => (None, None),
                }
            });

        // Carry from lower to upper limb (can be multiple bits for many operands)
        let carry_value = lower_sum.map(|s| s >> 16);

        // Result bits
        let result_lower = sum_value.map(|s| s & 0xFFFF);
        let result_upper = sum_value.map(|s| (s >> 16) & 0xFFFF);

        // Allocate result bits (32 bits)
        let mut result_bits = [const { None }; 32];
        for i in 0..32 {
            let bit_val = sum_value.map(|s| (s >> i) & 1 == 1);
            result_bits[i] = Some(Boolean::alloc(cs, bit_val)?);
        }
        let result_bits: [Boolean<V, C>; 32] = result_bits.map(|b| b.unwrap());

        // Allocate carry bits (ceil(log2(n)) bits, typically 3-4 for SHA-256)
        let carry_bits = Self::alloc_carry_bits(cs, carry_value, operands.len())?;

        // ========================================
        // Constraint 1: Lower limb
        // ========================================
        // Σ (operand_bits[0..16] as integer) = result_lower + carry × 2^16
        //
        // LHS: Σ_j Σ_i operand[j].bits[i] × 2^i  (for i in 0..16)
        // RHS: Σ_i result_bits[i] × 2^i + Σ_k carry_bits[k] × 2^(16+k)

        let mut lhs_lower = LinearCombination::<C>::zero();
        for operand in operands {
            for i in 0..16 {
                let coeff = C::from_i32(1 << i);
                if let Some(var) = operand.bits[i].var {
                    lhs_lower = lhs_lower + (coeff, var);
                } else if operand.bits[i].value == Some(true) {
                    lhs_lower = lhs_lower + (coeff, Variable::one());
                }
            }
        }

        let mut rhs_lower = LinearCombination::<C>::zero();
        // Result lower 16 bits
        for i in 0..16 {
            let coeff = C::from_i32(1 << i);
            if let Some(var) = result_bits[i].var {
                rhs_lower = rhs_lower + (coeff, var);
            }
        }
        // Carry × 2^16
        for (k, carry_bit) in carry_bits.iter().enumerate() {
            let coeff = C::from_i32(1 << (16 + k));
            if let Some(var) = carry_bit.var {
                rhs_lower = rhs_lower + (coeff, var);
            }
        }

        cs.enforce(
            lhs_lower,
            LinearCombination::one(),
            rhs_lower,
        );

        // ========================================
        // Constraint 2: Upper limb
        // ========================================
        // Σ (operand_bits[16..32] as integer) + carry = result_upper + overflow × 2^16
        //
        // We ignore overflow (it's the discarded high bits for mod 2^32)

        let mut lhs_upper = LinearCombination::<C>::zero();
        // Upper bits of operands
        for operand in operands {
            for i in 16..32 {
                let coeff = C::from_i32(1 << (i - 16));
                if let Some(var) = operand.bits[i].var {
                    lhs_upper = lhs_upper + (coeff, var);
                } else if operand.bits[i].value == Some(true) {
                    lhs_upper = lhs_upper + (coeff, Variable::one());
                }
            }
        }
        // Add carry from lower limb
        for (k, carry_bit) in carry_bits.iter().enumerate() {
            let coeff = C::from_i32(1 << k);
            if let Some(var) = carry_bit.var {
                lhs_upper = lhs_upper + (coeff, var);
            }
        }

        let mut rhs_upper = LinearCombination::<C>::zero();
        // Result upper 16 bits
        for i in 16..32 {
            let coeff = C::from_i32(1 << (i - 16));
            if let Some(var) = result_bits[i].var {
                rhs_upper = rhs_upper + (coeff, var);
            }
        }
        // Overflow bits (allocated but unconstrained elsewhere - just absorbs excess)
        let overflow_bits = Self::alloc_carry_bits(cs,
            upper_sum.and_then(|u| carry_value.map(|c| (u + c) >> 16)),
            operands.len())?;
        for (k, overflow_bit) in overflow_bits.iter().enumerate() {
            let coeff = C::from_i32(1 << (16 + k));
            if let Some(var) = overflow_bit.var {
                rhs_upper = rhs_upper + (coeff, var);
            }
        }

        cs.enforce(
            lhs_upper,
            LinearCombination::one(),
            rhs_upper,
        );

        Ok(UInt32::from_bits_le(result_bits))
    }

    /// Allocate carry bits for addition.
    /// Number of bits = ceil(log2(n)) where n = number of operands.
    fn alloc_carry_bits<CS: ConstraintSystem<V, C>>(
        cs: &mut CS,
        carry_value: Option<u32>,
        num_operands: usize,
    ) -> Result<Vec<Boolean<V, C>>, SynthesisError> {
        // Number of bits needed to represent max carry
        // Max carry from n operands summing 16-bit values: n × (2^16 - 1) / 2^16 ≈ n
        let num_bits = (num_operands as f64).log2().ceil() as usize + 1;

        let mut bits = Vec::with_capacity(num_bits);
        for i in 0..num_bits {
            let bit_val = carry_value.map(|c| (c >> i) & 1 == 1);
            bits.push(Boolean::alloc(cs, bit_val)?);
        }
        Ok(bits)
    }
}
```

### Constraint Count Analysis

**For addmany(4 operands):**

| Component | Constraints |
|-----------|-------------|
| Lower limb equality | 1 |
| Upper limb equality | 1 |
| Result bits (32 boolean) | 32 |
| Carry bits (2 for 4 operands) | 2 |
| Overflow bits (2) | 2 |
| **Total** | **38** |

**Carry bit calculation:** For n=4 operands, max carry = floor(4 × 65535 / 65536) = 3, needs 2 bits.

Compare to SmallMultiEq approach which requires batching and could need hundreds of constraints.

## SHA-256 Operations

### Sigma Functions (no constraints)

```rust
impl<W: Witness, C: Coefficient> UInt32<W, C> {
    /// σ0(x) = ROTR^7(x) ⊕ ROTR^18(x) ⊕ SHR^3(x)
    pub fn sha256_sigma0<CS: ConstraintSystem<W, C>>(
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
    pub fn sha256_sigma1<CS: ConstraintSystem<W, C>>(
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
    pub fn sha256_sum0<CS: ConstraintSystem<W, C>>(
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
    pub fn sha256_sum1<CS: ConstraintSystem<W, C>>(
        &self,
        cs: &mut CS,
    ) -> Result<Self, SynthesisError> {
        let rotr6 = self.rotr(6);
        let rotr11 = self.rotr(11);
        let rotr25 = self.rotr(25);
        let t = rotr6.xor(cs, &rotr11)?;
        t.xor(cs, &rotr25)
    }
}
```

### Ch and Maj Functions

```rust
impl<W: Witness, C: Coefficient> UInt32<W, C> {
    /// Ch(x, y, z) = (x ∧ y) ⊕ (¬x ∧ z)
    /// Optimized: ch = z ^ (x & (y ^ z))
    /// 96 constraints (32 bits × 3 ops: xor, and, xor)
    pub fn sha256_ch<CS: ConstraintSystem<W, C>>(
        cs: &mut CS,
        x: &Self,
        y: &Self,
        z: &Self,
    ) -> Result<Self, SynthesisError> {
        let mut result_bits = [const { None }; 32];
        for i in 0..32 {
            let y_xor_z = Boolean::xor(cs, &y.bits[i], &z.bits[i])?;
            let x_and_yxz = Boolean::and(cs, &x.bits[i], &y_xor_z)?;
            result_bits[i] = Some(Boolean::xor(cs, &z.bits[i], &x_and_yxz)?);
        }
        Ok(UInt32::from_bits_le(result_bits.map(|b| b.unwrap())))
    }

    /// Maj(x, y, z) = (x ∧ y) ⊕ (x ∧ z) ⊕ (y ∧ z)
    /// Optimized: maj = (x & y) ^ (z & (x ^ y))
    /// 128 constraints (32 bits × 4 ops: xor, and, and, xor)
    pub fn sha256_maj<CS: ConstraintSystem<W, C>>(
        cs: &mut CS,
        x: &Self,
        y: &Self,
        z: &Self,
    ) -> Result<Self, SynthesisError> {
        let mut result_bits = [const { None }; 32];
        for i in 0..32 {
            let x_xor_y = Boolean::xor(cs, &x.bits[i], &y.bits[i])?;
            let z_and_xxy = Boolean::and(cs, &z.bits[i], &x_xor_y)?;
            let x_and_y = Boolean::and(cs, &x.bits[i], &y.bits[i])?;
            result_bits[i] = Some(Boolean::xor(cs, &x_and_y, &z_and_xxy)?);
        }
        Ok(UInt32::from_bits_le(result_bits.map(|b| b.unwrap())))
    }
}
```

## SHA-256 Round Function

```rust
pub fn sha256_round<W, C, CS>(
    cs: &mut CS,
    state: &mut [UInt32<W, C>; 8],
    k: UInt32<W, C>,  // Round constant
    w: &UInt32<W, C>, // Message schedule word
) -> Result<(), SynthesisError>
where
    W: Witness,
    C: Coefficient,
    CS: ConstraintSystem<W, C>,
{
    let [a, b, c, d, e, f, g, h] = state;

    // T1 = h + Σ1(e) + Ch(e, f, g) + k + w
    let sum1_e = e.sha256_sum1(cs)?;
    let ch_efg = UInt32::sha256_ch(cs, e, f, g)?;
    let t1 = UInt32::add_many(cs, &[h.clone(), sum1_e, ch_efg, k, w.clone()])?;

    // T2 = Σ0(a) + Maj(a, b, c)
    let sum0_a = a.sha256_sum0(cs)?;
    let maj_abc = UInt32::sha256_maj(cs, a, b, c)?;
    let t2 = UInt32::add_many(cs, &[sum0_a, maj_abc])?;

    // Update state
    *h = g.clone();
    *g = f.clone();
    *f = e.clone();
    *e = UInt32::add_many(cs, &[d.clone(), t1.clone()])?;
    *d = c.clone();
    *c = b.clone();
    *b = a.clone();
    *a = UInt32::add_many(cs, &[t1, t2])?;

    Ok(())
}
```

## Comparison: SmallMultiEq vs Small-Value R1CS

| Aspect | SmallMultiEq (Field) | Small-Value R1CS |
|--------|---------------------|------------------|
| Coefficient type (C) | Field element (32 bytes) | i32 (4 bytes) |
| Witness type (W) | Field element (32 bytes) | i32 (4 bytes) |
| Accumulator type (Acc) | Field element | i64 (8 bytes) |
| Overflow handling | Batched equality constraints | Type-enforced bounds |
| addmany(4) constraints | Variable (batching dependent) | 38 fixed |
| Memory per constraint | ~96 bytes | ~12 bytes |
| Arithmetic ops | Field mul/add (~50 cycles) | Native int (~3 cycles) |

## Integration Notes

1. **SmallMultiEq removal**: The `SmallMultiEq` trait and its implementations (`NoBatchEq`, `BatchingEq`) are no longer needed. Addition is handled directly by `UInt32::add_many()`.

2. **Field conversion**: Convert to field elements only at commitment time using:
   ```rust
   fn signed_to_field<F: PrimeField>(v: i64) -> F {
       if v >= 0 {
           F::from(v as u64)
       } else {
           -F::from((-v) as u64)
       }
   }
   ```

3. **Sumcheck compatibility**: The small-value R1CS integrates with the existing sumcheck protocol. Matrix-vector products use `DelayedReduction` for efficient accumulation.

---

## Testing: Comparison with Bellpepper SHA-256

To verify correctness, the small-value SHA-256 implementation must produce identical outputs to Bellpepper's field-based SHA-256 gadget.

### Test Strategy

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use bellpepper::gadgets::sha256::sha256 as bellpepper_sha256;
    use bellpepper_core::test_cs::TestConstraintSystem;
    use halo2curves::pasta::Fq;
    use sha2::{Digest, Sha256};

    /// Test that small-value SHA-256 matches Bellpepper's SHA-256
    #[test]
    fn test_small_value_sha256_matches_bellpepper() {
        let preimage = b"test input for SHA-256 comparison";

        // ========================================
        // 1. Bellpepper's field-based SHA-256
        // ========================================
        let mut bp_cs = TestConstraintSystem::<Fq>::new();
        let bp_input_bits: Vec<Boolean> = bytes_to_bits(preimage);
        let bp_hash_bits = bellpepper_sha256(&mut bp_cs, &bp_input_bits).unwrap();
        let bp_hash_bytes = bits_to_bytes(&bp_hash_bits);

        assert!(bp_cs.is_satisfied(), "Bellpepper CS not satisfied");

        // ========================================
        // 2. Small-value SHA-256 (W=i32, C=i32)
        // ========================================
        let mut small_cs = SmallCS::<i32, i32>::new();
        let small_input_bits: Vec<Boolean<i32, i32>> = bytes_to_small_bits(preimage);
        let small_hash_bits = small_value_sha256(&mut small_cs, &small_input_bits).unwrap();
        let small_hash_bytes = small_bits_to_bytes(&small_hash_bits);

        // Verify constraint satisfaction using Acc=i64
        let z: Vec<i32> = small_cs.z();
        let shape = small_cs.to_shape();
        let (az, bz, cz): (Vec<i64>, Vec<i64>, Vec<i64>) = shape.multiply_vec(&z);
        assert!(
            az.iter().zip(&bz).zip(&cz).all(|((a, b), c)| {
                (*a as i128) * (*b as i128) == (*c as i128)
            }),
            "Small-value CS not satisfied"
        );

        // ========================================
        // 3. Compare outputs
        // ========================================
        assert_eq!(
            bp_hash_bytes, small_hash_bytes,
            "Bellpepper and small-value SHA-256 outputs differ!"
        );

        // ========================================
        // 4. Also verify against native SHA-256
        // ========================================
        let native_hash = Sha256::digest(preimage);
        assert_eq!(
            &bp_hash_bytes[..], &native_hash[..],
            "Circuit output doesn't match native SHA-256"
        );

        println!("Bellpepper constraints: {}", bp_cs.num_constraints());
        println!("Small-value constraints: {}", small_cs.num_constraints());
    }

    /// Run multiple random tests
    #[test]
    fn test_small_value_sha256_matches_bellpepper_random() {
        use rand::{Rng, SeedableRng, rngs::StdRng};
        let mut rng = StdRng::seed_from_u64(12345);

        for i in 0..16 {
            let len = rng.gen_range(1..=64);
            let preimage: Vec<u8> = (0..len).map(|_| rng.gen()).collect();

            // Bellpepper
            let mut bp_cs = TestConstraintSystem::<Fq>::new();
            let bp_hash = bellpepper_sha256(&mut bp_cs, &bytes_to_bits(&preimage)).unwrap();
            let bp_bytes = bits_to_bytes(&bp_hash);

            // Small-value
            let mut small_cs = SmallCS::<i32, i32>::new();
            let small_hash = small_value_sha256(&mut small_cs, &bytes_to_small_bits(&preimage)).unwrap();
            let small_bytes = small_bits_to_bytes(&small_hash);

            assert_eq!(
                bp_bytes, small_bytes,
                "Mismatch at iteration {}, len {}",
                i, len
            );
        }
    }
}
```

### Expected Results

| Metric | Bellpepper | Small-Value |
|--------|------------|-------------|
| Hash output | Identical | Identical |
| Constraint count | ~25,000 | ~25,000 (similar) |
| Memory usage | ~2.4 MB | ~300 KB |
| Synthesis time | Baseline | ~10x faster |
| Sumcheck time | Baseline | ~15x faster |

The constraint count should be similar because the circuit structure is the same. The performance improvements come from:
1. **Synthesis**: Native i32 arithmetic vs field arithmetic
2. **Memory**: 4-byte coefficients + 4-byte witnesses vs 32-byte field elements
3. **Sumcheck**: i64 accumulation with `DelayedReduction` vs field operations
