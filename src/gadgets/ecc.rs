//! Elliptic curve gadgets for in-circuit point operations
//!
//! This module provides `AllocatedPoint` and `AllocatedPointNonInfinity` structs
//! for performing elliptic curve arithmetic inside R1CS circuits.

#![allow(non_snake_case)]

use crate::{
  gadgets::utils::{
    alloc_num_equals, alloc_one, alloc_zero, conditionally_select, conditionally_select2,
    select_num_or_one, select_num_or_zero, select_num_or_zero2, select_one_or_diff2,
    select_one_or_num2, select_zero_or_num2,
  },
  traits::{Engine, Group},
};
use bellpepper_core::{
  boolean::{AllocatedBit, Boolean},
  num::AllocatedNum,
  ConstraintSystem, SynthesisError,
};
use ff::{Field, PrimeField};

/// `AllocatedPoint` provides an elliptic curve abstraction inside a circuit.
#[derive(Clone)]
pub struct AllocatedPoint<E: Engine> {
  /// The x-coordinate of the point.
  pub x: AllocatedNum<E::Base>,
  /// The y-coordinate of the point.
  pub y: AllocatedNum<E::Base>,
  /// Flag indicating if this is the point at infinity (1 = infinity, 0 = not infinity).
  pub is_infinity: AllocatedNum<E::Base>,
}

impl<E> AllocatedPoint<E>
where
  E: Engine,
{
  /// Allocates a new point on the curve using coordinates provided by `coords`.
  /// If coords = None, it allocates the default infinity point
  pub fn alloc<CS: ConstraintSystem<E::Base>>(
    mut cs: CS,
    coords: Option<(E::Base, E::Base, bool)>,
  ) -> Result<Self, SynthesisError> {
    let x = AllocatedNum::alloc(cs.namespace(|| "x"), || {
      Ok(coords.map_or(E::Base::ZERO, |c| c.0))
    })?;
    let y = AllocatedNum::alloc(cs.namespace(|| "y"), || {
      Ok(coords.map_or(E::Base::ZERO, |c| c.1))
    })?;
    let is_infinity = AllocatedNum::alloc(cs.namespace(|| "is_infinity"), || {
      Ok(if coords.map_or(true, |c| c.2) {
        E::Base::ONE
      } else {
        E::Base::ZERO
      })
    })?;
    cs.enforce(
      || "is_infinity is bit",
      |lc| lc + is_infinity.get_variable(),
      |lc| lc + CS::one() - is_infinity.get_variable(),
      |lc| lc,
    );

    Ok(AllocatedPoint { x, y, is_infinity })
  }

  /// checks if `self` is on the curve or if it is infinity
  pub fn check_on_curve<CS>(&self, mut cs: CS) -> Result<(), SynthesisError>
  where
    CS: ConstraintSystem<E::Base>,
  {
    // check that (x,y) is on the curve if it is not infinity
    // we will check that (1- is_infinity) * y^2 = (1-is_infinity) * (x^3 + Ax + B)
    // note that is_infinity is already restricted to be in the set {0, 1}
    let y_square = self.y.square(cs.namespace(|| "y_square"))?;
    let x_square = self.x.square(cs.namespace(|| "x_square"))?;
    let x_cube = self.x.mul(cs.namespace(|| "x_cube"), &x_square)?;

    let (a, b, _, _) = E::GE::group_params();
    // Optimization: check if a == 0 (common for BN254, Grumpkin, Pallas, Vesta)
    let a_is_zero = a == E::Base::ZERO;

    let rhs = AllocatedNum::alloc(cs.namespace(|| "rhs"), || {
      let is_inf = self
        .is_infinity
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?;
      if is_inf == E::Base::ONE {
        Ok(E::Base::ZERO)
      } else {
        let x_cube_val = x_cube
          .get_value()
          .ok_or(SynthesisError::AssignmentMissing)?;
        if a_is_zero {
          Ok(x_cube_val + b)
        } else {
          let x_val = self.x.get_value().ok_or(SynthesisError::AssignmentMissing)?;
          Ok(x_cube_val + x_val * a + b)
        }
      }
    })?;

    // When a=0, use simpler constraint without the Ax term
    if a_is_zero {
      cs.enforce(
        || "rhs = (1-is_infinity) * (x^3 + B)",
        |lc| lc + x_cube.get_variable() + (b, CS::one()),
        |lc| lc + CS::one() - self.is_infinity.get_variable(),
        |lc| lc + rhs.get_variable(),
      );
    } else {
      cs.enforce(
        || "rhs = (1-is_infinity) * (x^3 + Ax + B)",
        |lc| lc + x_cube.get_variable() + (a, self.x.get_variable()) + (b, CS::one()),
        |lc| lc + CS::one() - self.is_infinity.get_variable(),
        |lc| lc + rhs.get_variable(),
      );
    }

    // check that (1-infinity) * y_square = rhs
    cs.enforce(
      || "check that y_square * (1 - is_infinity) = rhs",
      |lc| lc + y_square.get_variable(),
      |lc| lc + CS::one() - self.is_infinity.get_variable(),
      |lc| lc + rhs.get_variable(),
    );

    Ok(())
  }

  /// Allocates a default point on the curve, set to the identity point.
  pub fn default<CS: ConstraintSystem<E::Base>>(mut cs: CS) -> Result<Self, SynthesisError> {
    let zero = alloc_zero(cs.namespace(|| "zero"));
    let one = alloc_one(cs.namespace(|| "one"));

    Ok(AllocatedPoint {
      x: zero.clone(),
      y: zero,
      is_infinity: one,
    })
  }

  /// Negates the provided point
  pub fn negate<CS: ConstraintSystem<E::Base>>(&self, mut cs: CS) -> Result<Self, SynthesisError> {
    let y = AllocatedNum::alloc(cs.namespace(|| "y"), || {
      let y_val = self.y.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      Ok(-y_val)
    })?;

    cs.enforce(
      || "check y = - self.y",
      |lc| lc + self.y.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc - y.get_variable(),
    );

    Ok(Self {
      x: self.x.clone(),
      y,
      is_infinity: self.is_infinity.clone(),
    })
  }

  /// Add two points (may be equal)
  pub fn add<CS: ConstraintSystem<E::Base>>(
    &self,
    mut cs: CS,
    other: &AllocatedPoint<E>,
  ) -> Result<Self, SynthesisError> {
    // Compute boolean equal indicating if self = other

    let equal_x = alloc_num_equals(
      cs.namespace(|| "check self.x == other.x"),
      &self.x,
      &other.x,
    )?;

    let equal_y = alloc_num_equals(
      cs.namespace(|| "check self.y == other.y"),
      &self.y,
      &other.y,
    )?;

    // Compute the result of the addition and the result of double self
    let result_from_add = self.add_internal(cs.namespace(|| "add internal"), other, &equal_x)?;
    let result_from_double = self.double(cs.namespace(|| "double"))?;

    // Output:
    // If (self == other) {
    //  return double(self)
    // }else {
    //  if (self.x == other.x){
    //      return infinity [negation]
    //  } else {
    //      return add(self, other)
    //  }
    // }
    let result_for_equal_x = AllocatedPoint::select_point_or_infinity(
      cs.namespace(|| "equal_y ? result_from_double : infinity"),
      &result_from_double,
      &Boolean::from(equal_y),
    )?;

    AllocatedPoint::conditionally_select(
      cs.namespace(|| "equal ? result_from_double : result_from_add"),
      &result_for_equal_x,
      &result_from_add,
      &Boolean::from(equal_x),
    )
  }

  /// Adds other point to this point and returns the result. Assumes that the two points are
  /// different and that both `other.is_infinity` and `this.is_infinity` are bits
  pub fn add_internal<CS: ConstraintSystem<E::Base>>(
    &self,
    mut cs: CS,
    other: &AllocatedPoint<E>,
    equal_x: &AllocatedBit,
  ) -> Result<Self, SynthesisError> {
    //************************************************************************/
    // lambda = (other.y - self.y) * (other.x - self.x).invert().unwrap();
    //************************************************************************/
    // First compute (other.x - self.x).inverse()
    // If either self or other are the infinity point or self.x = other.x  then compute bogus values
    // Specifically,
    // x_diff = self != inf && other != inf && self.x == other.x ? (other.x - self.x) : 1

    // Compute self.is_infinity OR other.is_infinity =
    // NOT(NOT(self.is_ifninity) AND NOT(other.is_infinity))
    let at_least_one_inf = AllocatedNum::alloc(cs.namespace(|| "at least one inf"), || {
      let self_inf = self
        .is_infinity
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?;
      let other_inf = other
        .is_infinity
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?;
      Ok(E::Base::ONE - (E::Base::ONE - self_inf) * (E::Base::ONE - other_inf))
    })?;
    cs.enforce(
      || "1 - at least one inf = (1-self.is_infinity) * (1-other.is_infinity)",
      |lc| lc + CS::one() - self.is_infinity.get_variable(),
      |lc| lc + CS::one() - other.is_infinity.get_variable(),
      |lc| lc + CS::one() - at_least_one_inf.get_variable(),
    );

    // Now compute x_diff_is_actual = at_least_one_inf OR equal_x
    let x_diff_is_actual =
      AllocatedNum::alloc(cs.namespace(|| "allocate x_diff_is_actual"), || {
        let eq_x = equal_x
          .get_value()
          .ok_or(SynthesisError::AssignmentMissing)?;
        if eq_x {
          Ok(E::Base::ONE)
        } else {
          at_least_one_inf
            .get_value()
            .ok_or(SynthesisError::AssignmentMissing)
        }
      })?;
    cs.enforce(
      || "1 - x_diff_is_actual = (1-equal_x) * (1-at_least_one_inf)",
      |lc| lc + CS::one() - at_least_one_inf.get_variable(),
      |lc| lc + CS::one() - equal_x.get_variable(),
      |lc| lc + CS::one() - x_diff_is_actual.get_variable(),
    );

    // x_diff = 1 if either self.is_infinity or other.is_infinity or self.x = other.x else self.x -
    // other.x
    let x_diff = select_one_or_diff2(
      cs.namespace(|| "Compute x_diff"),
      &other.x,
      &self.x,
      &x_diff_is_actual,
    )?;

    let lambda = AllocatedNum::alloc(cs.namespace(|| "lambda"), || {
      let x_diff_is_actual_val = x_diff_is_actual
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?;
      let x_diff_inv = if x_diff_is_actual_val == E::Base::ONE {
        // Set to default
        E::Base::ONE
      } else {
        // Set to the actual inverse
        let other_x = other.x.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        let self_x = self.x.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        (other_x - self_x).invert().unwrap()
      };

      let other_y = other.y.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let self_y = self.y.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      Ok((other_y - self_y) * x_diff_inv)
    })?;
    cs.enforce(
      || "Check that lambda is correct",
      |lc| lc + lambda.get_variable(),
      |lc| lc + x_diff.get_variable(),
      |lc| lc + other.y.get_variable() - self.y.get_variable(),
    );

    //************************************************************************/
    // x = lambda * lambda - self.x - other.x;
    //************************************************************************/
    let x = AllocatedNum::alloc(cs.namespace(|| "x"), || {
      let lambda_val = lambda.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let self_x = self.x.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let other_x = other.x.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      Ok(lambda_val * lambda_val - self_x - other_x)
    })?;
    cs.enforce(
      || "check that x is correct",
      |lc| lc + lambda.get_variable(),
      |lc| lc + lambda.get_variable(),
      |lc| lc + x.get_variable() + self.x.get_variable() + other.x.get_variable(),
    );

    //************************************************************************/
    // y = lambda * (self.x - x) - self.y;
    //************************************************************************/
    let y = AllocatedNum::alloc(cs.namespace(|| "y"), || {
      let lambda_val = lambda.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let self_x = self.x.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let x_val = x.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let self_y = self.y.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      Ok(lambda_val * (self_x - x_val) - self_y)
    })?;

    cs.enforce(
      || "Check that y is correct",
      |lc| lc + lambda.get_variable(),
      |lc| lc + self.x.get_variable() - x.get_variable(),
      |lc| lc + y.get_variable() + self.y.get_variable(),
    );

    //************************************************************************/
    // We only return the computed x, y if neither of the points is infinity and self.x != other.y
    // if self.is_infinity return other.clone()
    // elif other.is_infinity return self.clone()
    // elif self.x == other.x return infinity
    // Otherwise return the computed points.
    //************************************************************************/
    // Now compute the output x

    let x1 = conditionally_select2(
      cs.namespace(|| "x1 = other.is_infinity ? self.x : x"),
      &self.x,
      &x,
      &other.is_infinity,
    )?;

    let x = conditionally_select2(
      cs.namespace(|| "x = self.is_infinity ? other.x : x1"),
      &other.x,
      &x1,
      &self.is_infinity,
    )?;

    let y1 = conditionally_select2(
      cs.namespace(|| "y1 = other.is_infinity ? self.y : y"),
      &self.y,
      &y,
      &other.is_infinity,
    )?;

    let y = conditionally_select2(
      cs.namespace(|| "y = self.is_infinity ? other.y : y1"),
      &other.y,
      &y1,
      &self.is_infinity,
    )?;

    let is_infinity1 = select_num_or_zero2(
      cs.namespace(|| "is_infinity1 = other.is_infinity ? self.is_infinity : 0"),
      &self.is_infinity,
      &other.is_infinity,
    )?;

    let is_infinity = conditionally_select2(
      cs.namespace(|| "is_infinity = self.is_infinity ? other.is_infinity : is_infinity1"),
      &other.is_infinity,
      &is_infinity1,
      &self.is_infinity,
    )?;

    Ok(Self { x, y, is_infinity })
  }

  /// Doubles the supplied point.
  pub fn double<CS: ConstraintSystem<E::Base>>(&self, mut cs: CS) -> Result<Self, SynthesisError> {
    //*************************************************************/
    // lambda = (E::Base::from(3) * self.x * self.x + E::GE::A())
    //  * (E::Base::from(2)) * self.y).invert().unwrap();
    /*************************************************************/

    let (a, _, _, _) = E::GE::group_params();
    // Optimization: check if a == 0 (common for BN254, Grumpkin, Pallas, Vesta)
    let a_is_zero = a == E::Base::ZERO;

    // Compute tmp = (E::Base::ONE + E::Base::ONE)* self.y ? self != inf : 1
    let tmp_actual = AllocatedNum::alloc(cs.namespace(|| "tmp_actual"), || {
      let self_y = self.y.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      Ok(self_y + self_y)
    })?;
    cs.enforce(
      || "check tmp_actual",
      |lc| lc + CS::one() + CS::one(),
      |lc| lc + self.y.get_variable(),
      |lc| lc + tmp_actual.get_variable(),
    );

    let tmp = select_one_or_num2(cs.namespace(|| "tmp"), &tmp_actual, &self.is_infinity)?;

    // Now compute lambda as (E::Base::from(3) * self.x * self.x + E::GE::A()) * tmp_inv

    let prod_1 = AllocatedNum::alloc(cs.namespace(|| "alloc prod 1"), || {
      let self_x = self.x.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      Ok(E::Base::from(3) * self_x * self_x)
    })?;
    cs.enforce(
      || "Check prod 1",
      |lc| lc + (E::Base::from(3), self.x.get_variable()),
      |lc| lc + self.x.get_variable(),
      |lc| lc + prod_1.get_variable(),
    );

    let lambda = AllocatedNum::alloc(cs.namespace(|| "alloc lambda"), || {
      let is_inf = self
        .is_infinity
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?;
      let tmp_inv = if is_inf == E::Base::ONE {
        // Return default value 1
        E::Base::ONE
      } else {
        // Return the actual inverse
        let tmp_val = tmp.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        tmp_val.invert().unwrap()
      };

      let prod_1_val = prod_1.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      if a_is_zero {
        Ok(tmp_inv * prod_1_val)
      } else {
        Ok(tmp_inv * (prod_1_val + a))
      }
    })?;

    // When a=0, use simpler constraint: tmp * lambda = 3x²
    if a_is_zero {
      cs.enforce(
        || "Check lambda",
        |lc| lc + tmp.get_variable(),
        |lc| lc + lambda.get_variable(),
        |lc| lc + prod_1.get_variable(),
      );
    } else {
      cs.enforce(
        || "Check lambda",
        |lc| lc + tmp.get_variable(),
        |lc| lc + lambda.get_variable(),
        |lc| lc + prod_1.get_variable() + (a, CS::one()),
      );
    }

    /*************************************************************/
    //          x = lambda * lambda - self.x - self.x;
    /*************************************************************/

    let x = AllocatedNum::alloc(cs.namespace(|| "x"), || {
      let lambda_val = lambda.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let self_x = self.x.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      Ok((lambda_val * lambda_val) - self_x - self_x)
    })?;
    cs.enforce(
      || "Check x",
      |lc| lc + lambda.get_variable(),
      |lc| lc + lambda.get_variable(),
      |lc| lc + x.get_variable() + self.x.get_variable() + self.x.get_variable(),
    );

    /*************************************************************/
    //        y = lambda * (self.x - x) - self.y;
    /*************************************************************/

    let y = AllocatedNum::alloc(cs.namespace(|| "y"), || {
      let lambda_val = lambda.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let self_x = self.x.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let x_val = x.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let self_y = self.y.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      Ok(lambda_val * (self_x - x_val) - self_y)
    })?;
    cs.enforce(
      || "Check y",
      |lc| lc + lambda.get_variable(),
      |lc| lc + self.x.get_variable() - x.get_variable(),
      |lc| lc + y.get_variable() + self.y.get_variable(),
    );

    /*************************************************************/
    // Only return the computed x and y if the point is not infinity
    /*************************************************************/

    // x
    let x = select_zero_or_num2(cs.namespace(|| "final x"), &x, &self.is_infinity)?;

    // y
    let y = select_zero_or_num2(cs.namespace(|| "final y"), &y, &self.is_infinity)?;

    // is_infinity
    let is_infinity = self.is_infinity.clone();

    Ok(Self { x, y, is_infinity })
  }

  /// A gadget for scalar multiplication, optimized to use incomplete addition law.
  /// The optimization here is analogous to <https://github.com/arkworks-rs/r1cs-std/blob/6d64f379a27011b3629cf4c9cb38b7b7b695d5a0/src/groups/curves/short_weierstrass/mod.rs#L295>,
  /// except we use complete addition law over affine coordinates instead of projective coordinates for the tail bits
  pub fn scalar_mul<CS: ConstraintSystem<E::Base>>(
    &self,
    mut cs: CS,
    scalar_bits: &[AllocatedBit],
  ) -> Result<Self, SynthesisError> {
    let split_len = core::cmp::min(scalar_bits.len(), (E::Base::NUM_BITS - 2) as usize);
    let (incomplete_bits, complete_bits) = scalar_bits.split_at(split_len);

    // we convert AllocatedPoint into AllocatedPointNonInfinity; we deal with the case where self.is_infinity = 1 below
    let mut p = AllocatedPointNonInfinity::from_allocated_point(self);

    // we assume the first bit to be 1, so we must initialize acc to self and double it
    // we remove this assumption below
    let mut acc = p;
    p = acc.double_incomplete(cs.namespace(|| "double"))?;

    // perform the double-and-add loop to compute the scalar mul using incomplete addition law
    for (i, bit) in incomplete_bits.iter().enumerate().skip(1) {
      let temp = acc.add_incomplete(cs.namespace(|| format!("add {i}")), &p)?;
      acc = AllocatedPointNonInfinity::conditionally_select(
        cs.namespace(|| format!("acc_iteration_{i}")),
        &temp,
        &acc,
        &Boolean::from(bit.clone()),
      )?;

      p = p.double_incomplete(cs.namespace(|| format!("double {i}")))?;
    }

    // convert back to AllocatedPoint
    let res = {
      // we set acc.is_infinity = self.is_infinity
      let acc = acc.to_allocated_point(&self.is_infinity)?;

      // we remove the initial slack if bits[0] is as not as assumed (i.e., it is not 1)
      let acc_minus_initial = {
        let neg = self.negate(cs.namespace(|| "negate"))?;
        acc.add(cs.namespace(|| "res minus self"), &neg)
      }?;

      AllocatedPoint::conditionally_select(
        cs.namespace(|| "remove slack if necessary"),
        &acc,
        &acc_minus_initial,
        &Boolean::from(scalar_bits[0].clone()),
      )?
    };

    // when self.is_infinity = 1, return the default point, else return res
    // we already set res.is_infinity to be self.is_infinity, so we do not need to set it here
    let default = Self::default(cs.namespace(|| "default"))?;
    let x = conditionally_select2(
      cs.namespace(|| "check if self.is_infinity is zero (x)"),
      &default.x,
      &res.x,
      &self.is_infinity,
    )?;

    let y = conditionally_select2(
      cs.namespace(|| "check if self.is_infinity is zero (y)"),
      &default.y,
      &res.y,
      &self.is_infinity,
    )?;

    // we now perform the remaining scalar mul using complete addition law
    let mut acc = AllocatedPoint {
      x,
      y,
      is_infinity: res.is_infinity,
    };
    let mut p_complete = p.to_allocated_point(&self.is_infinity)?;

    for (i, bit) in complete_bits.iter().enumerate() {
      let temp = acc.add(cs.namespace(|| format!("add_complete {i}")), &p_complete)?;
      acc = AllocatedPoint::conditionally_select(
        cs.namespace(|| format!("acc_complete_iteration_{i}")),
        &temp,
        &acc,
        &Boolean::from(bit.clone()),
      )?;

      p_complete = p_complete.double(cs.namespace(|| format!("double_complete {i}")))?;
    }

    Ok(acc)
  }

  /// If condition outputs a otherwise outputs b
  pub fn conditionally_select<CS: ConstraintSystem<E::Base>>(
    mut cs: CS,
    a: &Self,
    b: &Self,
    condition: &Boolean,
  ) -> Result<Self, SynthesisError> {
    let x = conditionally_select(cs.namespace(|| "select x"), &a.x, &b.x, condition)?;

    let y = conditionally_select(cs.namespace(|| "select y"), &a.y, &b.y, condition)?;

    let is_infinity = conditionally_select(
      cs.namespace(|| "select is_infinity"),
      &a.is_infinity,
      &b.is_infinity,
      condition,
    )?;

    Ok(Self { x, y, is_infinity })
  }

  /// If condition outputs a otherwise infinity
  pub fn select_point_or_infinity<CS: ConstraintSystem<E::Base>>(
    mut cs: CS,
    a: &Self,
    condition: &Boolean,
  ) -> Result<Self, SynthesisError> {
    let x = select_num_or_zero(cs.namespace(|| "select x"), &a.x, condition)?;

    let y = select_num_or_zero(cs.namespace(|| "select y"), &a.y, condition)?;

    let is_infinity = select_num_or_one(
      cs.namespace(|| "select is_infinity"),
      &a.is_infinity,
      condition,
    )?;

    Ok(Self { x, y, is_infinity })
  }

  /// Enforce that self equals other.
  pub fn enforce_equal<CS: ConstraintSystem<E::Base>>(
    &self,
    mut cs: CS,
    other: &Self,
  ) -> Result<(), SynthesisError> {
    cs.enforce(
      || "check x equality",
      |lc| lc + self.x.get_variable() - other.x.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc,
    );
    cs.enforce(
      || "check y equality",
      |lc| lc + self.y.get_variable() - other.y.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc,
    );
    cs.enforce(
      || "check is_inf equality",
      |lc| lc + self.is_infinity.get_variable() - other.is_infinity.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc,
    );

    Ok(())
  }

  /// Add a constant point (known at circuit compile time) to this point.
  /// This is more efficient than `add` because we don't need to allocate
  /// variables for the constant point's coordinates.
  ///
  /// The constant point is assumed to not be at infinity.
  /// Handles the case where self is at infinity (returns the constant).
  /// Assumes self != constant and self != -constant when self is not infinity.
  pub fn add_constant<CS: ConstraintSystem<E::Base>>(
    &self,
    mut cs: CS,
    constant: (E::Base, E::Base), // (x, y) of constant point (not infinity)
  ) -> Result<Self, SynthesisError> {
    let (other_x, other_y) = constant;

    // lambda = (other_y - self.y) / (other_x - self.x)
    // When self.is_infinity = 1, we use bogus values (set denominator to 1)
    let lambda = AllocatedNum::alloc(cs.namespace(|| "lambda"), || {
      let is_inf = self
        .is_infinity
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?;
      if is_inf == E::Base::ONE {
        Ok(E::Base::ONE) // bogus value when self is infinity
      } else {
        let self_x = self.x.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        let self_y = self.y.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        Ok((other_y - self_y) * (other_x - self_x).invert().unwrap())
      }
    })?;

    // x_diff = is_infinity ? 1 : (other_x - self.x)
    let x_diff = AllocatedNum::alloc(cs.namespace(|| "x_diff"), || {
      let is_inf = self
        .is_infinity
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?;
      if is_inf == E::Base::ONE {
        Ok(E::Base::ONE)
      } else {
        let self_x = self.x.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        Ok(other_x - self_x)
      }
    })?;

    // Constraint: x_diff = is_infinity * 1 + (1 - is_infinity) * (other_x - self.x)
    //           = is_infinity + (other_x - self.x) - is_infinity * (other_x - self.x)
    // Rearranged: x_diff - is_infinity - other_x + self.x = -is_infinity * (other_x - self.x - 1)
    // Simpler: (1 - is_infinity) * (other_x - self.x) + is_infinity = x_diff
    cs.enforce(
      || "x_diff = is_infinity ? 1 : (other_x - self.x)",
      |lc| lc + (other_x, CS::one()) - self.x.get_variable() - CS::one(),
      |lc| lc + CS::one() - self.is_infinity.get_variable(),
      |lc| lc + x_diff.get_variable() - CS::one(),
    );

    // Constraint: lambda * x_diff = other_y - self.y (when not infinity)
    // But when is_infinity=1, x_diff=1 and lambda can be anything, so we need:
    // lambda * x_diff = (1 - is_infinity) * (other_y - self.y)
    let y_diff_if_not_inf = AllocatedNum::alloc(cs.namespace(|| "y_diff_if_not_inf"), || {
      let is_inf = self
        .is_infinity
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?;
      if is_inf == E::Base::ONE {
        Ok(E::Base::ZERO)
      } else {
        let self_y = self.y.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        Ok(other_y - self_y)
      }
    })?;

    cs.enforce(
      || "y_diff_if_not_inf = (1 - is_infinity) * (other_y - self.y)",
      |lc| lc + CS::one() - self.is_infinity.get_variable(),
      |lc| lc + (other_y, CS::one()) - self.y.get_variable(),
      |lc| lc + y_diff_if_not_inf.get_variable(),
    );

    cs.enforce(
      || "lambda * x_diff = y_diff_if_not_inf",
      |lc| lc + lambda.get_variable(),
      |lc| lc + x_diff.get_variable(),
      |lc| lc + y_diff_if_not_inf.get_variable(),
    );

    // x_result = lambda² - self.x - other_x
    let x_computed = AllocatedNum::alloc(cs.namespace(|| "x_computed"), || {
      let lambda_val = lambda.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let self_x = self.x.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      Ok(lambda_val * lambda_val - self_x - other_x)
    })?;
    cs.enforce(
      || "x_computed = lambda² - self.x - other_x",
      |lc| lc + lambda.get_variable(),
      |lc| lc + lambda.get_variable(),
      |lc| lc + x_computed.get_variable() + self.x.get_variable() + (other_x, CS::one()),
    );

    // y_computed = lambda * (self.x - x_computed) - self.y
    let y_computed = AllocatedNum::alloc(cs.namespace(|| "y_computed"), || {
      let lambda_val = lambda.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let self_x = self.x.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let x_val = x_computed.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let self_y = self.y.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      Ok(lambda_val * (self_x - x_val) - self_y)
    })?;

    cs.enforce(
      || "y_computed = lambda * (self.x - x_computed) - self.y",
      |lc| lc + lambda.get_variable(),
      |lc| lc + self.x.get_variable() - x_computed.get_variable(),
      |lc| lc + y_computed.get_variable() + self.y.get_variable(),
    );

    // Final result: if self.is_infinity, return constant, else return computed
    // Allocate constants for the select
    let other_x_var = AllocatedNum::alloc(cs.namespace(|| "other_x_alloc"), || Ok(other_x))?;
    let other_y_var = AllocatedNum::alloc(cs.namespace(|| "other_y_alloc"), || Ok(other_y))?;

    let x = conditionally_select2(
      cs.namespace(|| "x = is_infinity ? other_x : x_computed"),
      &other_x_var,
      &x_computed,
      &self.is_infinity,
    )?;

    let y = conditionally_select2(
      cs.namespace(|| "y = is_infinity ? other_y : y_computed"),
      &other_y_var,
      &y_computed,
      &self.is_infinity,
    )?;

    // Result is never infinity (constant is not infinity, and if self was infinity we return constant)
    let is_infinity = alloc_zero(cs.namespace(|| "is_infinity = 0"));

    Ok(Self { x, y, is_infinity })
  }
}

#[derive(Clone)]
/// `AllocatedPoint` but one that is guaranteed to be not infinity
pub struct AllocatedPointNonInfinity<E: Engine> {
  /// The x-coordinate of the point.
  pub x: AllocatedNum<E::Base>,
  /// The y-coordinate of the point.
  pub y: AllocatedNum<E::Base>,
}

impl<E: Engine> AllocatedPointNonInfinity<E> {
  /// Turns an `AllocatedPoint` into an `AllocatedPointNonInfinity` (assumes it is not infinity)
  pub fn from_allocated_point(p: &AllocatedPoint<E>) -> Self {
    Self {
      x: p.x.clone(),
      y: p.y.clone(),
    }
  }

  /// Returns an `AllocatedPoint` from an `AllocatedPointNonInfinity`
  pub fn to_allocated_point(
    &self,
    is_infinity: &AllocatedNum<E::Base>,
  ) -> Result<AllocatedPoint<E>, SynthesisError> {
    Ok(AllocatedPoint {
      x: self.x.clone(),
      y: self.y.clone(),
      is_infinity: is_infinity.clone(),
    })
  }

  /// Add two points assuming self != +/- other
  pub fn add_incomplete<CS>(&self, mut cs: CS, other: &Self) -> Result<Self, SynthesisError>
  where
    CS: ConstraintSystem<E::Base>,
  {
    // allocate a free variable that an honest prover sets to lambda = (y2-y1)/(x2-x1)
    let lambda = AllocatedNum::alloc(cs.namespace(|| "lambda"), || {
      let other_x = other.x.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let self_x = self.x.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      if other_x == self_x {
        Ok(E::Base::ONE)
      } else {
        let other_y = other.y.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        let self_y = self.y.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        Ok((other_y - self_y) * (other_x - self_x).invert().unwrap())
      }
    })?;
    cs.enforce(
      || "Check that lambda is computed correctly",
      |lc| lc + lambda.get_variable(),
      |lc| lc + other.x.get_variable() - self.x.get_variable(),
      |lc| lc + other.y.get_variable() - self.y.get_variable(),
    );

    //************************************************************************/
    // x = lambda * lambda - self.x - other.x;
    //************************************************************************/
    let x = AllocatedNum::alloc(cs.namespace(|| "x"), || {
      let lambda_val = lambda.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let self_x = self.x.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let other_x = other.x.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      Ok(lambda_val * lambda_val - self_x - other_x)
    })?;
    cs.enforce(
      || "check that x is correct",
      |lc| lc + lambda.get_variable(),
      |lc| lc + lambda.get_variable(),
      |lc| lc + x.get_variable() + self.x.get_variable() + other.x.get_variable(),
    );

    //************************************************************************/
    // y = lambda * (self.x - x) - self.y;
    //************************************************************************/
    let y = AllocatedNum::alloc(cs.namespace(|| "y"), || {
      let lambda_val = lambda.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let self_x = self.x.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let x_val = x.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let self_y = self.y.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      Ok(lambda_val * (self_x - x_val) - self_y)
    })?;

    cs.enforce(
      || "Check that y is correct",
      |lc| lc + lambda.get_variable(),
      |lc| lc + self.x.get_variable() - x.get_variable(),
      |lc| lc + y.get_variable() + self.y.get_variable(),
    );

    Ok(Self { x, y })
  }

  /// doubles the point; since this is called with a point not at infinity, it is guaranteed to be not infinity
  pub fn double_incomplete<CS: ConstraintSystem<E::Base>>(
    &self,
    mut cs: CS,
  ) -> Result<Self, SynthesisError> {
    // lambda = (3 x^2 + a) / 2 * y
    let (a, _, _, _) = E::GE::group_params();
    // Optimization: check if a == 0 (common for BN254, Grumpkin, Pallas, Vesta)
    let a_is_zero = a == E::Base::ZERO;

    let x_sq = self.x.square(cs.namespace(|| "x_sq"))?;

    let lambda = AllocatedNum::alloc(cs.namespace(|| "lambda"), || {
      let x_sq_val = x_sq.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let self_y = self.y.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let n = if a_is_zero {
        E::Base::from(3) * x_sq_val
      } else {
        E::Base::from(3) * x_sq_val + a
      };
      let d = E::Base::from(2) * self_y;
      if d == E::Base::ZERO {
        Ok(E::Base::ONE)
      } else {
        Ok(n * d.invert().unwrap())
      }
    })?;

    // When a=0, use simpler constraint: lambda * 2y = 3x²
    if a_is_zero {
      cs.enforce(
        || "Check that lambda is computed correctly",
        |lc| lc + lambda.get_variable(),
        |lc| lc + (E::Base::from(2), self.y.get_variable()),
        |lc| lc + (E::Base::from(3), x_sq.get_variable()),
      );
    } else {
      cs.enforce(
        || "Check that lambda is computed correctly",
        |lc| lc + lambda.get_variable(),
        |lc| lc + (E::Base::from(2), self.y.get_variable()),
        |lc| lc + (E::Base::from(3), x_sq.get_variable()) + (a, CS::one()),
      );
    }

    let x = AllocatedNum::alloc(cs.namespace(|| "x"), || {
      let lambda_val = lambda.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let self_x = self.x.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      Ok(lambda_val * lambda_val - self_x - self_x)
    })?;

    cs.enforce(
      || "check that x is correct",
      |lc| lc + lambda.get_variable(),
      |lc| lc + lambda.get_variable(),
      |lc| lc + x.get_variable() + (E::Base::from(2), self.x.get_variable()),
    );

    let y = AllocatedNum::alloc(cs.namespace(|| "y"), || {
      let lambda_val = lambda.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let self_x = self.x.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let x_val = x.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let self_y = self.y.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      Ok(lambda_val * (self_x - x_val) - self_y)
    })?;

    cs.enforce(
      || "Check that y is correct",
      |lc| lc + lambda.get_variable(),
      |lc| lc + self.x.get_variable() - x.get_variable(),
      |lc| lc + y.get_variable() + self.y.get_variable(),
    );

    Ok(Self { x, y })
  }

  /// Add a constant point (known at circuit compile time) to this point.
  /// This is more efficient than `add_incomplete` because we don't need to allocate
  /// variables for the constant point's coordinates - they go directly into the
  /// linear combinations.
  ///
  /// Assumes self != constant and self != -constant (incomplete addition).
  pub fn add_constant<CS: ConstraintSystem<E::Base>>(
    &self,
    mut cs: CS,
    constant: (E::Base, E::Base), // (x, y) of constant point
  ) -> Result<Self, SynthesisError> {
    let (other_x, other_y) = constant;

    // lambda = (other_y - self.y) / (other_x - self.x)
    let lambda = AllocatedNum::alloc(cs.namespace(|| "lambda"), || {
      let self_x = self.x.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let self_y = self.y.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      if other_x == self_x {
        Ok(E::Base::ONE)
      } else {
        Ok((other_y - self_y) * (other_x - self_x).invert().unwrap())
      }
    })?;

    // Constraint: lambda * (other_x - self.x) = other_y - self.y
    // Using constants directly in the linear combination
    cs.enforce(
      || "Check that lambda is computed correctly",
      |lc| lc + lambda.get_variable(),
      |lc| lc + (other_x, CS::one()) - self.x.get_variable(),
      |lc| lc + (other_y, CS::one()) - self.y.get_variable(),
    );

    // x = lambda² - self.x - other_x
    let x = AllocatedNum::alloc(cs.namespace(|| "x"), || {
      let lambda_val = lambda.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let self_x = self.x.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      Ok(lambda_val * lambda_val - self_x - other_x)
    })?;
    cs.enforce(
      || "check that x is correct",
      |lc| lc + lambda.get_variable(),
      |lc| lc + lambda.get_variable(),
      |lc| lc + x.get_variable() + self.x.get_variable() + (other_x, CS::one()),
    );

    // y = lambda * (self.x - x) - self.y
    let y = AllocatedNum::alloc(cs.namespace(|| "y"), || {
      let lambda_val = lambda.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let self_x = self.x.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let x_val = x.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let self_y = self.y.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      Ok(lambda_val * (self_x - x_val) - self_y)
    })?;

    cs.enforce(
      || "Check that y is correct",
      |lc| lc + lambda.get_variable(),
      |lc| lc + self.x.get_variable() - x.get_variable(),
      |lc| lc + y.get_variable() + self.y.get_variable(),
    );

    Ok(Self { x, y })
  }

  /// If condition outputs a otherwise outputs b
  pub fn conditionally_select<CS: ConstraintSystem<E::Base>>(
    mut cs: CS,
    a: &Self,
    b: &Self,
    condition: &Boolean,
  ) -> Result<Self, SynthesisError> {
    let x = conditionally_select(cs.namespace(|| "select x"), &a.x, &b.x, condition)?;
    let y = conditionally_select(cs.namespace(|| "select y"), &a.y, &b.y, condition)?;

    Ok(Self { x, y })
  }

  /// Allocate a new point from coordinates.
  pub fn alloc<CS: ConstraintSystem<E::Base>>(
    mut cs: CS,
    coords: Option<(E::Base, E::Base)>,
  ) -> Result<AllocatedPointNonInfinity<E>, SynthesisError> {
    let x = AllocatedNum::alloc(cs.namespace(|| "x"), || {
      coords.map_or(Err(SynthesisError::AssignmentMissing), |c| Ok(c.0))
    })?;
    let y = AllocatedNum::alloc(cs.namespace(|| "y"), || {
      coords.map_or(Err(SynthesisError::AssignmentMissing), |c| Ok(c.1))
    })?;

    Ok(AllocatedPointNonInfinity { x, y })
  }

  /// Conditional select using an AllocatedNum instead of Boolean.
  pub fn conditionally_select2<CS: ConstraintSystem<E::Base>>(
    mut cs: CS,
    a: &Self,
    b: &Self,
    condition: &AllocatedNum<E::Base>,
  ) -> Result<AllocatedPointNonInfinity<E>, SynthesisError> {
    let x = conditionally_select2(cs.namespace(|| "select x"), &a.x, &b.x, condition)?;
    let y = conditionally_select2(cs.namespace(|| "select y"), &a.y, &b.y, condition)?;

    Ok(AllocatedPointNonInfinity { x, y })
  }

  /// Enforce that self equals other.
  pub fn enforce_equal<CS: ConstraintSystem<E::Base>>(
    &self,
    mut cs: CS,
    other: &Self,
  ) -> Result<(), SynthesisError> {
    cs.enforce(
      || "check x equality",
      |lc| lc + self.x.get_variable() - other.x.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc,
    );
    cs.enforce(
      || "check y equality",
      |lc| lc + self.y.get_variable() - other.y.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc,
    );

    Ok(())
  }
}
