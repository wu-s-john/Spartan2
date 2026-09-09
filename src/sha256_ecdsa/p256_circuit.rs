// SPDX-License-Identifier: MIT
//! Native P-256 constraints. Complete projective formulas are RCB 2015,
//! algorithms 4 and 6 (https://eprint.iacr.org/2015/1060), as arranged in
//! RustCrypto primeorder 0.13.6, point_arithmetic.rs (MIT/Apache-2.0).
//! The RustCrypto MIT notice is retained in licenses/RustCrypto-MIT.

use crate::neutronnova::Scalar as F;
use bellpepper::gadgets::uint32::UInt32;
use bellpepper_core::{
  ConstraintSystem, LinearCombination, SynthesisError,
  boolean::{AllocatedBit, Boolean},
};
use ff::Field;
use num_bigint::BigUint;

pub(super) const ORDER: &str = "ffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551";
const PRIME: &str = "ffffffff00000001000000000000000000000000ffffffffffffffffffffffff";
const CURVE_B: &str = "5ac635d8aa3a93e7b3ebbd55769886bc651d06b0cc53b0f63bce3c3e27d2604b";
const GX: &str = "6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296";
const GY: &str = "4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5";

fn integer(hex: &str) -> BigUint {
  BigUint::parse_bytes(hex.as_bytes(), 16).unwrap()
}
fn field(n: &BigUint) -> F {
  n.to_bytes_be()
    .iter()
    .fold(F::ZERO, |v, b| v * F::from(256) + F::from(u64::from(*b)))
}
fn constant<CS: ConstraintSystem<F>>(n: F) -> Num {
  Num {
    lc: LinearCombination::zero() + (n, CS::one()),
    value: Some(n),
  }
}

#[derive(Clone)]
struct Num {
  lc: LinearCombination<F>,
  value: Option<F>,
}
impl Num {
  fn add(&self, b: &Self) -> Self {
    Self {
      lc: self.lc.clone() + &b.lc,
      value: self.value.zip(b.value).map(|(a, b)| a + b),
    }
  }
  fn sub(&self, b: &Self) -> Self {
    Self {
      lc: self.lc.clone() - &b.lc,
      value: self.value.zip(b.value).map(|(a, b)| a - b),
    }
  }
  fn scale(&self, b: F) -> Self {
    Self {
      lc: LinearCombination::zero() + (b, &self.lc),
      value: self.value.map(|a| a * b),
    }
  }
  fn mul<CS: ConstraintSystem<F>>(&self, mut cs: CS, b: &Self) -> Result<Self, SynthesisError> {
    let value = self.value.zip(b.value).map(|(a, b)| a * b);
    let var = cs.alloc(
      || "product",
      || value.ok_or(SynthesisError::AssignmentMissing),
    )?;
    cs.enforce(
      || "multiplication",
      |lc| lc + &self.lc,
      |lc| lc + &b.lc,
      |lc| lc + var,
    );
    Ok(Self {
      lc: LinearCombination::zero() + var,
      value,
    })
  }
  fn equal<CS: ConstraintSystem<F>>(&self, mut cs: CS, b: &Self) {
    cs.enforce(
      || "equality",
      |lc| lc + &self.lc - &b.lc,
      |lc| lc + CS::one(),
      |lc| lc,
    );
  }
  fn nonzero<CS: ConstraintSystem<F>>(&self, mut cs: CS) -> Result<(), SynthesisError> {
    let inv = cs.alloc(
      || "inverse",
      || {
        let value = self.value.ok_or(SynthesisError::AssignmentMissing)?;
        Ok(Option::<F>::from(value.invert()).unwrap_or(F::ZERO))
      },
    )?;
    cs.enforce(
      || "nonzero",
      |lc| lc + &self.lc,
      |lc| lc + inv,
      |lc| lc + CS::one(),
    );
    Ok(())
  }
}

fn from_bits<CS: ConstraintSystem<F>>(bits: &[Boolean]) -> Num {
  let mut lc = LinearCombination::zero();
  let mut value = Some(F::ZERO);
  let mut weight = F::ONE;
  for b in bits {
    lc = lc + &b.lc(CS::one(), weight);
    value = value
      .zip(b.get_value())
      .map(|(a, b)| if b { a + weight } else { a });
    weight = weight.double();
  }
  Num { lc, value }
}

pub(super) fn publicize_word<CS: ConstraintSystem<F>>(
  mut cs: CS,
  word: &UInt32,
  value: Option<u32>,
) -> Result<(), SynthesisError> {
  let input = cs.alloc_input(
    || "public word",
    || {
      value
        .map(|v| F::from(u64::from(v)))
        .ok_or(SynthesisError::AssignmentMissing)
    },
  )?;
  let n = from_bits::<CS>(&word.clone().into_bits());
  cs.enforce(
    || "bind public word",
    |lc| lc + &n.lc,
    |lc| lc + CS::one(),
    |lc| lc + input,
  );
  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;
  use bellpepper_core::test_cs::TestConstraintSystem;
  use p256::{ProjectivePoint, Scalar, elliptic_curve::sec1::ToEncodedPoint};

  fn check_finite<CS: ConstraintSystem<F>>(mut cs: CS, p: &Point, expected: ProjectivePoint) {
    let encoded = expected.to_affine().to_encoded_point(false);
    let x = field(&BigUint::from_bytes_be(encoded.x().unwrap()));
    let y = field(&BigUint::from_bytes_be(encoded.y().unwrap()));
    p.x.equal(cs.namespace(|| "x"), &p.z.scale(x));
    p.y.equal(cs.namespace(|| "y"), &p.z.scale(y));
    p.z.nonzero(cs.namespace(|| "finite")).unwrap();
  }

  #[test]
  fn complete_formulas_cover_identity_doubling_and_opposites() {
    let mut cs = TestConstraintSystem::<F>::new();
    let g = Point::affine(
      cs.namespace(|| "G"),
      constant::<TestConstraintSystem<F>>(field(&integer(GX))),
      constant::<TestConstraintSystem<F>>(field(&integer(GY))),
    )
    .unwrap();
    let identity = Point::identity::<TestConstraintSystem<F>>();
    let neg = Point {
      x: g.x.clone(),
      y: g.y.scale(-F::ONE),
      z: g.z.clone(),
    };
    let zero = g.add(cs.namespace(|| "G plus negative G"), &neg).unwrap();
    zero.x.equal(
      cs.namespace(|| "infinity x"),
      &constant::<TestConstraintSystem<F>>(F::ZERO),
    );
    zero.z.equal(
      cs.namespace(|| "infinity z"),
      &constant::<TestConstraintSystem<F>>(F::ZERO),
    );
    zero
      .y
      .nonzero(cs.namespace(|| "nonzero projective infinity"))
      .unwrap();
    let left = identity
      .add(cs.namespace(|| "identity plus G"), &g)
      .unwrap();
    check_finite(
      cs.namespace(|| "check left"),
      &left,
      ProjectivePoint::GENERATOR,
    );
    let right = g
      .add(cs.namespace(|| "G plus identity"), &identity)
      .unwrap();
    check_finite(
      cs.namespace(|| "check right"),
      &right,
      ProjectivePoint::GENERATOR,
    );
    let doubled = g.double(cs.namespace(|| "double G")).unwrap();
    check_finite(
      cs.namespace(|| "check double"),
      &doubled,
      ProjectivePoint::GENERATOR * Scalar::from(2u64),
    );
    let added = g.add(cs.namespace(|| "G plus G"), &g).unwrap();
    check_finite(
      cs.namespace(|| "check self add"),
      &added,
      ProjectivePoint::GENERATOR * Scalar::from(2u64),
    );
    let back = zero
      .add(cs.namespace(|| "infinity result plus G"), &g)
      .unwrap();
    check_finite(
      cs.namespace(|| "check back"),
      &back,
      ProjectivePoint::GENERATOR,
    );
    let doubled_zero = identity.double(cs.namespace(|| "double identity")).unwrap();
    doubled_zero.z.equal(
      cs.namespace(|| "doubled identity is infinity"),
      &constant::<TestConstraintSystem<F>>(F::ZERO),
    );
    doubled_zero
      .y
      .nonzero(cs.namespace(|| "doubled identity is nonzero projective"))
      .unwrap();
    assert!(cs.is_satisfied());
  }

  #[test]
  fn integer_bound_excludes_the_modular_wrap_branch() {
    let prime = integer(PRIME);
    let order = integer(ORDER);
    for (r, enabled, expected) in [
      (BigUint::from(1u8), true, true),
      (&prime - &order, true, false),
      (&order - 1u8, true, false),
      (&order - 1u8, false, true),
    ] {
      let mut bytes = [0u8; 32];
      let encoded = r.to_bytes_be();
      bytes[32 - encoded.len()..].copy_from_slice(&encoded);
      let mut cs = TestConstraintSystem::<F>::new();
      let (_, bits) = alloc_integer(cs.namespace(|| "r"), Some(bytes), false).unwrap();
      let guard =
        Boolean::from(AllocatedBit::alloc(cs.namespace(|| "high"), Some(enabled)).unwrap());
      at_most(
        cs.namespace(|| "no wrap"),
        &bits,
        &(&prime - &order - 1u8),
        guard,
      )
      .unwrap();
      assert_eq!(cs.is_satisfied(), expected);
    }
  }
}

fn alloc_integer<CS: ConstraintSystem<F>>(
  mut cs: CS,
  bytes: Option<[u8; 32]>,
  public: bool,
) -> Result<(Num, Vec<Boolean>), SynthesisError> {
  let mut words = Vec::new();
  for i in 0..8 {
    let value = bytes.map(|b| u32::from_be_bytes(b[4 * i..4 * i + 4].try_into().unwrap()));
    let word = UInt32::alloc(cs.namespace(|| format!("word {i}")), value)?;
    if public {
      publicize_word(cs.namespace(|| format!("public {i}")), &word, value)?;
    }
    words.push(word);
  }
  let bits = words
    .into_iter()
    .rev()
    .flat_map(UInt32::into_bits)
    .collect::<Vec<_>>();
  Ok((from_bits::<CS>(&bits), bits))
}

// Lexicographic comparison to a constant, enabled by guard. Using bound-1
// gives a strict integer bound without arithmetic modulo the proof field.
fn at_most<CS: ConstraintSystem<F>>(
  mut cs: CS,
  bits: &[Boolean],
  bound: &BigUint,
  guard: Boolean,
) -> Result<(), SynthesisError> {
  let mut equal = guard;
  for i in (0..bits.len()).rev() {
    if bound.bit(i as u64) {
      equal = Boolean::and(cs.namespace(|| format!("equal {i}")), &equal, &bits[i])?;
    } else {
      cs.enforce(
        || format!("bound {i}"),
        |lc| lc + &equal.lc(CS::one(), F::ONE),
        |lc| lc + &bits[i].lc(CS::one(), F::ONE),
        |lc| lc,
      );
    }
  }
  Ok(())
}

#[derive(Clone)]
struct Point {
  x: Num,
  y: Num,
  z: Num,
}
impl Point {
  fn identity<CS: ConstraintSystem<F>>() -> Self {
    Self {
      x: constant::<CS>(F::ZERO),
      y: constant::<CS>(F::ONE),
      z: constant::<CS>(F::ZERO),
    }
  }
  fn affine<CS: ConstraintSystem<F>>(mut cs: CS, x: Num, y: Num) -> Result<Self, SynthesisError> {
    let xx = x.mul(cs.namespace(|| "x square"), &x)?;
    let xxx = x.mul(cs.namespace(|| "x cube"), &xx)?;
    let yy = y.mul(cs.namespace(|| "y square"), &y)?;
    yy.equal(
      cs.namespace(|| "curve equation"),
      &xxx
        .sub(&x.scale(F::from(3)))
        .add(&constant::<CS>(field(&integer(CURVE_B)))),
    );
    Ok(Self {
      x,
      y,
      z: constant::<CS>(F::ONE),
    })
  }
  fn add<CS: ConstraintSystem<F>>(&self, mut cs: CS, b: &Self) -> Result<Self, SynthesisError> {
    let curve_b = field(&integer(CURVE_B));
    let three = F::from(3);
    let xx = self.x.mul(cs.namespace(|| "xx"), &b.x)?;
    let yy = self.y.mul(cs.namespace(|| "yy"), &b.y)?;
    let zz = self.z.mul(cs.namespace(|| "zz"), &b.z)?;
    let xy = self
      .x
      .add(&self.y)
      .mul(cs.namespace(|| "xy"), &b.x.add(&b.y))?
      .sub(&xx.add(&yy));
    let yz = self
      .y
      .add(&self.z)
      .mul(cs.namespace(|| "yz"), &b.y.add(&b.z))?
      .sub(&yy.add(&zz));
    let xz = self
      .x
      .add(&self.z)
      .mul(cs.namespace(|| "xz"), &b.x.add(&b.z))?
      .sub(&xx.add(&zz));
    let bzz3 = xz.sub(&zz.scale(curve_b)).scale(three);
    let ym = yy.sub(&bzz3);
    let yp = yy.add(&bzz3);
    let zz3 = zz.scale(three);
    let bxz3 = xz.scale(curve_b).sub(&zz3.add(&xx)).scale(three);
    let xm = xx.scale(three).sub(&zz3);
    let x = yp
      .mul(cs.namespace(|| "x first"), &xy)?
      .sub(&yz.mul(cs.namespace(|| "x second"), &bxz3)?);
    let y = yp
      .mul(cs.namespace(|| "y first"), &ym)?
      .add(&xm.mul(cs.namespace(|| "y second"), &bxz3)?);
    let z = ym
      .mul(cs.namespace(|| "z first"), &yz)?
      .add(&xy.mul(cs.namespace(|| "z second"), &xm)?);
    Ok(Self { x, y, z })
  }
  fn double<CS: ConstraintSystem<F>>(&self, mut cs: CS) -> Result<Self, SynthesisError> {
    let curve_b = field(&integer(CURVE_B));
    let three = F::from(3);
    let xx = self.x.mul(cs.namespace(|| "xx"), &self.x)?;
    let yy = self.y.mul(cs.namespace(|| "yy"), &self.y)?;
    let zz = self.z.mul(cs.namespace(|| "zz"), &self.z)?;
    let xy2 = self
      .x
      .mul(cs.namespace(|| "xy"), &self.y)?
      .scale(F::from(2));
    let xz2 = self
      .x
      .mul(cs.namespace(|| "xz"), &self.z)?
      .scale(F::from(2));
    let bzz3 = zz.scale(curve_b).sub(&xz2).scale(three);
    let ym = yy.sub(&bzz3);
    let yp = yy.add(&bzz3);
    let yf = yp.mul(cs.namespace(|| "y fragment"), &ym)?;
    let xf = ym.mul(cs.namespace(|| "x fragment"), &xy2)?;
    let zz3 = zz.scale(three);
    let bxz6 = xz2.scale(curve_b).sub(&zz3.add(&xx)).scale(three);
    let xm = xx.scale(three).sub(&zz3);
    let y = yf.add(&xm.mul(cs.namespace(|| "y tail"), &bxz6)?);
    let yz2 = self
      .y
      .mul(cs.namespace(|| "yz"), &self.z)?
      .scale(F::from(2));
    let x = xf.sub(&bxz6.mul(cs.namespace(|| "x tail"), &yz2)?);
    let z = yz2.mul(cs.namespace(|| "z"), &yy)?.scale(F::from(4));
    Ok(Self { x, y, z })
  }
  fn select<CS: ConstraintSystem<F>>(
    mut cs: CS,
    bit: &Boolean,
    a: &Self,
    b: &Self,
  ) -> Result<Self, SynthesisError> {
    let bit = from_bits::<CS>(std::slice::from_ref(bit));
    let x = b.x.add(&bit.mul(cs.namespace(|| "x"), &a.x.sub(&b.x))?);
    let y = b.y.add(&bit.mul(cs.namespace(|| "y"), &a.y.sub(&b.y))?);
    let z = b.z.add(&bit.mul(cs.namespace(|| "z"), &a.z.sub(&b.z))?);
    Ok(Self { x, y, z })
  }
  fn mul<CS: ConstraintSystem<F>>(
    &self,
    mut cs: CS,
    bits: &[Boolean],
  ) -> Result<Self, SynthesisError> {
    let mut result = Self::identity::<CS>();
    for (i, bit) in bits.iter().enumerate().rev() {
      let doubled = result.double(cs.namespace(|| format!("double {i}")))?;
      let added = doubled.add(cs.namespace(|| format!("add {i}")), self)?;
      result = Self::select(
        cs.namespace(|| format!("select {i}")),
        bit,
        &added,
        &doubled,
      )?;
    }
    Ok(result)
  }
}

pub(super) fn constrain<CS: ConstraintSystem<F>>(
  mut cs: CS,
  inputs: [Option<[u8; 32]>; 5],
  r_point: Option<([u8; 32], [u8; 32])>,
) -> Result<(), SynthesisError> {
  let (_, digest) = alloc_integer(cs.namespace(|| "digest"), inputs[0], true)?;
  let (qx, qxb) = alloc_integer(cs.namespace(|| "public key x"), inputs[1], true)?;
  let (qy, qyb) = alloc_integer(cs.namespace(|| "public key y"), inputs[2], true)?;
  let (r, rb) = alloc_integer(cs.namespace(|| "signature r"), inputs[3], true)?;
  let (s, sb) = alloc_integer(cs.namespace(|| "signature s"), inputs[4], true)?;
  let prime = integer(PRIME);
  let order = integer(ORDER);
  for (name, bits, bound) in [
    ("qx", &qxb, &prime),
    ("qy", &qyb, &prime),
    ("r", &rb, &order),
    ("s", &sb, &order),
  ] {
    at_most(
      cs.namespace(|| format!("canonical {name}")),
      bits,
      &(bound - 1u8),
      Boolean::constant(true),
    )?;
  }
  r.nonzero(cs.namespace(|| "r nonzero"))?;
  s.nonzero(cs.namespace(|| "s nonzero"))?;
  let q = Point::affine(cs.namespace(|| "Q"), qx, qy)?;
  let (rx, rxb) = alloc_integer(cs.namespace(|| "R x"), r_point.map(|p| p.0), false)?;
  let (ry, ryb) = alloc_integer(cs.namespace(|| "R y"), r_point.map(|p| p.1), false)?;
  at_most(
    cs.namespace(|| "canonical Rx"),
    &rxb,
    &(&prime - 1u8),
    Boolean::constant(true),
  )?;
  at_most(
    cs.namespace(|| "canonical Ry"),
    &ryb,
    &(&prime - 1u8),
    Boolean::constant(true),
  )?;
  let high = Boolean::from(AllocatedBit::alloc(
    cs.namespace(|| "x reduction bit"),
    r_point.map(|p| BigUint::from_bytes_be(&p.0) >= order),
  )?);
  rx.equal(
    cs.namespace(|| "x reduction"),
    &r.add(&from_bits::<CS>(std::slice::from_ref(&high)).scale(field(&order))),
  );
  at_most(
    cs.namespace(|| "x reduction no wrap"),
    &rb,
    &(&prime - &order - 1u8),
    high,
  )?;
  let rp = Point::affine(cs.namespace(|| "R"), rx, ry)?;
  let g = Point {
    x: constant::<CS>(field(&integer(GX))),
    y: constant::<CS>(field(&integer(GY))),
    z: constant::<CS>(F::ONE),
  };
  let left = rp.mul(cs.namespace(|| "sR"), &sb)?;
  let zg = g.mul(cs.namespace(|| "zG"), &digest)?;
  let rq = q.mul(cs.namespace(|| "rQ"), &rb)?;
  let right = zg.add(cs.namespace(|| "zG plus rQ"), &rq)?;
  left.z.nonzero(cs.namespace(|| "left finite"))?;
  right.z.nonzero(cs.namespace(|| "right finite"))?;
  let lx = left.x.mul(cs.namespace(|| "left x"), &right.z)?;
  let rx = right.x.mul(cs.namespace(|| "right x"), &left.z)?;
  lx.equal(cs.namespace(|| "same x"), &rx);
  let ly = left.y.mul(cs.namespace(|| "left y"), &right.z)?;
  let ry = right.y.mul(cs.namespace(|| "right y"), &left.z)?;
  ly.equal(cs.namespace(|| "same y"), &ry);
  Ok(())
}
