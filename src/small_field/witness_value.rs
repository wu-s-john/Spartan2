// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! `WitnessValue` trait for generic witness types.
//!
//! This trait abstracts over the witness representation (bool, i8) so that
//! the commit/prove/bind pipeline is generic over the witness type.

use crate::{errors::SpartanError, provider::traits::DlogGroupExt};
use ff::PrimeField;

/// Trait for witness value types that can be committed and used in sumcheck.
///
/// Each witness type has an `Extended` type used for Lagrange extension in
/// the inner sumcheck. For bool witnesses, the extension type is `i16` (since
/// extending {0,1}^l0 to {∞,0,1}^l0 produces values growing by 3× per round;
/// i16 safely handles l0 ≤ 9, i.e., 3^9 = 19683 ≤ i16::MAX = 32767). For i8
/// witnesses, the extension type is `i8`.
pub trait WitnessValue: Copy + Default + PartialEq + Send + Sync + 'static {
  /// The extended type after Lagrange interpolation.
  type Extended: Copy
    + Default
    + Send
    + Sync
    + std::ops::Add<Output = Self::Extended>
    + std::ops::Sub<Output = Self::Extended>;

  /// Convert to the extended type.
  fn to_extended(self) -> Self::Extended;

  /// Convert to a field element.
  fn to_field<F: PrimeField>(self) -> F;

  /// Multi-scalar multiplication dispatching through `DlogGroupExt`.
  fn msm<G: DlogGroupExt>(
    values: &[Self],
    bases: &[G::AffineGroupElement],
  ) -> Result<G, SpartanError>;

  /// Compute `a_bound * (2·hi - lo)` for the degree-2 sumcheck evaluation point.
  ///
  /// Default: converts to field. Override for binary/small types to avoid field muls.
  #[inline(always)]
  fn eval_2_contribution<F: PrimeField>(hi: Self, lo: Self, a_bound: F) -> F {
    let z_bound = hi.to_field::<F>() + hi.to_field::<F>() - lo.to_field::<F>();
    a_bound * z_bound
  }

  /// Compute `da * (hi - lo)` for the BDDT leading coefficient.
  ///
  /// Default: converts to field. Override for binary/small types to avoid field muls.
  #[inline(always)]
  fn leading_contribution<F: PrimeField>(hi: Self, lo: Self, da: F) -> F {
    let dz = hi.to_field::<F>() - lo.to_field::<F>();
    da * dz
  }

  /// Compute the bind contribution: `lo * (1 - r) + hi * r`.
  ///
  /// Default: converts to field and multiplies. Override for binary types to use
  /// conditional adds instead of field multiplies.
  #[inline(always)]
  fn bind_lo_hi<F: PrimeField>(lo: Self, hi: Self, r: F, one_minus_r: F) -> F {
    lo.to_field::<F>() * one_minus_r + hi.to_field::<F>() * r
  }
}

impl WitnessValue for bool {
  /// `i16` safely handles l0 ≤ 9: max extension value is 3^9 = 19683 ≤ i16::MAX.
  type Extended = i16;

  #[inline(always)]
  fn to_extended(self) -> i16 {
    self as i16
  }

  #[inline(always)]
  fn to_field<F: PrimeField>(self) -> F {
    if self { F::ONE } else { F::ZERO }
  }

  fn msm<G: DlogGroupExt>(
    values: &[Self],
    bases: &[G::AffineGroupElement],
  ) -> Result<G, SpartanError> {
    G::vartime_multiscalar_mul_bool(values, bases)
  }

  /// Fast path: `z_bound = 2·hi - lo` for bool ∈ {0,1} has only 4 cases,
  /// all computable with conditional adds — zero field multiplies.
  #[inline(always)]
  fn eval_2_contribution<F: PrimeField>(hi: bool, lo: bool, a_bound: F) -> F {
    match (hi, lo) {
      (false, false) => F::ZERO,          // z_bound = 0
      (false, true) => -a_bound,          // z_bound = -1
      (true, false) => a_bound + a_bound, // z_bound = 2
      (true, true) => a_bound,            // z_bound = 1
    }
  }

  /// Fast path: `da * (hi - lo)` for bool ∈ {0,1} has only 4 cases,
  /// all computable with conditional adds — zero field multiplies.
  #[inline(always)]
  fn leading_contribution<F: PrimeField>(hi: bool, lo: bool, da: F) -> F {
    match (hi, lo) {
      (false, false) => F::ZERO, // dz = 0
      (false, true) => -da,      // dz = -1
      (true, false) => da,       // dz = 1
      (true, true) => F::ZERO,   // dz = 0
    }
  }

  /// Fast path: `lo * (1-r) + hi * r` for bool ∈ {0,1} has only 4 cases,
  /// all computable with conditional field elements — zero field multiplies.
  #[inline(always)]
  fn bind_lo_hi<F: PrimeField>(lo: bool, hi: bool, r: F, one_minus_r: F) -> F {
    match (lo, hi) {
      (false, false) => F::ZERO,
      (false, true) => r,
      (true, false) => one_minus_r,
      (true, true) => F::ONE,
    }
  }
}

impl WitnessValue for i8 {
  type Extended = i8;

  #[inline(always)]
  fn to_extended(self) -> i8 {
    self
  }

  #[inline(always)]
  fn to_field<F: PrimeField>(self) -> F {
    match self {
      0 => F::ZERO,
      1 => F::ONE,
      v if v > 0 => F::from(v as u64),
      v => -F::from((-v) as u64),
    }
  }

  fn msm<G: DlogGroupExt>(
    values: &[Self],
    bases: &[G::AffineGroupElement],
  ) -> Result<G, SpartanError> {
    G::vartime_multiscalar_mul_signed_small(values, bases)
  }
}
