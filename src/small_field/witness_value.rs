// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! `WitnessValue` trait for generic witness types.
//!
//! This trait abstracts over the witness representation (bool, i8) so that
//! the commit/prove/bind pipeline is generic over the witness type.

use crate::errors::SpartanError;
use crate::provider::traits::DlogGroupExt;
use ff::PrimeField;

/// Trait for witness value types that can be committed and used in sumcheck.
///
/// Each witness type has an `Extended` type used for Lagrange extension in
/// the inner sumcheck. For bool witnesses, the extension type is `i8` (since
/// extending {0,1} to {∞,0,1} produces values in {-1,0,1,2}). For i8 witnesses,
/// the extension type is also `i8`.
pub trait WitnessValue: Copy + Send + Sync + 'static {
  /// The extended type after Lagrange interpolation.
  type Extended: Copy + Default + Send + Sync
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
    use_parallelism_internally: bool,
  ) -> Result<G, SpartanError>;
}

impl WitnessValue for bool {
  type Extended = i8;

  #[inline(always)]
  fn to_extended(self) -> i8 {
    self as i8
  }

  #[inline(always)]
  fn to_field<F: PrimeField>(self) -> F {
    if self { F::ONE } else { F::ZERO }
  }

  fn msm<G: DlogGroupExt>(
    values: &[Self],
    bases: &[G::AffineGroupElement],
    use_parallelism_internally: bool,
  ) -> Result<G, SpartanError> {
    G::vartime_multiscalar_mul_bool(values, bases, use_parallelism_internally)
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
    use_parallelism_internally: bool,
  ) -> Result<G, SpartanError> {
    G::vartime_multiscalar_mul_signed_small(values, bases, use_parallelism_internally)
  }
}
