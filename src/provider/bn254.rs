// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! This module implements the Spartan traits for BN254 curve (G1)
use crate::{
  impl_traits,
  provider::{
    msm::{msm, msm_small},
    traits::{DlogGroup, DlogGroupExt},
  },
  traits::{Group, PrimeFieldExt, transcript::TranscriptReprTrait},
};
use digest::{ExtendableOutput, Update};
use ff::FromUniformBytes;
use halo2curves::{
  CurveAffine, CurveExt,
  bn256::{G1 as Bn254, G1Affine as Bn254Affine},
  group::{Curve, Group as AnotherGroup, cofactor::CofactorCurveAffine},
};
use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{Num, ToPrimitive};
use rayon::prelude::*;
use sha3::Shake256;
use std::io::Read;

/// Re-exports that give access to the standard aliases used in the code base, for bn254
pub mod bn254 {
  pub use halo2curves::bn256::{Fq as Base, Fr as Scalar, G1 as Point, G1Affine as Affine};
}

impl_traits!(
  bn254,
  Bn254,
  Bn254Affine,
  "30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001",
  "30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47"
);

impl<G: Group> TranscriptReprTrait<G> for bn254::Base {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    self.to_bytes().into_iter().rev().collect()
  }
}
