// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! This module provides a multi-scalar multiplication routine.
//!
//! For full-field-element MSM, we use a signed-decomposition + bit-width-partitioning strategy:
//!
//! 1. **Signed scalar decomposition**: For each scalar `s`, we compare `num_bits(s)` vs
//!    `num_bits(p - s)` and use whichever representation is smaller, negating the base point
//!    if we use `p - s`. This halves the effective scalar range.
//! 2. **Bit-width partitioning**: Scalars are routed to the optimal algorithm based on their
//!    actual bit-width after signed reduction: binary accumulation for 0/1, single-window
//!    bucket sort for ≤10 bits, multi-window Pippenger for ≤32 bits, and halo2curves for the rest.
//! 3. **XYZZ bucket coordinates**: Extended Jacobian `(X, Y, ZZ, ZZZ)` provides cheaper
//!    mixed addition (7M + 2S) compared to standard Jacobian (~11M + 5S for proj+affine).
//!
//! The MSM implementations are adapted from Nova/halo2/jolt.
use crate::{errors::SpartanError, start_span};
use ff::{Field, PrimeField};
use halo2curves::{CurveAffine, group::Group};
use num_integer::Integer;
use num_traits::{ToPrimitive, Zero};
use rayon::{current_num_threads, prelude::*};
use tracing::info;

// ==================================================================================
// XYZZ (Extended Jacobian) Bucket coordinates
// ==================================================================================

/// Extended Jacobian (XYZZ) coordinates for efficient MSM bucket accumulation.
///
/// Stores `(X, Y, ZZ, ZZZ)` where `ZZ = Z²` and `ZZZ = Z³` for a Jacobian point
/// with coordinates `(X/ZZ, Y/ZZZ)` in affine.
///
/// Mixed addition (affine + XYZZ) costs 7M + 2S vs ~11M + 5S for standard projective+affine.
/// Formula source: <https://www.hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html>
///
/// The doubling formula handles both `a = 0` (Pallas, Vesta) and `a ≠ 0` (T256)
/// via the `curve_a` parameter passed to `double_in_place`.
#[derive(Copy, Clone)]
struct BucketXYZZ<F: Field> {
  x: F,
  y: F,
  zz: F,
  zzz: F,
}

impl<F: Field> BucketXYZZ<F> {
  /// The point at infinity (identity).
  #[inline]
  fn zero() -> Self {
    Self {
      x: F::ONE,
      y: F::ONE,
      zz: F::ZERO,
      zzz: F::ZERO,
    }
  }

  /// Check if this is the identity.
  #[inline]
  fn is_zero(&self) -> bool {
    self.zz == F::ZERO
  }

  /// Double in place (dbl-2008-s-1, general formula with curve parameter `a`).
  /// Cost: 2M + 5S + 7add (a=0) or 3M + 5S + 8add (a≠0)
  fn double_in_place(&mut self, curve_a: F) {
    if self.is_zero() {
      return;
    }
    // U = 2*Y1
    let u = self.y.double();
    // V = U^2
    let v = u.square();
    // W = U*V
    let w = u * v;
    // S = X1*V
    let s = self.x * v;
    // M = 3*X1^2 + a*ZZ^2
    let x_sq = self.x.square();
    let m = if curve_a == F::ZERO {
      x_sq.double() + x_sq
    } else {
      x_sq.double() + x_sq + curve_a * self.zz.square()
    };
    // X3 = M^2 - 2*S
    self.x = m.square() - s.double();
    // Y3 = M*(S - X3) - W*Y1
    self.y = m * (s - self.x) - w * self.y;
    // ZZ3 = V*ZZ1
    self.zz *= v;
    // ZZZ3 = W*ZZZ1
    self.zzz *= w;
  }

  /// XYZZ += XYZZ (full addition, add-2008-s).
  fn add_assign_bucket(&mut self, other: &Self, curve_a: F) {
    if other.is_zero() {
      return;
    }
    if self.is_zero() {
      *self = *other;
      return;
    }
    let u1 = self.x * other.zz;
    let u2 = other.x * self.zz;
    let s1 = self.y * other.zzz;
    let s2 = other.y * self.zzz;

    if u1 == u2 {
      if s1 == s2 {
        self.double_in_place(curve_a);
      } else {
        *self = Self::zero();
      }
      return;
    }
    let p = u2 - u1;
    let r = s2 - s1;
    let pp = p.square();
    let ppp = p * pp;
    let q = u1 * pp;
    self.x = r.square() - ppp - q.double();
    self.y = r * (q - self.x) - s1 * ppp;
    self.zz = self.zz * other.zz * pp;
    self.zzz = self.zzz * other.zzz * ppp;
  }
}

/// Compute curve parameter `a` from the generator and its double.
/// For `y² = x³ + ax + b`: a = (dy² - dx³ - gy² + gx³) / (dx - gx)
#[inline]
fn compute_curve_a<C: CurveAffine>() -> C::Base {
  use halo2curves::group::Curve;
  let g = C::generator();
  let g2 = (g + g).to_affine();
  let gc = g.coordinates().unwrap();
  let g2c = g2.coordinates().unwrap();
  let (gx, gy) = (*gc.x(), *gc.y());
  let (dx, dy) = (*g2c.x(), *g2c.y());
  let num = dy.square() - dx.square() * dx - gy.square() + gx.square() * gx;
  let den = dx - gx;
  num * den.invert().unwrap()
}

/// Mixed addition: BucketXYZZ += CurveAffine point (madd-2008-s).
/// Cost: 7M + 2S
#[inline]
fn bucket_add_affine<C: CurveAffine>(bucket: &mut BucketXYZZ<C::Base>, p: &C, curve_a: C::Base) {
  if bool::from(p.is_identity()) {
    return;
  }
  let coords = p.coordinates().unwrap();
  let px = *coords.x();
  let py = *coords.y();

  if bucket.is_zero() {
    bucket.x = px;
    bucket.y = py;
    bucket.zz = C::Base::ONE;
    bucket.zzz = C::Base::ONE;
    return;
  }
  // U2 = X2*ZZ1, S2 = Y2*ZZZ1
  let u2 = px * bucket.zz;
  let s2 = py * bucket.zzz;

  if bucket.x == u2 {
    if bucket.y == s2 {
      bucket.double_in_place(curve_a);
    } else {
      *bucket = BucketXYZZ::zero();
    }
    return;
  }
  let p_val = u2 - bucket.x;
  let r = s2 - bucket.y;
  let pp = p_val.square();
  let ppp = p_val * pp;
  let q = bucket.x * pp;
  bucket.x = r.square() - ppp - q.double();
  bucket.y = r * (q - bucket.x) - bucket.y * ppp;
  bucket.zz *= pp;
  bucket.zzz *= ppp;
}

/// Mixed addition using raw (x, y) coordinates (madd-2008-s).
/// Avoids CurveAffine::coordinates() overhead since caller pre-extracts them.
/// Cost: 7M + 2S
#[inline]
fn bucket_add_affine_xy<C: CurveAffine>(
  bucket: &mut BucketXYZZ<C::Base>,
  px: C::Base,
  py: C::Base,
  curve_a: C::Base,
) {
  if bucket.is_zero() {
    bucket.x = px;
    bucket.y = py;
    bucket.zz = C::Base::ONE;
    bucket.zzz = C::Base::ONE;
    return;
  }
  let u2 = px * bucket.zz;
  let s2 = py * bucket.zzz;

  if bucket.x == u2 {
    if bucket.y == s2 {
      bucket.double_in_place(curve_a);
    } else {
      *bucket = BucketXYZZ::zero();
    }
    return;
  }
  let p_val = u2 - bucket.x;
  let r = s2 - bucket.y;
  let pp = p_val.square();
  let ppp = p_val * pp;
  let q = bucket.x * pp;
  bucket.x = r.square() - ppp - q.double();
  bucket.y = r * (q - bucket.x) - bucket.y * ppp;
  bucket.zz *= pp;
  bucket.zzz *= ppp;
}

/// Convert XYZZ bucket to projective curve point.
///
/// Computes affine coordinates `(X/ZZ, Y/ZZZ)` then converts to projective.
/// Only called O(windows) times per thread, so the field inversion cost is negligible.
#[inline]
fn bucket_to_curve<C: CurveAffine>(bucket: &BucketXYZZ<C::Base>) -> C::CurveExt {
  if bucket.is_zero() {
    return C::CurveExt::identity();
  }
  let zz_inv = bucket.zz.invert().unwrap();
  let zzz_inv = bucket.zzz.invert().unwrap();
  let x = bucket.x * zz_inv;
  let y = bucket.y * zzz_inv;
  let ct: Option<C> = C::from_xy(x, y).into();
  ct.expect("XYZZ bucket should produce a valid curve point")
    .into()
}

// ==================================================================================
// Scalar utilities
// ==================================================================================

/// Count significant bits in a field element (from its little-endian repr).
#[inline]
fn scalar_num_bits<F: PrimeField>(s: &F) -> u32 {
  let repr = s.to_repr();
  let bytes = repr.as_ref();
  for i in (0..bytes.len()).rev() {
    if bytes[i] != 0 {
      return i as u32 * 8 + (8 - bytes[i].leading_zeros());
    }
  }
  0
}

/// Extract the low 64 bits from a field element's little-endian representation.
#[inline]
fn repr_low_u64<F: PrimeField>(s: &F) -> u64 {
  let repr = s.to_repr();
  let bytes = repr.as_ref();
  let mut buf = [0u8; 8];
  let len = bytes.len().min(8);
  buf[..len].copy_from_slice(&bytes[..len]);
  u64::from_le_bytes(buf)
}

#[inline(always)]
fn serial_window_size(num_bases: usize) -> usize {
  if num_bases < 4 {
    1
  } else if num_bases < 32 {
    3
  } else {
    let c_base = (f64::from(num_bases as u32)).ln().ceil() as usize;
    let cost = |c: usize| ((256 + c - 1) / c) * (num_bases + (1 << (c - 1)));
    if cost(c_base + 1) < cost(c_base) {
      c_base + 1
    } else {
      c_base
    }
  }
}

#[inline(always)]
fn serial_get_at<F: PrimeField>(segment: usize, c: usize, bytes: &F::Repr) -> usize {
  let skip_bits = segment * c;
  let skip_bytes = skip_bits / 8;

  if skip_bytes >= 32 {
    return 0;
  }

  let mut v = [0; 8];
  for (v, o) in v.iter_mut().zip(bytes.as_ref()[skip_bytes..].iter()) {
    *v = *o;
  }

  let mut tmp = u64::from_le_bytes(v);
  tmp >>= skip_bits - (skip_bytes * 8);
  tmp %= 1 << c;

  tmp as usize
}

struct SerialMsmPrepared<C: CurveAffine> {
  boolean_sum: C::Curve,
  reprs: Vec<<C::Scalar as PrimeField>::Repr>,
  pts_x: Vec<C::Base>,
  pts_y: Vec<C::Base>,
  max_num_bits: usize,
}

struct SharedAffineCoords<C: CurveAffine> {
  curve_a: C::Base,
  pts_x: Vec<C::Base>,
  pts_y: Vec<C::Base>,
  non_identity: Vec<bool>,
}

impl<C: CurveAffine> SharedAffineCoords<C> {
  fn new(bases: &[C]) -> Self {
    let mut pts_x = Vec::with_capacity(bases.len());
    let mut pts_y = Vec::with_capacity(bases.len());
    let mut non_identity = Vec::with_capacity(bases.len());

    for base in bases {
      if bool::from(base.is_identity()) {
        pts_x.push(C::Base::ZERO);
        pts_y.push(C::Base::ZERO);
        non_identity.push(false);
      } else {
        let coords = base.coordinates().unwrap();
        pts_x.push(*coords.x());
        pts_y.push(*coords.y());
        non_identity.push(true);
      }
    }

    Self {
      curve_a: compute_curve_a::<C>(),
      pts_x,
      pts_y,
      non_identity,
    }
  }
}

fn prepare_serial_msm_inputs<C: CurveAffine, PushPoint>(
  coeffs: &[C::Scalar],
  bases: &[C],
  mut push_point: PushPoint,
) -> SerialMsmPrepared<C>
where
  PushPoint: FnMut(usize, &C, &mut Vec<C::Base>, &mut Vec<C::Base>),
{
  let mut boolean_sum = C::Curve::identity();
  let mut reprs: Vec<<C::Scalar as PrimeField>::Repr> = Vec::new();
  let mut pts_x: Vec<C::Base> = Vec::new();
  let mut pts_y: Vec<C::Base> = Vec::new();

  let field_byte_size = <C::Scalar as PrimeField>::Repr::default().as_ref().len();
  let mut acc_or = [0u8; 32];

  for (i, (s, b)) in coeffs.iter().zip(bases.iter()).enumerate() {
    if *s == C::Scalar::ZERO || bool::from(b.is_identity()) {
      continue;
    }
    if *s == C::Scalar::ONE {
      boolean_sum += b;
      continue;
    }

    let repr = s.to_repr();
    for (a, &byte) in acc_or[..field_byte_size]
      .iter_mut()
      .zip(repr.as_ref().iter())
    {
      *a |= byte;
    }
    reprs.push(repr);
    push_point(i, b, &mut pts_x, &mut pts_y);
  }

  let max_num_bits = if reprs.is_empty() {
    0
  } else {
    let max_byte_size = field_byte_size
      - acc_or[..field_byte_size]
        .iter()
        .rev()
        .position(|v| *v != 0)
        .unwrap_or(field_byte_size);
    max_byte_size * 8
  };

  SerialMsmPrepared {
    boolean_sum,
    reprs,
    pts_x,
    pts_y,
    max_num_bits,
  }
}

fn finish_serial_msm<C: CurveAffine>(
  num_bases: usize,
  curve_a: C::Base,
  prepared: SerialMsmPrepared<C>,
) -> C::Curve {
  let SerialMsmPrepared {
    boolean_sum,
    reprs,
    pts_x,
    pts_y,
    max_num_bits,
  } = prepared;

  if reprs.is_empty() || max_num_bits == 0 {
    return boolean_sum;
  }

  let c = serial_window_size(num_bases);
  debug_assert!(c < 31, "window size c={c} would overflow 1i32 << c");

  let half = 1usize << (c - 1);
  let num_windows = max_num_bits / c + 1;
  let n_scalars = reprs.len();

  let mut signed_digits = vec![0i32; num_windows * n_scalars];
  for (i, repr) in reprs.iter().enumerate() {
    let mut carry = 0u32;
    for seg in 0..num_windows {
      let raw = serial_get_at::<C::Scalar>(seg, c, repr) as u32 + carry;
      let (digit, new_carry) = if (raw as usize) <= half {
        (raw as i32, 0)
      } else {
        (raw as i32 - (1i32 << c), 1)
      };
      signed_digits[seg * n_scalars + i] = digit;
      carry = new_carry;
    }
  }

  let non_boolean_sum = {
    let num_buckets = half;
    let mut buckets: Vec<AffineBucket<C::Base>> = vec![AffineBucket::Empty; num_buckets];
    let mut schedule = [ScheduledAdd {
      base_idx: 0,
      bucket_idx: 0,
      negate: false,
    }; BATCH_AFFINE_SIZE];
    let mut xyzz_fallback: Vec<BucketXYZZ<C::Base>> = vec![BucketXYZZ::zero(); num_buckets];
    let mut acc: BucketXYZZ<C::Base> = BucketXYZZ::zero();

    for segment in (0..num_windows).rev() {
      (0..c).for_each(|_| acc.double_in_place(curve_a));

      for b in &mut buckets {
        *b = AffineBucket::Empty;
      }
      for b in &mut xyzz_fallback {
        *b = BucketXYZZ::zero();
      }
      let mut sched_count: usize = 0;
      let mut in_schedule = vec![false; num_buckets];

      let digit_base = segment * n_scalars;
      for i in 0..n_scalars {
        let d = signed_digits[digit_base + i];
        if d == 0 {
          continue;
        }

        let (bucket_idx, negate) = if d > 0 {
          ((d as usize) - 1, false)
        } else {
          ((-d as usize) - 1, true)
        };

        if in_schedule[bucket_idx] {
          let py = if negate { -pts_y[i] } else { pts_y[i] };
          bucket_add_affine_xy::<C>(&mut xyzz_fallback[bucket_idx], pts_x[i], py, curve_a);
        } else if buckets[bucket_idx].is_empty() {
          let py = if negate { -pts_y[i] } else { pts_y[i] };
          buckets[bucket_idx] = AffineBucket::Point { x: pts_x[i], y: py };
        } else {
          schedule[sched_count] = ScheduledAdd {
            base_idx: i,
            bucket_idx,
            negate,
          };
          in_schedule[bucket_idx] = true;
          sched_count += 1;

          if sched_count == BATCH_AFFINE_SIZE {
            batch_affine_add::<C>(&mut buckets, &schedule, sched_count, &pts_x, &pts_y);
            sched_count = 0;
            in_schedule.iter_mut().for_each(|v| *v = false);
          }
        }
      }

      if sched_count > 0 {
        batch_affine_add::<C>(&mut buckets, &schedule, sched_count, &pts_x, &pts_y);
      }

      let mut running_sum: BucketXYZZ<C::Base> = BucketXYZZ::zero();
      let mut window_acc: BucketXYZZ<C::Base> = BucketXYZZ::zero();
      for idx in (0..num_buckets).rev() {
        match buckets[idx] {
          AffineBucket::Point { x, y } => {
            bucket_add_affine_xy::<C>(&mut running_sum, x, y, curve_a);
          }
          AffineBucket::Empty => {}
        }
        running_sum.add_assign_bucket(&xyzz_fallback[idx], curve_a);
        window_acc.add_assign_bucket(&running_sum, curve_a);
      }
      acc.add_assign_bucket(&window_acc, curve_a);
    }

    bucket_to_curve::<C>(&acc)
  };

  boolean_sum + non_boolean_sum
}

fn cpu_msm_serial_with_shared_coords<C: CurveAffine>(
  coeffs: &[C::Scalar],
  bases: &[C],
  shared: &SharedAffineCoords<C>,
) -> C::Curve {
  let prepared = prepare_serial_msm_inputs(coeffs, bases, |idx, _base, pts_x, pts_y| {
    if shared.non_identity[idx] {
      pts_x.push(shared.pts_x[idx]);
      pts_y.push(shared.pts_y[idx]);
    }
  });
  finish_serial_msm(coeffs.len(), shared.curve_a, prepared)
}

pub(crate) fn batch_msm_common_bases<C: CurveAffine>(
  coeffs: &[&[C::Scalar]],
  bases: &[C],
) -> Result<Vec<C::Curve>, SpartanError> {
  if coeffs.iter().any(|row| row.len() > bases.len()) {
    return Err(SpartanError::InvalidInputLength {
      reason: "Batch MSM: row length exceeds number of bases".to_string(),
    });
  }

  let shared = SharedAffineCoords::new(bases);
  Ok(
    coeffs
      .par_iter()
      .map(|row| cpu_msm_serial_with_shared_coords(row, &bases[..row.len()], &shared))
      .collect(),
  )
}

// ==================================================================================
// Main MSM with signed decomposition + bit-width partitioning
// ==================================================================================

// ==================================================================================
// Batch affine bucket accumulation
// ==================================================================================

/// Batch size for Montgomery batch inversion in bucket accumulation.
const BATCH_AFFINE_SIZE: usize = 64;

/// Affine bucket for batch affine MSM. Stores None (empty) or an affine point.
#[derive(Clone, Copy)]
enum AffineBucket<F: Field> {
  Empty,
  Point { x: F, y: F },
}

impl<F: Field> AffineBucket<F> {
  #[inline]
  fn is_empty(&self) -> bool {
    matches!(self, AffineBucket::Empty)
  }
}

/// A scheduled point addition for batch processing.
#[derive(Clone, Copy)]
struct ScheduledAdd {
  base_idx: usize,
  bucket_idx: usize,
  negate: bool,
}

/// Process a batch of scheduled affine additions using Montgomery's trick.
/// Each addition computes: bucket += point (or bucket -= point if negate).
/// Uses a single batch inversion for all additions in the batch.
/// Cost: ~4M per addition (amortized) vs 7M+2S for XYZZ mixed addition.
fn batch_affine_add<C: CurveAffine>(
  buckets: &mut [AffineBucket<C::Base>],
  schedule: &[ScheduledAdd],
  count: usize,
  pts_x: &[C::Base],
  pts_y: &[C::Base],
) {
  if count == 0 {
    return;
  }

  // Phase 1: compute all denominators and accumulate for batch inversion
  let mut denoms = [C::Base::ZERO; BATCH_AFFINE_SIZE];
  let mut lambdas = [C::Base::ZERO; BATCH_AFFINE_SIZE];
  let mut acc = C::Base::ONE;

  for i in 0..count {
    let s = &schedule[i];
    let bkt = &buckets[s.bucket_idx];

    match bkt {
      AffineBucket::Empty => {
        // Will be handled in phase 2 directly
        denoms[i] = C::Base::ONE; // placeholder
      }
      AffineBucket::Point { x: bx, y: by } => {
        let py = if s.negate {
          -pts_y[s.base_idx]
        } else {
          pts_y[s.base_idx]
        };
        let px = pts_x[s.base_idx];

        if *bx == px {
          if *by == py {
            // Doubling case: denominator = 2*y
            let denom = by.double();
            denoms[i] = denom;
            lambdas[i] = acc * (px.square().double() + px.square()); // 3x^2 (for a=0 curves)
            acc *= denom;
          } else {
            // Point at infinity case (P + (-P))
            denoms[i] = C::Base::ONE; // placeholder
          }
        } else {
          // Regular addition: denominator = x2 - x1
          let denom = px - *bx;
          denoms[i] = denom;
          lambdas[i] = acc * (py - *by);
          acc *= denom;
        }
      }
    }
  }

  // Phase 2: batch inversion using Montgomery's trick
  let acc_inv = acc.invert().unwrap_or(C::Base::ONE);
  let mut running_inv = acc_inv;

  // Process in reverse to recover individual inverses
  for i in (0..count).rev() {
    let s = &schedule[i];
    let bkt = &buckets[s.bucket_idx];

    match bkt {
      AffineBucket::Empty => {
        // Direct assignment
        let py = if s.negate {
          -pts_y[s.base_idx]
        } else {
          pts_y[s.base_idx]
        };
        buckets[s.bucket_idx] = AffineBucket::Point {
          x: pts_x[s.base_idx],
          y: py,
        };
      }
      AffineBucket::Point { x: bx, y: by } => {
        let py = if s.negate {
          -pts_y[s.base_idx]
        } else {
          pts_y[s.base_idx]
        };
        let px = pts_x[s.base_idx];

        if *bx == px && *by != py {
          // P + (-P) = O
          buckets[s.bucket_idx] = AffineBucket::Empty;
          // Don't update running_inv since denom was placeholder
        } else if *bx == px && *by == py {
          // Doubling
          let lambda = lambdas[i] * running_inv;
          running_inv *= denoms[i];

          let x3 = lambda.square() - bx.double();
          let y3 = lambda * (*bx - x3) - *by;
          buckets[s.bucket_idx] = AffineBucket::Point { x: x3, y: y3 };
        } else {
          // Regular addition
          let lambda = lambdas[i] * running_inv;
          running_inv *= denoms[i];

          let x3 = lambda.square() - *bx - px;
          let y3 = lambda * (*bx - x3) - *by;
          buckets[s.bucket_idx] = AffineBucket::Point { x: x3, y: y3 };
        }
      }
    }
  }
}

/// Serial windowed Pippenger MSM with batch affine bucket accumulation.
///
/// Uses Booth encoding (signed digits) with batch affine additions using
/// Montgomery's trick. Each bucket addition costs ~4M (amortized) instead of
/// 7M+2S for XYZZ mixed addition. Designed to run without spawning rayon tasks.
fn cpu_msm_serial<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
  let curve_a = compute_curve_a::<C>();
  let prepared = prepare_serial_msm_inputs(coeffs, bases, |_idx, base, pts_x, pts_y| {
    let coords = base.coordinates().unwrap();
    pts_x.push(*coords.x());
    pts_y.push(*coords.y());
  });
  finish_serial_msm(bases.len(), curve_a, prepared)
}

/// Simple MSM fallback for very small inputs.
fn msm_simple<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
  coeffs
    .iter()
    .zip(bases.iter())
    .fold(C::Curve::identity(), |acc, (coeff, base)| {
      acc + *base * coeff
    })
}

/// Accumulate bases (sum of affine points, for binary MSM).
fn accumulate_bases<C: CurveAffine>(bases: &[C]) -> C::Curve {
  let num_threads = current_num_threads();
  if bases.is_empty() {
    return C::Curve::identity();
  }
  if bases.len() > num_threads {
    let chunk = bases.len().div_ceil(num_threads);
    bases
      .par_chunks(chunk)
      .map(|chunk| {
        chunk.iter().fold(C::Curve::identity(), |mut acc, b| {
          acc += *b;
          acc
        })
      })
      .reduce(C::Curve::identity, |a, b| a + b)
  } else {
    bases.iter().fold(C::Curve::identity(), |mut acc, b| {
      acc += *b;
      acc
    })
  }
}

/// Performs an optimized multi-scalar multiplication for full field element scalars.
///
/// Uses signed scalar decomposition to halve the effective scalar range,
/// then partitions scalars by bit-width to route each group to the optimal MSM algorithm.
/// Small scalars (≤64 bits) use bucket-sort MSMs with XYZZ coordinates;
/// large scalars delegate to halo2curves' `msm_best`.
///
/// # Errors
/// Returns `SpartanError::InvalidInputLength` if coeffs and bases have different lengths.
/// Parallel MSM designed for standalone (non-nested) contexts.
/// Uses halo2curves' parallel MSM which spawns its own rayon threads.
/// Call this when NOT inside an existing par_iter (e.g., PCS prove, IPA prove).
pub fn msm_standalone<C: CurveAffine>(
  coeffs: &[C::Scalar],
  bases: &[C],
) -> Result<C::Curve, SpartanError> {
  if coeffs.len() != bases.len() {
    return Err(SpartanError::InvalidInputLength {
      reason: "MSM: Coefficients and bases must have the same length".to_string(),
    });
  }
  if coeffs.is_empty() {
    return Ok(C::Curve::identity());
  }
  Ok(halo2curves::msm::msm_best(coeffs, bases))
}

pub fn msm<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> Result<C::Curve, SpartanError> {
  let (_msm_span, msm_t) = start_span!("msm", size = coeffs.len());

  if coeffs.len() != bases.len() {
    return Err(SpartanError::InvalidInputLength {
      reason: "MSM: Coefficients and bases must have the same length".to_string(),
    });
  }

  let n = coeffs.len();
  if n == 0 {
    return Ok(C::Curve::identity());
  }

  // For very small inputs, use the simple fallback
  if n <= 16 {
    return Ok(msm_simple(coeffs, bases));
  }

  // For moderate inputs, the signed-decomposition + classification overhead
  // is not worth it. Use serial Pippenger with XYZZ buckets — this avoids
  // nested rayon parallelism when the caller already parallelizes externally
  // (e.g. Hyrax commit runs 38 row MSMs in parallel).
  if n <= 8192 {
    let result = cpu_msm_serial(coeffs, bases);
    if msm_t.elapsed().as_millis() > 10 {
      info!(elapsed_ms = %msm_t.elapsed().as_millis(), size = coeffs.len(), "msm");
    }
    return Ok(result);
  }

  // Group indices: 0=unit_pos, 1=unit_neg, 2=pos≤8, 3=neg≤8,
  // 4=pos≤16, 5=neg≤16, 6=pos≤32, 7=neg≤32, 8=pos≤64, 9=neg≤64, 10=large
  const NUM_GROUPS: usize = 11;

  // Phase 1: Classify each scalar in parallel
  // Encode as u64: top 4 bits = group, bottom 60 bits = original index
  let classified: Vec<u64> = coeffs
    .par_iter()
    .enumerate()
    .filter_map(|(i, s)| {
      if bool::from(s.is_zero()) || bool::from(bases[i].is_identity()) {
        return None;
      }
      let neg_s = -(*s);
      let bits_s = scalar_num_bits(s);
      let bits_neg = scalar_num_bits(&neg_s);

      let group = if bits_s <= 1 {
        0u8 // unit positive
      } else if bits_neg <= 1 {
        1u8 // unit negative
      } else if bits_s <= 8 {
        2u8
      } else if bits_neg <= 8 {
        3u8
      } else if bits_s <= 16 {
        4u8
      } else if bits_neg <= 16 {
        5u8
      } else if bits_s <= 32 {
        6u8
      } else if bits_neg <= 32 {
        7u8
      } else if bits_s <= 64 {
        8u8
      } else if bits_neg <= 64 {
        9u8
      } else {
        10u8 // large
      };
      Some(((i as u64) & 0x0FFF_FFFF_FFFF_FFFF) | ((group as u64) << 60))
    })
    .collect();

  if classified.is_empty() {
    return Ok(C::Curve::identity());
  }

  // Phase 2: Sort by group for efficient partitioning
  let mut classified = classified;
  classified.par_sort_unstable_by_key(|v| (v >> 60) as u8);

  let extract_group = |v: u64| (v >> 60) as u8;
  let extract_index = |v: u64| (v & 0x0FFF_FFFF_FFFF_FFFF) as usize;

  // Find partition boundaries
  let mut boundaries = [0usize; NUM_GROUPS + 1];
  {
    let mut pos = 0;
    for g in 0..NUM_GROUPS as u8 {
      boundaries[g as usize] = pos;
      pos += classified[pos..].partition_point(|v| extract_group(*v) <= g);
    }
    boundaries[NUM_GROUPS] = classified.len();
  }

  // Helper to extract (bases, u64_scalars) for a group range
  let extract_u64_group = |start: usize, end: usize, negate: bool| -> (Vec<C>, Vec<u64>) {
    classified[start..end]
      .iter()
      .map(|&v| {
        let idx = extract_index(v);
        let b = bases[idx];
        let s = if negate { -coeffs[idx] } else { coeffs[idx] };
        (b, repr_low_u64(&s))
      })
      .unzip()
  };

  // Helper to extract bases for unit groups
  let extract_binary_group = |start: usize, end: usize| -> Vec<C> {
    classified[start..end]
      .iter()
      .map(|&v| bases[extract_index(v)])
      .collect()
  };

  // Phase 3: Compute MSM for each group in parallel
  let (g0_start, g0_end) = (boundaries[0], boundaries[1]);
  let (g1_start, g1_end) = (boundaries[1], boundaries[2]);
  let (g2_start, g2_end) = (boundaries[2], boundaries[3]);
  let (g3_start, g3_end) = (boundaries[3], boundaries[4]);
  let (g4_start, g4_end) = (boundaries[4], boundaries[5]);
  let (g5_start, g5_end) = (boundaries[5], boundaries[6]);
  let (g6_start, g6_end) = (boundaries[6], boundaries[7]);
  let (g7_start, g7_end) = (boundaries[7], boundaries[8]);
  let (g8_start, g8_end) = (boundaries[8], boundaries[9]);
  let (g9_start, g9_end) = (boundaries[9], boundaries[10]);
  let (g10_start, g10_end) = (boundaries[10], boundaries[11]);

  // Execute all groups in parallel using nested rayon joins
  let (binary_result, small_and_large_result) = rayon::join(
    || {
      let (pos, neg) = rayon::join(
        || {
          let bases_pos = extract_binary_group(g0_start, g0_end);
          accumulate_bases::<C>(&bases_pos)
        },
        || {
          let bases_neg = extract_binary_group(g1_start, g1_end);
          accumulate_bases::<C>(&bases_neg)
        },
      );
      pos - neg
    },
    || {
      let (small_result, large_result) = rayon::join(
        || {
          let ((r8, r16), (r32, r64)) = rayon::join(
            || {
              rayon::join(
                || {
                  let (pos_b, pos_s) = extract_u64_group(g2_start, g2_end, false);
                  let (neg_b, neg_s) = extract_u64_group(g3_start, g3_end, true);
                  msm_small_with_max_num_bits(&pos_s, &pos_b, 8)
                    - msm_small_with_max_num_bits(&neg_s, &neg_b, 8)
                },
                || {
                  let (pos_b, pos_s) = extract_u64_group(g4_start, g4_end, false);
                  let (neg_b, neg_s) = extract_u64_group(g5_start, g5_end, true);
                  msm_small_with_max_num_bits(&pos_s, &pos_b, 16)
                    - msm_small_with_max_num_bits(&neg_s, &neg_b, 16)
                },
              )
            },
            || {
              rayon::join(
                || {
                  let (pos_b, pos_s) = extract_u64_group(g6_start, g6_end, false);
                  let (neg_b, neg_s) = extract_u64_group(g7_start, g7_end, true);
                  msm_small_with_max_num_bits(&pos_s, &pos_b, 32)
                    - msm_small_with_max_num_bits(&neg_s, &neg_b, 32)
                },
                || {
                  let (pos_b, pos_s) = extract_u64_group(g8_start, g8_end, false);
                  let (neg_b, neg_s) = extract_u64_group(g9_start, g9_end, true);
                  msm_small_with_max_num_bits(&pos_s, &pos_b, 64)
                    - msm_small_with_max_num_bits(&neg_s, &neg_b, 64)
                },
              )
            },
          );
          r8 + r16 + r32 + r64
        },
        || {
          // Large scalars: delegate to halo2curves' optimized MSM
          if g10_start >= g10_end {
            return C::Curve::identity();
          }
          let (large_bases, large_coeffs): (Vec<C>, Vec<C::Scalar>) = classified
            [g10_start..g10_end]
            .iter()
            .map(|&v| {
              let idx = extract_index(v);
              (bases[idx], coeffs[idx])
            })
            .unzip();
          halo2curves::msm::msm_best(&large_coeffs, &large_bases)
        },
      );
      small_result + large_result
    },
  );

  let result = binary_result + small_and_large_result;

  if msm_t.elapsed().as_millis() > 10 {
    info!(elapsed_ms = %msm_t.elapsed().as_millis(), size = coeffs.len(), "msm");
  }
  Ok(result)
}

fn num_bits(n: usize) -> usize {
  if n == 0 { 0 } else { (n.ilog2() + 1) as usize }
}

// ==================================================================================
// Small-scalar MSM with XYZZ buckets
// ==================================================================================

/// Internal helper: MSM for small scalars with a known max bit-width.
fn msm_small_with_max_num_bits<
  C: CurveAffine,
  T: Integer + Into<u64> + Copy + Sync + ToPrimitive,
>(
  scalars: &[T],
  bases: &[C],
  max_num_bits: usize,
) -> C::Curve {
  if scalars.is_empty() {
    return C::Curve::identity();
  }
  assert_eq!(bases.len(), scalars.len());

  match max_num_bits {
    0 => C::identity().into(),
    1 => msm_binary(scalars, bases),
    2..=10 => msm_10(scalars, bases, max_num_bits),
    11..=32 => msm_small_rest(scalars, bases, max_num_bits),
    _ => {
      // For >32-bit scalars, halo2curves' msm_best is faster than our
      // bucket-sort Pippenger (e.g., 192ms vs 244ms at u64, 2^20 points).
      let field_scalars: Vec<C::ScalarExt> = scalars
        .iter()
        .map(|s| C::ScalarExt::from((*s).into()))
        .collect();
      halo2curves::msm::msm_best(&field_scalars, bases)
    }
  }
}

/// Multi-scalar multiplication using the best algorithm for the given scalars.
///
/// # Errors
/// Returns `SpartanError::InvalidInputLength` if bases and scalars have different lengths.
/// Returns `SpartanError::InternalError` if scalars contain values that cannot be processed.
pub fn msm_small<C: CurveAffine, T: Integer + Into<u64> + Copy + Sync + ToPrimitive>(
  scalars: &[T],
  bases: &[C],
) -> Result<C::Curve, SpartanError> {
  let (_msm_small_span, msm_small_t) = start_span!("msm_small", size = scalars.len());

  if bases.len() != scalars.len() {
    return Err(SpartanError::InvalidInputLength {
      reason: "MSM Small: Coefficients and bases must have the same length".to_string(),
    });
  }

  let max_scalar = scalars.iter().max().ok_or(SpartanError::InternalError {
    reason: "Unable to find maximum value".to_string(),
  })?;
  let max_scalar_usize = max_scalar.to_usize().ok_or(SpartanError::InternalError {
    reason: "Unable to convert maximum value to usize".to_string(),
  })?;
  let max_num_bits = num_bits(max_scalar_usize);
  let result = match max_num_bits {
    0 => C::identity().into(),
    1 => {
      let (_binary_span, binary_t) = start_span!("msm_binary");
      let result = msm_binary(scalars, bases);
      if binary_t.elapsed().as_millis() != 0 {
        info!(elapsed_ms = %binary_t.elapsed().as_millis(), size = scalars.len(), "msm_binary");
      }
      result
    }
    2..=10 => {
      let (_msm_10_span, msm_10_t) = start_span!("msm_10", max_bits = max_num_bits);
      let result = msm_10(scalars, bases, max_num_bits);
      info!(elapsed_ms = %msm_10_t.elapsed().as_millis(), max_bits = max_num_bits, "msm_10");
      result
    }
    _ => {
      let (_msm_rest_span, msm_rest_t) = start_span!("msm_small_rest", max_bits = max_num_bits);
      let result = msm_small_rest(scalars, bases, max_num_bits);
      info!(elapsed_ms = %msm_rest_t.elapsed().as_millis(), max_bits = max_num_bits, "msm_small_rest");
      result
    }
  };

  if msm_small_t.elapsed().as_millis() != 0 {
    info!(elapsed_ms = %msm_small_t.elapsed().as_millis(), size = scalars.len(), max_bits = max_num_bits, "msm_small");
  }
  Ok(result)
}

#[inline(always)]
fn msm_binary<C: CurveAffine, T: Integer + Sync>(scalars: &[T], bases: &[C]) -> C::Curve {
  assert_eq!(scalars.len(), bases.len());
  let num_threads = current_num_threads();
  let process_chunk = |scalars: &[T], bases: &[C]| {
    let mut acc = C::Curve::identity();
    scalars
      .iter()
      .zip(bases.iter())
      .filter(|(scalar, _)| !scalar.is_zero())
      .for_each(|(_, base)| {
        acc += *base;
      });
    acc
  };

  if scalars.len() > num_threads {
    let chunk = scalars.len() / num_threads;
    scalars
      .par_chunks(chunk)
      .zip(bases.par_chunks(chunk))
      .map(|(scalars, bases)| process_chunk(scalars, bases))
      .reduce(C::Curve::identity, |sum, evl| sum + evl)
  } else {
    process_chunk(scalars, bases)
  }
}

/// MSM for boolean scalars: sum bases where bit is true.
///
/// Pippenger bucketing is useless for binary scalars — there's only 1 nonzero bucket.
/// This directly sums the matching bases with parallel chunking.
pub fn msm_bool<C: CurveAffine>(bits: &[bool], bases: &[C]) -> Result<C::Curve, SpartanError> {
  if bits.len() != bases.len() {
    return Err(SpartanError::InvalidInputLength {
      reason: "msm_bool: bits and bases must have the same length".to_string(),
    });
  }

  let num_threads = if bits.len() > 1024 {
    current_num_threads()
  } else {
    1
  };

  let process_chunk = |bits: &[bool], bases: &[C]| {
    bits.iter().zip(bases.iter()).fold(
      C::Curve::identity(),
      |acc, (&bit, base)| {
        if bit { acc + base } else { acc }
      },
    )
  };

  let result = if bits.len() > num_threads {
    let chunk = bits.len() / num_threads;
    bits
      .par_chunks(chunk)
      .zip(bases.par_chunks(chunk))
      .map(|(b, g)| process_chunk(b, g))
      .reduce(C::Curve::identity, |a, b| a + b)
  } else {
    process_chunk(bits, bases)
  };

  Ok(result)
}

/// MSM optimized for up to 10-bit scalars, using XYZZ bucket coordinates.
#[inline(always)]
fn msm_10<C: CurveAffine, T: Into<u64> + Zero + Copy + Sync>(
  scalars: &[T],
  bases: &[C],
  max_num_bits: usize,
) -> C::Curve {
  fn msm_10_serial<C: CurveAffine, T: Into<u64> + Zero + Copy>(
    scalars: &[T],
    bases: &[C],
    max_num_bits: usize,
  ) -> C::Curve {
    let curve_a = compute_curve_a::<C>();
    let num_buckets: usize = 1 << max_num_bits;
    let mut buckets: Vec<BucketXYZZ<C::Base>> = vec![BucketXYZZ::zero(); num_buckets];

    scalars
      .iter()
      .zip(bases.iter())
      .filter(|(scalar, _base)| !scalar.is_zero())
      .for_each(|(scalar, base)| {
        let bucket_index: u64 = (*scalar).into();
        bucket_add_affine::<C>(&mut buckets[bucket_index as usize], base, curve_a);
      });

    let mut result: BucketXYZZ<C::Base> = BucketXYZZ::zero();
    let mut running_sum: BucketXYZZ<C::Base> = BucketXYZZ::zero();
    for b in buckets.into_iter().skip(1).rev() {
      running_sum.add_assign_bucket(&b, curve_a);
      result.add_assign_bucket(&running_sum, curve_a);
    }
    bucket_to_curve::<C>(&result)
  }

  let num_threads = current_num_threads();
  if scalars.len() > num_threads {
    let chunk_size = scalars.len() / num_threads;
    scalars
      .par_chunks(chunk_size)
      .zip(bases.par_chunks(chunk_size))
      .map(|(scalars_chunk, bases_chunk)| msm_10_serial(scalars_chunk, bases_chunk, max_num_bits))
      .reduce(C::Curve::identity, |sum, evl| sum + evl)
  } else {
    msm_10_serial(scalars, bases, max_num_bits)
  }
}

#[inline(always)]
fn msm_small_rest<C: CurveAffine, T: Into<u64> + Zero + Copy + Sync>(
  scalars: &[T],
  bases: &[C],
  max_num_bits: usize,
) -> C::Curve {
  fn msm_small_rest_serial<C: CurveAffine, T: Into<u64> + Zero + Copy>(
    scalars: &[T],
    bases: &[C],
    max_num_bits: usize,
  ) -> C::Curve {
    let curve_a = compute_curve_a::<C>();
    let mut c = if bases.len() < 32 {
      3
    } else {
      compute_ln(bases.len()) + 2
    };

    if max_num_bits == 32 || max_num_bits == 64 {
      c = 8;
    }

    let scalars_and_bases_iter = scalars.iter().zip(bases).filter(|(s, _base)| !s.is_zero());
    let window_starts: Vec<usize> = (0..max_num_bits).step_by(c).collect();

    // Each window is of size `c`.
    // We divide up the bits 0..num_bits into windows of size `c`, and
    // process each such window.
    let window_sums: Vec<C::CurveExt> = window_starts
      .iter()
      .map(|&w_start| {
        let mut res: BucketXYZZ<C::Base> = BucketXYZZ::zero();
        // We don't need the "zero" bucket, so we only have 2^c - 1 buckets.
        let mut buckets: Vec<BucketXYZZ<C::Base>> = vec![BucketXYZZ::zero(); (1 << c) - 1];
        // This clone is cheap, because the iterator contains just a
        // pointer and an index into the original vectors.
        scalars_and_bases_iter.clone().for_each(|(&scalar, base)| {
          let scalar: u64 = scalar.into();
          if scalar == 1 {
            // We only process unit scalars once in the first window.
            if w_start == 0 {
              bucket_add_affine::<C>(&mut res, base, curve_a);
            }
          } else {
            let mut scalar = scalar;

            // We right-shift by w_start, thus getting rid of the
            // lower bits.
            scalar >>= w_start;

            // We mod the remaining bits by 2^{window size}, thus taking `c` bits.
            scalar %= 1 << c;

            // If the scalar is non-zero, we update the corresponding
            // bucket.
            // (Recall that `buckets` doesn't have a zero bucket.)
            if scalar != 0 {
              bucket_add_affine::<C>(&mut buckets[(scalar - 1) as usize], base, curve_a);
            }
          }
        });

        // Prefix sum using XYZZ coordinates
        let mut running_sum: BucketXYZZ<C::Base> = BucketXYZZ::zero();
        for b in buckets.into_iter().rev() {
          running_sum.add_assign_bucket(&b, curve_a);
          res.add_assign_bucket(&running_sum, curve_a);
        }
        bucket_to_curve::<C>(&res)
      })
      .collect();

    // We store the sum for the lowest window.
    let lowest = *window_sums.first().unwrap();

    // We're traversing windows from high to low.
    lowest
      + window_sums[1..]
        .iter()
        .rev()
        .fold(C::CurveExt::identity(), |mut total, sum_i| {
          total += sum_i;
          for _ in 0..c {
            total = total.double();
          }
          total
        })
  }

  let num_threads = current_num_threads();
  if scalars.len() > num_threads {
    let chunk_size = scalars.len() / num_threads;
    scalars
      .par_chunks(chunk_size)
      .zip(bases.par_chunks(chunk_size))
      .map(|(scalars_chunk, bases_chunk)| {
        msm_small_rest_serial(scalars_chunk, bases_chunk, max_num_bits)
      })
      .reduce(C::Curve::identity, |sum, evl| sum + evl)
  } else {
    msm_small_rest_serial(scalars, bases, max_num_bits)
  }
}

/// Multi-scalar multiplication for signed small scalars (e.g. {-1, 0, 1, 2}).
///
/// Uses separate positive/negative bucket accumulators with summation-by-parts
/// and XYZZ bucket coordinates.
/// For witnesses in {-1, 0, 1, 2}, this uses only 3 buckets total.
///
/// # Errors
/// Returns `SpartanError::InvalidInputLength` if bases and scalars have different lengths.
pub fn msm_signed_small<C: CurveAffine>(
  scalars: &[i8],
  bases: &[C],
) -> Result<C::Curve, SpartanError> {
  if bases.len() != scalars.len() {
    return Err(SpartanError::InvalidInputLength {
      reason: "MSM Signed Small: Coefficients and bases must have the same length".to_string(),
    });
  }

  if scalars.is_empty() {
    return Ok(C::Curve::identity());
  }

  let num_threads = current_num_threads();

  if scalars.len() > num_threads {
    let chunk_size = scalars.len() / num_threads;
    Ok(
      scalars
        .par_chunks(chunk_size)
        .zip(bases.par_chunks(chunk_size))
        .map(|(s, b)| msm_signed_small_serial(s, b))
        .reduce(C::Curve::identity, |sum, evl| sum + evl),
    )
  } else {
    Ok(msm_signed_small_serial(scalars, bases))
  }
}

fn msm_signed_small_serial<C: CurveAffine>(scalars: &[i8], bases: &[C]) -> C::Curve {
  let curve_a = compute_curve_a::<C>();
  let mut max_pos: i8 = 0;
  let mut max_neg: i8 = 0;
  for &s in scalars {
    if s > max_pos {
      max_pos = s;
    } else if s < max_neg {
      max_neg = s;
    }
  }

  if max_pos == 0 && max_neg == 0 {
    return C::Curve::identity();
  }

  let num_pos_buckets = max_pos as usize;
  let num_neg_buckets = (-max_neg) as usize;

  let mut pos_buckets: Vec<BucketXYZZ<C::Base>> = vec![BucketXYZZ::zero(); num_pos_buckets];
  let mut neg_buckets: Vec<BucketXYZZ<C::Base>> = vec![BucketXYZZ::zero(); num_neg_buckets];

  for (&scalar, base) in scalars.iter().zip(bases.iter()) {
    if scalar > 0 {
      bucket_add_affine::<C>(&mut pos_buckets[(scalar - 1) as usize], base, curve_a);
    } else if scalar < 0 {
      bucket_add_affine::<C>(&mut neg_buckets[(-scalar - 1) as usize], base, curve_a);
    }
  }

  // Summation-by-parts for positive buckets
  let mut pos_result: BucketXYZZ<C::Base> = BucketXYZZ::zero();
  let mut running_sum: BucketXYZZ<C::Base> = BucketXYZZ::zero();
  for bucket in pos_buckets.into_iter().rev() {
    running_sum.add_assign_bucket(&bucket, curve_a);
    pos_result.add_assign_bucket(&running_sum, curve_a);
  }

  // Summation-by-parts for negative buckets
  let mut neg_result: BucketXYZZ<C::Base> = BucketXYZZ::zero();
  running_sum = BucketXYZZ::zero();
  for bucket in neg_buckets.into_iter().rev() {
    running_sum.add_assign_bucket(&bucket, curve_a);
    neg_result.add_assign_bucket(&running_sum, curve_a);
  }

  bucket_to_curve::<C>(&pos_result) - bucket_to_curve::<C>(&neg_result)
}

#[inline(always)]
fn compute_ln(a: usize) -> usize {
  // log2(a) * ln(2)
  if a == 0 {
    0 // Handle edge case where log2 is undefined
  } else {
    a.ilog2() as usize * 69 / 100
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::pasta::{pallas, vesta};
  use ff::Field;
  use halo2curves::{CurveAffine, group::Group};
  use rand_core::OsRng;

  fn test_general_msm_with<F: Field, A: CurveAffine<ScalarExt = F>>(n: usize) {
    let coeffs = (0..n).map(|_| F::random(OsRng)).collect::<Vec<_>>();
    let bases = (0..n)
      .map(|_| A::from(A::generator() * F::random(OsRng)))
      .collect::<Vec<_>>();

    assert_eq!(coeffs.len(), bases.len());
    let naive = coeffs
      .iter()
      .zip(bases.iter())
      .fold(A::CurveExt::identity(), |acc, (coeff, base)| {
        acc + *base * coeff
      });
    let msm = msm(&coeffs, &bases);

    assert_eq!(naive, msm.unwrap())
  }

  #[test]
  fn test_general_msm() {
    test_general_msm_with::<pallas::Scalar, pallas::Affine>(8);
    test_general_msm_with::<vesta::Scalar, vesta::Affine>(8);
  }

  #[test]
  fn test_general_msm_128() {
    // n=128 exercises signed-digit code path with larger window sizes
    test_general_msm_with::<pallas::Scalar, pallas::Affine>(128);
    test_general_msm_with::<vesta::Scalar, vesta::Affine>(128);
  }

  fn test_msm_ux_with<F: PrimeField, A: CurveAffine<ScalarExt = F>>() {
    let n = 8;
    let bases = (0..n)
      .map(|_| A::from(A::generator() * F::random(OsRng)))
      .collect::<Vec<_>>();

    for bit_width in [1, 4, 8, 10, 16, 20, 32, 40, 64] {
      println!("bit_width: {bit_width}");
      assert!(bit_width <= 64); // Ensure we don't overflow F::from
      let mask = if bit_width == 64 {
        u64::MAX
      } else {
        (1u64 << bit_width) - 1
      };
      let coeffs: Vec<u64> = (0..n)
        .map(|_| rand::random::<u64>() & mask)
        .collect::<Vec<_>>();
      let coeffs_scalar: Vec<F> = coeffs.iter().map(|b| F::from(*b)).collect::<Vec<_>>();
      let general = msm(&coeffs_scalar, &bases);
      let integer = msm_small(&coeffs, &bases);

      assert_eq!(general.unwrap(), integer.unwrap());
    }
  }

  #[test]
  fn test_msm_ux() {
    test_msm_ux_with::<pallas::Scalar, pallas::Affine>();
    test_msm_ux_with::<vesta::Scalar, vesta::Affine>();
  }

  fn test_msm_signed_small_with<F: PrimeField, A: CurveAffine<ScalarExt = F>>() {
    let n = 64;
    let bases = (0..n)
      .map(|_| A::from(A::generator() * F::random(OsRng)))
      .collect::<Vec<_>>();

    // Test {-1, 0, 1, 2} scalars
    let scalars: Vec<i8> = (0..n).map(|i| (i % 4) as i8 - 1).collect(); // -1, 0, 1, 2, -1, 0, ...

    let naive =
      scalars
        .iter()
        .zip(bases.iter())
        .fold(A::CurveExt::identity(), |acc, (&s, base)| {
          if s == 0 {
            acc
          } else if s > 0 {
            acc + *base * F::from(s as u64)
          } else {
            acc - *base * F::from((-s) as u64)
          }
        });

    let result = msm_signed_small(&scalars, &bases).unwrap();
    assert_eq!(naive, result);

    // Test all zeros
    let zeros = vec![0i8; n];
    assert_eq!(
      msm_signed_small(&zeros, &bases).unwrap(),
      A::CurveExt::identity()
    );

    // Test all positive
    let pos = vec![1i8; n];
    let naive_pos = bases
      .iter()
      .fold(A::CurveExt::identity(), |acc, base| acc + base);
    assert_eq!(msm_signed_small(&pos, &bases).unwrap(), naive_pos);

    // Test all negative
    let neg = vec![-1i8; n];
    assert_eq!(msm_signed_small(&neg, &bases).unwrap(), -naive_pos);
  }

  #[test]
  fn test_msm_signed_small() {
    test_msm_signed_small_with::<pallas::Scalar, pallas::Affine>();
    test_msm_signed_small_with::<vesta::Scalar, vesta::Affine>();
  }

  /// Regression test: MSM must handle identity bases without panicking.
  fn test_msm_identity_bases_with<F: Field, A: CurveAffine<ScalarExt = F>>() {
    let n = 32; // Must be >16 to exercise the signed-decomposition path
    let mut coeffs = (0..n).map(|_| F::random(OsRng)).collect::<Vec<_>>();
    let mut bases = (0..n)
      .map(|_| A::from(A::generator() * F::random(OsRng)))
      .collect::<Vec<_>>();

    // Replace a few bases with identity and give them non-zero scalars
    bases[0] = A::identity();
    bases[3] = A::identity();
    bases[n - 1] = A::identity();
    coeffs[0] = F::ONE;
    coeffs[3] = F::random(OsRng);

    let naive = coeffs
      .iter()
      .zip(bases.iter())
      .fold(A::CurveExt::identity(), |acc, (coeff, base)| {
        acc + *base * coeff
      });
    let result = msm(&coeffs, &bases);

    assert_eq!(naive, result.unwrap());
  }

  #[test]
  fn test_msm_identity_bases() {
    test_msm_identity_bases_with::<pallas::Scalar, pallas::Affine>();
    test_msm_identity_bases_with::<vesta::Scalar, vesta::Affine>();
  }

  fn test_batch_msm_common_bases_with<F: Field, A: CurveAffine<ScalarExt = F>>() {
    let n = 128;
    let bases = (0..n)
      .map(|_| A::from(A::generator() * F::random(OsRng)))
      .collect::<Vec<_>>();
    let coeff_rows = vec![
      (0..n).map(|_| F::random(OsRng)).collect::<Vec<_>>(),
      (0..n).map(|_| F::random(OsRng)).collect::<Vec<_>>(),
      (0..(n - 17)).map(|_| F::random(OsRng)).collect::<Vec<_>>(),
    ];
    let coeff_refs = coeff_rows.iter().map(Vec::as_slice).collect::<Vec<_>>();

    let expected = coeff_rows
      .iter()
      .map(|row| msm(row, &bases[..row.len()]).unwrap())
      .collect::<Vec<_>>();
    let actual = batch_msm_common_bases(&coeff_refs, &bases).unwrap();

    assert_eq!(expected, actual);
  }

  #[test]
  fn test_batch_msm_common_bases() {
    test_batch_msm_common_bases_with::<pallas::Scalar, pallas::Affine>();
    test_batch_msm_common_bases_with::<vesta::Scalar, vesta::Affine>();
  }
}
