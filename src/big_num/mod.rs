// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT

//! Big-number arithmetic for optimized sumcheck.
//!
//! This module provides wide-integer accumulators and delayed modular reduction
//! for field × field and field × small-value products, enabling faster sumcheck
//! by batching Montgomery and Barrett reduction operations.

pub(crate) mod barrett;
pub(crate) mod delayed_reduction;
pub(crate) mod field_reduction_constants;
pub(crate) mod macros;
pub(crate) mod montgomery;
pub(crate) mod small_value_field;
pub(crate) mod wide_mul;

mod limbs;

pub use delayed_reduction::DelayedReduction;
pub use field_reduction_constants::FieldReductionConstants;
#[allow(unused_imports)] // TODO: Remove when small_sumcheck module is added
pub use limbs::{SignedWideLimbs, SubMagResult, WideLimbs, sub_mag};
#[allow(unused_imports)] // TODO: Remove when small_sumcheck module is added
pub use small_value_field::{SmallValueField, i64_to_field, try_field_to_i64, vec_to_small};
#[allow(unused_imports)] // TODO: Remove when small_sumcheck module is added
pub use wide_mul::WideMul;
