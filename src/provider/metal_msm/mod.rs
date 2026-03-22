//! Metal GPU-accelerated Multi-Scalar Multiplication (MSM) for Pallas curve.
//!
//! Adapted from [zkmopro/gpu-acceleration](https://github.com/zkmopro/gpu-acceleration).

#[allow(missing_docs)]
pub mod host;
#[allow(missing_docs)]
pub mod metal_msm;
#[allow(missing_docs)]
pub mod utils;

pub use metal_msm::metal_msm_pallas;
