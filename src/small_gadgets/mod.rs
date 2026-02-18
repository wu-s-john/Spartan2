//! Small-value gadgets for R1CS circuits.
//!
//! This module provides gadgets (Boolean, UInt32, SHA-256) that work with
//! the small-value constraint system, using native integer types instead
//! of field elements.
//!
//! # Available Gadgets
//!
//! - [`Boolean<W, C>`] - Boolean values with XOR, AND, NOT operations
//! - [`UInt32<W, C>`] - 32-bit unsigned integers with SHA-256 operations
//! - [`sha256`] - SHA-256 hash function circuit

mod boolean;
mod sha256;
mod uint32;

pub use boolean::Boolean;
pub use sha256::{bits_to_bytes, bytes_to_bits, sha256_compression, sha256_padding, small_sha256};
pub use uint32::UInt32;
