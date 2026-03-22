use ff::{Field, PrimeField};
use halo2curves::CurveAffine;
use rayon::prelude::*;

/// Convert a 32-byte little-endian representation to 16 × 16-bit limbs (stored as u32).
fn bytes_to_16bit_limbs(bytes: &[u8; 32]) -> [u32; 16] {
    let mut limbs = [0u32; 16];
    for i in 0..16 {
        let lo = bytes[2 * i] as u32;
        let hi = bytes[2 * i + 1] as u32;
        limbs[i] = lo | (hi << 8);
    }
    limbs
}

/// Convert 16 × 16-bit limbs back to 32-byte little-endian representation.
pub fn limbs_to_bytes(limbs: &[u32; 16]) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    for i in 0..16 {
        bytes[2 * i] = (limbs[i] & 0xFF) as u8;
        bytes[2 * i + 1] = ((limbs[i] >> 8) & 0xFF) as u8;
    }
    bytes
}

/// Convert 16 × 16-bit limbs into [u64; 4] little-endian.
pub fn limbs_to_u64x4(limbs: &[u32]) -> [u64; 4] {
    let mut result = [0u64; 4];
    for i in 0..16 {
        let word_idx = i / 4; // which u64
        let shift = (i % 4) * 16;
        result[word_idx] |= (limbs[i] as u64) << shift;
    }
    result
}

/// Pack 16-bit limbs into halfword (2 × 16-bit per u32) format for GPU input.
/// Takes 16 limbs, produces 8 packed u32s.
fn pack_limbs_halfword(limbs: &[u32; 16]) -> [u32; 8] {
    let mut packed = [0u32; 8];
    for i in 0..8 {
        packed[i] = limbs[2 * i] | (limbs[2 * i + 1] << 16);
    }
    packed
}

/// Pack affine points and scalars into GPU-ready u32 buffers.
///
/// Points: each point contributes 8 (x) + 8 (y) = 16 packed u32s.
/// Scalars: each scalar contributes 8 packed u32s.
///
/// The GPU shader expects halfword-packed 16-bit limbs.
pub fn pack_affine_and_scalars<C: CurveAffine>(
    bases: &[C],
    scalars: &[C::Scalar],
    _num_limbs: usize,
) -> (Vec<u32>, Vec<u32>)
where
    C::Base: ff::PrimeField,
    C::Scalar: ff::PrimeField,
{
    let num_elements = bases.len();
    let packed_per_coord = 8; // 16 limbs / 2
    let coords_per_point = packed_per_coord * 2; // x + y

    let mut coords = vec![0u32; num_elements * coords_per_point];
    let mut scalars_u32 = vec![0u32; num_elements * packed_per_coord];

    const CHUNK_SIZE: usize = 1024;

    bases
        .par_chunks(CHUNK_SIZE)
        .zip(scalars.par_chunks(CHUNK_SIZE))
        .zip(coords.par_chunks_mut(CHUNK_SIZE * coords_per_point))
        .zip(scalars_u32.par_chunks_mut(CHUNK_SIZE * packed_per_coord))
        .for_each(
            |(((base_chunk, scalar_chunk), coord_chunk), scalar_u32_chunk)| {
                for (i, (pt, sc)) in base_chunk.iter().zip(scalar_chunk.iter()).enumerate() {
                    // Extract affine coordinates
                    let coords_opt = pt.coordinates();
                    let (x_repr, y_repr) = if coords_opt.is_some().into() {
                        let c = coords_opt.unwrap();
                        let x_repr = c.x().to_repr();
                        let y_repr = c.y().to_repr();
                        (x_repr, y_repr)
                    } else {
                        // Point at infinity — use zeros
                        (C::Base::ZERO.to_repr(), C::Base::ZERO.to_repr())
                    };

                    let x_bytes: &[u8; 32] = x_repr.as_ref().try_into().unwrap();
                    let y_bytes: &[u8; 32] = y_repr.as_ref().try_into().unwrap();

                    let x_limbs = bytes_to_16bit_limbs(x_bytes);
                    let y_limbs = bytes_to_16bit_limbs(y_bytes);

                    let x_packed = pack_limbs_halfword(&x_limbs);
                    let y_packed = pack_limbs_halfword(&y_limbs);

                    let coord_start = i * coords_per_point;
                    coord_chunk[coord_start..coord_start + packed_per_coord]
                        .copy_from_slice(&x_packed);
                    coord_chunk[coord_start + packed_per_coord..coord_start + coords_per_point]
                        .copy_from_slice(&y_packed);

                    // Pack scalar
                    let sc_repr = sc.to_repr();
                    let sc_bytes: &[u8; 32] = sc_repr.as_ref().try_into().unwrap();
                    let sc_limbs = bytes_to_16bit_limbs(sc_bytes);
                    let sc_packed = pack_limbs_halfword(&sc_limbs);

                    let scalar_start = i * packed_per_coord;
                    scalar_u32_chunk[scalar_start..scalar_start + packed_per_coord]
                        .copy_from_slice(&sc_packed);
                }
            },
        );

    (coords, scalars_u32)
}
