#include "extract_word_from_bytes_le.metal"
#include <metal_stdlib>
using namespace metal;

kernel void decompose_scalars_batched(
    device const uint* scalars [[buffer(0), access(read)]],
    device uint* chunks [[buffer(1), access(write)]],
    constant uint4& params [[buffer(2), access(read)]],
    constant uint2& batch_params [[buffer(3), access(read)]],
    uint gid [[thread_position_in_grid]])
{
    const uint input_size = params[0];
    const uint window_size = params[1];
    const uint num_columns = params[2];
    const uint num_subtask = params[3];
    const uint batch_size = batch_params[0];
    const uint total_scalars = batch_params[1];

    const uint id = gid;
    if (id >= total_scalars) {
        return;
    }

    const uint batch_idx = id / input_size;
    if (batch_idx >= batch_size) {
        return;
    }
    const uint point_idx = id - (batch_idx * input_size);

    uint scalar_bytes[16];

#pragma unroll(8)
    for (uint i = 0u; i < 8u; i++) {
        uint s = scalars[id * 8u + i];
        uint hi = s >> 16u;
        uint lo = s & 0xFFFFu;
        scalar_bytes[15u - (i * 2u)] = lo;
        scalar_bytes[15u - (i * 2u) - 1u] = hi;
    }

    uint l = num_columns;
    uint s = l / 2u;
    uint carry = 0u;

    for (uint i = 0u; i < num_subtask; i++) {
        uint chunk_val;
        if (i < num_subtask - 1u) {
            chunk_val = extract_word_from_bytes_le(scalar_bytes, i, window_size);
        } else {
            chunk_val =
                scalar_bytes[0] >> (((num_subtask * window_size - 256u) + 16u) - window_size);
        }

        int slice_val = int(chunk_val + carry);
        if (slice_val >= int(s)) {
            slice_val = (int(l) - slice_val) * (-1);
            carry = 1u;
        } else {
            carry = 0u;
        }

        uint offset = (batch_idx * num_subtask + i) * input_size;
        chunks[point_idx + offset] = uint(slice_val) + s;
    }
}
