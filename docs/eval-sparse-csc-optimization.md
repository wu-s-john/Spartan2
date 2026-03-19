# eval_sparse CSC Optimization Report

## Summary

Optimized `compute_eval_table_sparse` (`bind_row_vars_combined_small`) by converting
the matrix scatter operation from CSR (row-major) to CSC (column-major) format.

**Results at 1024B SHA-256:**
- Before: 29-30ms (median)
- After: 20-22ms (median), best runs 19ms
- **27-37% improvement**

**Results at 128B SHA-256:**
- Before: 6ms
- After: 1-2ms
- **3-6x improvement**

## Root Cause Analysis

The original CSR-based scatter loop had a fundamental cache problem:

| Metric | Actual (1024B) | Plan's Estimate |
|--------|----------------|-----------------|
| Dense columns | 559,179 | ~200K |
| Rows (unpadded) | 968,724 | ~170K |
| Total NNZ | 4.15M | ~4.4M |
| rx array | 32MB | ~5.4MB |
| Buffer per thread | 18MB | ~6.4MB |

Both the rx array (32MB) and per-thread buffer (18MB) exceed L2 cache (~16MB on
Apple M4). The scatter loop does random writes to the buffer, causing L3/memory
misses at ~15-30ns each. With 4.15M entries and 10 threads, cache misses alone
account for ~12ms.

Additionally, each thread allocated its own 18MB buffer (180MB total across 10
threads), and a thread-buffer reduction step summed 9 buffers into one (~3ms).

## Solution: CSC with Cache-Blocked Column Partitioning

### Key Changes

1. **CSC (column-sorted) representation** built at setup time in `new_int()`.
   For each matrix (A, B, C), entries are stored sorted by dense column with
   col_ptr/row_indices/values arrays (like standard CSC format). Within each
   column, entries are partitioned: +/-1 entries first, then non-unit entries.

2. **Column-major iteration** in `bind_row_vars_combined_small`. Instead of
   iterating rows and scattering to random buffer positions, iterate columns
   and accumulate from random rx positions. Buffer writes become sequential
   (L1 hits), while rx reads hit L3 (~15ns).

3. **Inline A/B/C combination**. For each column, accumulate A, B, C sums into
   register-local variables, then combine: `result[c] = A + r*B + r^2*C`.
   This eliminates separate buffer allocations and a combine pass.

4. **Cache blocking**. Columns are partitioned into blocks of ~7000 columns.
   Each block's rx working set (~50K unique rows = ~1.6MB) fits comfortably
   in L2, improving cache hit rates.

5. **No thread-buffer reduction**. Each Rayon task writes to a disjoint slice
   of the output buffer via `par_chunks_mut`. This eliminates the 9-thread
   buffer reduction step (previously ~3ms).

### Why the Original Plan's Delayed Reduction Failed

The plan proposed replacing the `Vec<E::Scalar>` buffer (32 bytes/column) with
`Vec<SignedWideLimbs<5>>` (80 bytes/column) to avoid per-entry modular reduction.
This was a net negative because:

- The buffer grew 2.5x (from 18MB to 45MB per thread), far exceeding L2 cache
- The arithmetic savings (~20 instructions per entry) were dwarfed by the
  additional L3/memory misses caused by the larger buffer
- The problem is **memory-bound, not compute-bound**

A single `WideLimbs<5>` buffer (40 bytes/column, 25% larger) was also tried
but showed no improvement for the same reason.

### Memory Overhead

The CSC representation adds ~100MB of setup-time storage (3 matrices x
(col_ptr + row_indices + values + unit_ends)). This is a one-time cost
during `new_int()` and does not affect the hot path.

## Experiments Tried

| Experiment | Result | Reason |
|-----------|--------|--------|
| SignedWideLimbs<5> buffer (80B/col) | Regression: 37-42ms | 2.5x buffer busts L2 cache |
| WideLimbs<5> buffer (40B/col) | Regression: 34-42ms | 25% larger still hurts random access |
| CSC with precomputed rx_r/rx_r_sq | No improvement: 27-38ms | 3x32MB rx data exceeds L3 |
| CSC with separate A/B/C buffers | Modest: 25ms median | Combine pass adds ~4ms |
| CSC inline combine + cache blocking | **Best: 22ms median** | Minimal overhead, good cache |
| Software prefetch | Blocked | `#![forbid(unsafe_code)]` |

## Files Changed

- `src/r1cs/mod.rs`: Added CSC fields to `SplitR1CSShape`, CSC build in
  `new_int()`, CSC-based `bind_row_vars_combined_small`
- `src/small_constraint_system/mod.rs`: No changes (reverted dead code)
