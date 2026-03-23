# Program: Spartan ZK Prove < 100ms via GPU-Accelerated MSM

## Goal

Get **Spartan ZK total prove time under 100ms** by GPU-accelerating the MSM commitment phase on Apple M4 Metal GPU.

Baseline: 183ms. Target: <100ms. The commit_rest MSM at 97ms (53% of prove) is the primary target.

## Approach

Use [zkmopro/gpu-acceleration](https://github.com/zkmopro/gpu-acceleration) as a foundation. It provides a complete Metal MSM implementation (MIT/Apache-2.0 licensed) for BN254 using Metal Shading Language. The field arithmetic and MSM kernels are **generic** — parameterized by constants in a header file. Adding Pallas support requires only changing the field constants (modulus, Barrett µ, Montgomery N0, generator coordinates). No kernel rewrites needed.

## Metric

The single number to optimize is `TOTAL PROVE (prep+prove)` from:
```bash
cargo run --release --example rs_shuffle_bp_full -- --zk
```

Run 3 times, take the **median**.

## Architecture

### GPU MSM Module

Vendor the Metal MSM from `zkmopro/gpu-acceleration/mopro-msm/src/msm/metal_msm/` into `src/provider/metal_msm/`.

Key components:
- `shader/constants.metal` — field parameters (CHANGE: BN254 → Pallas)
- `shader/bigint/` — 256-bit integer arithmetic in MSL (NO CHANGE)
- `shader/field/` — Fp add, sub, reduce in MSL (NO CHANGE)
- `shader/mont_backend/` — Montgomery multiplication in MSL (NO CHANGE)
- `shader/curve/` — Jacobian point add, double, mixed add, scalar mul (NO CHANGE)
- `shader/cuzk/` — Pippenger MSM kernel via sparse matrix-vector product (NO CHANGE)
- `host/` — Rust Metal wrapper (ADAPT: arkworks types → halo2curves types)

### Integration Point

The Hyrax commit in `src/provider/pcs/hyrax_pc.rs` currently does:
```
(0..num_rows).into_par_iter().map(|i| {
    E::GE::vartime_multiscalar_mul(row_scalars, generators)
}).collect()
```

For GPU, batch ALL row MSMs into a **single GPU dispatch**:
```
gpu_msm::batch_msm(all_row_scalars, generators)  // one Metal command buffer
```

This amortizes the ~0.3ms Metal dispatch overhead across all 19 MSMs instead of paying it 19 times.

### Feature Flag

All GPU code behind `#[cfg(feature = "gpu")]`. CPU path unchanged without the flag.

```toml
[features]
gpu = ["metal", "objc"]
```

## Files You CAN Modify

| File | Role |
|------|------|
| `src/provider/metal_msm/` | NEW: vendored Metal MSM module |
| `src/provider/msm.rs` | Add `msm_gpu()` behind feature flag |
| `src/provider/pcs/hyrax_pc.rs` | Batch GPU MSM in `commit()` |
| `src/provider/traits.rs` | Override `batch_vartime_multiscalar_mul` for GPU |
| `src/provider/pasta.rs` | Conditional GPU `DlogGroupExt` impl |
| `src/provider/mod.rs` | Export metal_msm module |
| `Cargo.toml` | Add metal/objc deps, gpu feature |

## Files You MUST NOT Modify

- `examples/rs_shuffle_bp_full.rs` — benchmark harness
- `src/rs_shuffle_bp/` — shuffle circuit definition
- `src/timing/` — instrumentation
- `autoresearch/` — previous CPU optimization work
- Anything outside this repo

## Key Technical Details

### Pallas Constants for Metal

The `constants.metal` header needs these Pallas-specific values (16 × 16-bit limbs):
- **Base field modulus p** = `0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001`
- **Scalar field q** = `0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001`
- **Curve**: y² = x³ + 5 (a = 0, b = 5)
- **Barrett µ**: `⌊2^512 / p⌋`
- **Montgomery N0**: `-p⁻¹ mod 2^16`
- **Generator point**: Pallas G1 generator in Jacobian coordinates

### Type Conversion (halo2curves ↔ GPU)

The GPU works with 16 × 16-bit limbs. Conversion from halo2curves:
- `pallas::Scalar` → extract 32 bytes LE → split into 16 × u16 limbs
- `pallas::Affine` → extract (x, y) as 32 bytes each → split into limbs
- GPU result (Jacobian x, y, z limbs) → reconstruct `pallas::Point`

### Unified Memory Advantage

Apple M4 uses unified memory — CPU and GPU share the same physical memory. No PCIe transfer needed. The conversion overhead is purely computational (limb splitting/joining), not memory transfer.

## Experiment Protocol

```
1. Clone and vendor zkmopro/gpu-acceleration Metal MSM
2. Generate Pallas constants.metal
3. Adapt Rust host code for halo2curves types
4. Build and test: GPU MSM result matches CPU MSM result
5. Standalone GPU MSM benchmark: measure latency at N=8192 and N=155K
6. If GPU is faster: integrate into Hyrax commit with batch dispatch
7. Benchmark full prove pipeline
8. Tune: window size, batch size, GPU thread configuration
```

## results.tsv Format

```
commit	prove_ms	commit_rest_ms	outer_sc_ms	inner_sc_ms	pcs_ms	nifs_ms	verify_ms	status	description
```

Tab-separated.

## Security Constraints — NON-NEGOTIABLE

- GPU MSM must produce **identical results** to CPU MSM
- All scalar field operations remain on full 256-bit elements
- No truncation of scalars or curve points
- Zero-knowledge blinding unchanged
- Fiat-Shamir transcript unchanged
- The GPU is a **computation accelerator only** — it does not change the protocol
