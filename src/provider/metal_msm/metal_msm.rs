use crate::provider::metal_msm::host::metal_wrapper::MetalHelper;
use crate::provider::metal_msm::host::shader_manager::{ShaderManager, ShaderManagerConfig, ShaderType};
use crate::provider::metal_msm::utils::limbs_conversion::{limbs_to_u64x4, pack_affine_and_scalars};
use crate::provider::metal_msm::utils::mont_reduction::raw_reduction_pallas;
use crate::provider::metal_msm::utils::window_size_optimizer::fetch_gpu_core_count_and_simd_width_from_device;

use crate::provider::pasta::pallas;

use ff::Field;
use group::Group;
use halo2curves::CurveAffine;
use rayon::prelude::*;
use std::error::Error;

/// Configuration for Metal MSM pipeline
#[derive(Clone, Debug)]
pub struct MetalMSMConfig {
    pub num_limbs: usize,
    pub log_limb_size: u32,
}

impl Default for MetalMSMConfig {
    fn default() -> Self {
        Self {
            num_limbs: 16,
            log_limb_size: 16,
        }
    }
}

impl From<MetalMSMConfig> for ShaderManagerConfig {
    fn from(config: MetalMSMConfig) -> Self {
        Self {
            num_limbs: config.num_limbs,
            log_limb_size: config.log_limb_size,
        }
    }
}

/// Main Metal MSM pipeline with pre-compiled shaders
pub struct MetalMSMPipeline {
    config: MetalMSMConfig,
    shader_manager: ShaderManager,
    _gpu_cores: usize,
    simd_width: usize,
}

impl MetalMSMPipeline {
    fn new(config: MetalMSMConfig) -> Result<Self, Box<dyn Error>> {
        let shader_config: ShaderManagerConfig = config.clone().into();
        let shader_manager = ShaderManager::new(shader_config)?;
        let (gpu_cores, simd_width) =
            fetch_gpu_core_count_and_simd_width_from_device(shader_manager.device());

        Ok(Self {
            config,
            shader_manager,
            _gpu_cores: gpu_cores,
            simd_width,
        })
    }

    fn with_default_config() -> Result<Self, Box<dyn Error>> {
        Self::new(MetalMSMConfig::default())
    }

    /// Execute the complete MSM pipeline on GPU
    fn execute_pipeline(
        &self,
        bases: &[pallas::Affine],
        scalars: &[pallas::Scalar],
        input_size: usize,
        window_size: usize,
        scale_factor: usize,
    ) -> Result<pallas::Point, Box<dyn Error>> {
        let num_columns = 1 << window_size;
        // Pallas scalar field is 255 bits
        let num_subtasks = (255f32 / window_size as f32).ceil() as usize;

        // Stage 0: Pack inputs
        let (coords, scals) = pack_affine_and_scalars(bases, scalars, self.config.num_limbs);

        // Stage 1: Convert Point & Scalar Decomposition
        let stage1 = ConvertPointAndScalarDecompose::new(&self.shader_manager);
        let c_workgroup_size = self.simd_width * scale_factor;
        // Total threads needed = input_size
        // thread_groups = ceil(input_size / c_workgroup_size)
        let total_groups = (input_size + c_workgroup_size - 1) / c_workgroup_size;
        let c_num_x_workgroups = total_groups.min(c_workgroup_size).max(1);
        let c_num_y_workgroups = ((total_groups + c_num_x_workgroups - 1) / c_num_x_workgroups).max(1);
        let c_num_z_workgroups = 1;

        let (point_x, point_y, scalar_chunks) = stage1.execute(
            &coords,
            &scals,
            input_size,
            window_size,
            num_columns,
            num_subtasks,
            c_num_x_workgroups,
            c_num_y_workgroups,
            c_num_z_workgroups,
            c_workgroup_size,
        )?;

        // Stage 2: Transpose
        let stage2 = Transpose::new(&self.shader_manager);
        let (csc_col_ptr, csc_val_idxs) = stage2.execute(
            &scalar_chunks,
            num_subtasks,
            input_size,
            num_columns,
            1, 1, 1,
            num_subtasks,
        )?;

        // Stage 3: SMVP
        let stage3 = SMVP::new(&self.shader_manager);
        let s_workgroup_size = self.simd_width * scale_factor;
        let (bucket_x, bucket_y, bucket_z) = stage3.execute(
            &csc_col_ptr,
            &csc_val_idxs,
            &point_x,
            &point_y,
            input_size,
            num_subtasks,
            num_columns,
            self.simd_width,
            s_workgroup_size,
        )?;

        // Stage 4: Parallel Bucket Reduction
        let stage4 = PBPR::new(&self.shader_manager);
        let b_workgroup_size = self.simd_width * scale_factor;
        let (g_points_x, g_points_y, g_points_z) = stage4.execute(
            &bucket_x,
            &bucket_y,
            &bucket_z,
            num_subtasks,
            num_columns,
            num_subtasks,
            num_subtasks,
            num_subtasks, 1, 1,
            num_subtasks, 1, 1,
            b_workgroup_size,
        )?;

        // Stage 5 (CPU): final Horner reduction
        let result = self.final_reduction(
            &g_points_x,
            &g_points_y,
            &g_points_z,
            num_subtasks,
            window_size,
            b_workgroup_size,
        )?;

        Ok(result)
    }

    /// Final reduction on CPU with Horner's method
    fn final_reduction(
        &self,
        g_points_x: &[u32],
        g_points_y: &[u32],
        g_points_z: &[u32],
        num_subtasks: usize,
        window_size: usize,
        pbpr_workgroup_size: usize,
    ) -> Result<pallas::Point, Box<dyn Error>> {
        let gpu_points: Vec<pallas::Point> = (0..num_subtasks)
            .into_par_iter()
            .map(|i| {
                let mut accumulated_point = pallas::Point::identity();

                for j in 0..pbpr_workgroup_size {
                    let flat_idx = i * pbpr_workgroup_size + j;
                    let limb_start_idx = flat_idx * self.config.num_limbs;
                    let limb_end_idx = (flat_idx + 1) * self.config.num_limbs;

                    let xr_limbs = &g_points_x[limb_start_idx..limb_end_idx];
                    let yr_limbs = &g_points_y[limb_start_idx..limb_end_idx];
                    let zr_limbs = &g_points_z[limb_start_idx..limb_end_idx];

                    // Convert limbs to u64x4
                    let xr_u64 = limbs_to_u64x4(xr_limbs);
                    let yr_u64 = limbs_to_u64x4(yr_limbs);
                    let zr_u64 = limbs_to_u64x4(zr_limbs);

                    // Montgomery reduction
                    let x_u64 = raw_reduction_pallas(xr_u64);
                    let y_u64 = raw_reduction_pallas(yr_u64);
                    let z_u64 = raw_reduction_pallas(zr_u64);

                    // Convert to Pallas field elements
                    let x = pallas::Base::from_raw(x_u64);
                    let y = pallas::Base::from_raw(y_u64);
                    let z = pallas::Base::from_raw(z_u64);

                    // Construct Jacobian point — check if it's the point at infinity
                    if z == pallas::Base::ZERO {
                        continue;
                    }

                    // Convert from Jacobian (X, Y, Z) to affine then to Point
                    // Affine: x = X/Z², y = Y/Z³
                    let z_inv = z.invert().unwrap();
                    let z_inv2 = z_inv * z_inv;
                    let z_inv3 = z_inv2 * z_inv;
                    let ax = x * z_inv2;
                    let ay = y * z_inv3;

                    // Construct affine point and convert to projective
                    let affine = pallas::Affine::from_xy(ax, ay);
                    if affine.is_some().into() {
                        accumulated_point += pallas::Point::from(affine.unwrap());
                    }
                }

                accumulated_point
            })
            .collect();

        // Horner's method
        let m = pallas::Scalar::from(1u64 << window_size);
        let mut result = gpu_points[gpu_points.len() - 1];

        if gpu_points.len() > 1 {
            for i in (0..gpu_points.len() - 1).rev() {
                result = result * m;
                result = result + gpu_points[i];
            }
        }

        Ok(result)
    }
}

// ==================== Pipeline Stages ====================

struct ConvertPointAndScalarDecompose<'a> {
    shader_manager: &'a ShaderManager,
}

impl<'a> ConvertPointAndScalarDecompose<'a> {
    fn new(shader_manager: &'a ShaderManager) -> Self {
        Self { shader_manager }
    }

    fn execute(
        &self,
        coords: &[u32],
        scalars: &[u32],
        input_size: usize,
        window_size: usize,
        num_columns: usize,
        num_subtasks: usize,
        c_num_x_workgroups: usize,
        c_num_y_workgroups: usize,
        c_num_z_workgroups: usize,
        c_workgroup_size: usize,
    ) -> Result<(Vec<u32>, Vec<u32>, Vec<u32>), Box<dyn Error>> {
        let mut helper = MetalHelper::with_device(self.shader_manager.device().clone());
        let shader = self
            .shader_manager
            .get_shader(&ShaderType::ConvertPointAndDecompose)
            .ok_or("ConvertPointAndDecompose shader not found")?;

        let coords_buf = helper.create_buffer(&coords.to_vec());
        let scalars_buf = helper.create_buffer(&scalars.to_vec());
        let out_point_x =
            helper.create_empty_buffer(input_size * self.shader_manager.config().num_limbs);
        let out_point_y =
            helper.create_empty_buffer(input_size * self.shader_manager.config().num_limbs);
        let out_scalar_chunks = helper.create_empty_buffer(input_size * num_subtasks);

        let params_buf = helper.create_buffer(&vec![
            input_size as u32,
            window_size as u32,
            num_columns as u32,
            num_subtasks as u32,
        ]);

        let thread_group_count = helper.create_thread_group_size(
            c_num_x_workgroups as u64,
            c_num_y_workgroups as u64,
            c_num_z_workgroups as u64,
        );
        let threads_per_threadgroup =
            helper.create_thread_group_size(c_workgroup_size as u64, 1, 1);

        helper.execute_shader_with_pipeline(
            &shader.pipeline_state,
            &[
                &coords_buf,
                &scalars_buf,
                &out_point_x,
                &out_point_y,
                &out_scalar_chunks,
                &params_buf,
            ],
            &thread_group_count,
            &threads_per_threadgroup,
        );

        let point_x = helper.read_results(
            &out_point_x,
            input_size * self.shader_manager.config().num_limbs,
        );
        let point_y = helper.read_results(
            &out_point_y,
            input_size * self.shader_manager.config().num_limbs,
        );
        let scalar_chunks = helper.read_results(&out_scalar_chunks, input_size * num_subtasks);

        helper.drop_all_buffers();
        Ok((point_x, point_y, scalar_chunks))
    }
}

struct Transpose<'a> {
    shader_manager: &'a ShaderManager,
}

impl<'a> Transpose<'a> {
    fn new(shader_manager: &'a ShaderManager) -> Self {
        Self { shader_manager }
    }

    fn execute(
        &self,
        scalar_chunks: &[u32],
        num_subtasks: usize,
        input_size: usize,
        num_columns: usize,
        t_num_x_workgroups: usize,
        t_num_y_workgroups: usize,
        t_num_z_workgroups: usize,
        t_workgroup_size: usize,
    ) -> Result<(Vec<u32>, Vec<u32>), Box<dyn Error>> {
        let mut helper = MetalHelper::with_device(self.shader_manager.device().clone());
        let shader = self
            .shader_manager
            .get_shader(&ShaderType::Transpose)
            .ok_or("Transpose shader not found")?;

        let in_chunks_buf = helper.create_buffer(&scalar_chunks.to_vec());
        let out_csc_col_ptr =
            helper.create_empty_buffer(num_subtasks * ((num_columns + 1) as usize) * 4);
        let out_csc_val_idxs = helper.create_empty_buffer(scalar_chunks.len());
        let out_curr = helper.create_empty_buffer(num_subtasks * (num_columns as usize) * 4);

        let params_buf = helper.create_buffer(&vec![num_columns as u32, input_size as u32]);

        let thread_group_count = helper.create_thread_group_size(
            t_num_x_workgroups as u64,
            t_num_y_workgroups as u64,
            t_num_z_workgroups as u64,
        );
        let threads_per_threadgroup =
            helper.create_thread_group_size(t_workgroup_size as u64, 1, 1);

        helper.execute_shader_with_pipeline(
            &shader.pipeline_state,
            &[
                &in_chunks_buf,
                &out_csc_col_ptr,
                &out_csc_val_idxs,
                &out_curr,
                &params_buf,
            ],
            &thread_group_count,
            &threads_per_threadgroup,
        );

        let csc_col_ptr = helper.read_results(
            &out_csc_col_ptr,
            num_subtasks * ((num_columns + 1) as usize) * 4,
        );
        let csc_val_idxs = helper.read_results(&out_csc_val_idxs, scalar_chunks.len());

        helper.drop_all_buffers();
        Ok((csc_col_ptr, csc_val_idxs))
    }
}

struct SMVP<'a> {
    shader_manager: &'a ShaderManager,
}

impl<'a> SMVP<'a> {
    fn new(shader_manager: &'a ShaderManager) -> Self {
        Self { shader_manager }
    }

    fn execute(
        &self,
        csc_col_ptr: &[u32],
        csc_val_idxs: &[u32],
        point_x: &[u32],
        point_y: &[u32],
        input_size: usize,
        num_subtasks: usize,
        num_columns: usize,
        simd_width: usize,
        s_workgroup_size: usize,
    ) -> Result<(Vec<u32>, Vec<u32>, Vec<u32>), Box<dyn Error>> {
        let mut helper = MetalHelper::with_device(self.shader_manager.device().clone());
        let shader = self
            .shader_manager
            .get_shader(&ShaderType::SMVP)
            .ok_or("SMVP shader not found")?;

        let half_columns = num_columns / 2;
        let bucket_size =
            half_columns * self.shader_manager.config().num_limbs * 4 * num_subtasks;

        let row_ptr_buf = helper.create_buffer(&csc_col_ptr.to_vec());
        let val_idx_buf = helper.create_buffer(&csc_val_idxs.to_vec());
        let point_x_buf = helper.create_buffer(&point_x.to_vec());
        let point_y_buf = helper.create_buffer(&point_y.to_vec());

        let bucket_x_buf = helper.create_empty_buffer(bucket_size);
        let bucket_y_buf = helper.create_empty_buffer(bucket_size);
        let bucket_z_buf = helper.create_empty_buffer(bucket_size);

        let num_subtask_chunk_size = 4u32;
        for offset in (0..num_subtasks as u32).step_by(num_subtask_chunk_size as usize) {
            let remaining_subtasks = (num_subtasks as u32 - offset).min(num_subtask_chunk_size);
            let valid_threads = half_columns as u64 * remaining_subtasks as u64;

            let max_y = ((valid_threads as usize)
                / (s_workgroup_size * remaining_subtasks as usize))
                .max(1);
            let s_num_y_workgroups = simd_width.min(max_y) as u64;
            let s_num_z_workgroups = remaining_subtasks as u64;

            let threads_per_grid =
                (s_workgroup_size as u64) * s_num_y_workgroups * s_num_z_workgroups;
            let s_num_x_workgroups = (valid_threads + threads_per_grid - 1) / threads_per_grid;

            let params_buf = helper.create_buffer(&vec![
                input_size as u32,
                num_columns as u32,
                num_subtasks as u32,
                offset,
            ]);

            let thread_group_count = helper.create_thread_group_size(
                s_num_x_workgroups,
                s_num_y_workgroups,
                s_num_z_workgroups,
            );
            let threads_per_threadgroup =
                helper.create_thread_group_size(s_workgroup_size as u64, 1, 1);

            helper.execute_shader_with_pipeline(
                &shader.pipeline_state,
                &[
                    &row_ptr_buf,
                    &val_idx_buf,
                    &point_x_buf,
                    &point_y_buf,
                    &bucket_x_buf,
                    &bucket_y_buf,
                    &bucket_z_buf,
                    &params_buf,
                ],
                &thread_group_count,
                &threads_per_threadgroup,
            );
        }

        let bucket_x = helper.read_results(&bucket_x_buf, bucket_size);
        let bucket_y = helper.read_results(&bucket_y_buf, bucket_size);
        let bucket_z = helper.read_results(&bucket_z_buf, bucket_size);

        helper.drop_all_buffers();
        Ok((bucket_x, bucket_y, bucket_z))
    }
}

struct PBPR<'a> {
    shader_manager: &'a ShaderManager,
}

impl<'a> PBPR<'a> {
    fn new(shader_manager: &'a ShaderManager) -> Self {
        Self { shader_manager }
    }

    #[allow(clippy::too_many_arguments)]
    fn execute(
        &self,
        bucket_x: &[u32],
        bucket_y: &[u32],
        bucket_z: &[u32],
        num_subtasks: usize,
        num_columns: usize,
        num_subtasks_per_bpr_1: usize,
        num_subtasks_per_bpr_2: usize,
        b_num_x_workgroups: usize,
        b_num_y_workgroups: usize,
        b_num_z_workgroups: usize,
        b_2_num_x_workgroups: usize,
        b_2_num_y_workgroups: usize,
        b_2_num_z_workgroups: usize,
        b_workgroup_size: usize,
    ) -> Result<(Vec<u32>, Vec<u32>, Vec<u32>), Box<dyn Error>> {
        let mut helper = MetalHelper::with_device(self.shader_manager.device().clone());
        let stage1_shader = self
            .shader_manager
            .get_shader(&ShaderType::BPRStage1)
            .ok_or("BPRStage1 shader not found")?;
        let stage2_shader = self
            .shader_manager
            .get_shader(&ShaderType::BPRStage2)
            .ok_or("BPRStage2 shader not found")?;

        let bucket_sum_x_buf = helper.create_buffer(&bucket_x.to_vec());
        let bucket_sum_y_buf = helper.create_buffer(&bucket_y.to_vec());
        let bucket_sum_z_buf = helper.create_buffer(&bucket_z.to_vec());

        let g_points_size =
            num_subtasks * b_workgroup_size * self.shader_manager.config().num_limbs * 4;
        let g_points_x_buf = helper.create_empty_buffer(g_points_size);
        let g_points_y_buf = helper.create_empty_buffer(g_points_size);
        let g_points_z_buf = helper.create_empty_buffer(g_points_size);

        // Stage 1
        for subtask_chunk_idx in (0..num_subtasks).step_by(num_subtasks_per_bpr_1) {
            let params = vec![
                subtask_chunk_idx as u32,
                num_columns as u32,
                num_subtasks_per_bpr_1 as u32,
                0u32,
            ];
            let params_buf = helper.create_buffer(&params);

            let thread_group_count = helper.create_thread_group_size(
                b_num_x_workgroups as u64,
                b_num_y_workgroups as u64,
                b_num_z_workgroups as u64,
            );
            let threads_per_threadgroup =
                helper.create_thread_group_size(b_workgroup_size as u64, 1, 1);

            helper.execute_shader_with_pipeline(
                &stage1_shader.pipeline_state,
                &[
                    &bucket_sum_x_buf,
                    &bucket_sum_y_buf,
                    &bucket_sum_z_buf,
                    &g_points_x_buf,
                    &g_points_y_buf,
                    &g_points_z_buf,
                    &params_buf,
                ],
                &thread_group_count,
                &threads_per_threadgroup,
            );
        }

        // Stage 2
        for subtask_chunk_idx in (0..num_subtasks).step_by(num_subtasks_per_bpr_2) {
            let params = vec![
                subtask_chunk_idx as u32,
                num_columns as u32,
                num_subtasks_per_bpr_2 as u32,
                0u32,
            ];
            let params_buf = helper.create_buffer(&params);

            let thread_group_count = helper.create_thread_group_size(
                b_2_num_x_workgroups as u64,
                b_2_num_y_workgroups as u64,
                b_2_num_z_workgroups as u64,
            );
            let threads_per_threadgroup =
                helper.create_thread_group_size(b_workgroup_size as u64, 1, 1);

            helper.execute_shader_with_pipeline(
                &stage2_shader.pipeline_state,
                &[
                    &bucket_sum_x_buf,
                    &bucket_sum_y_buf,
                    &bucket_sum_z_buf,
                    &g_points_x_buf,
                    &g_points_y_buf,
                    &g_points_z_buf,
                    &params_buf,
                ],
                &thread_group_count,
                &threads_per_threadgroup,
            );
        }

        let g_points_x = helper.read_results(&g_points_x_buf, g_points_size);
        let g_points_y = helper.read_results(&g_points_y_buf, g_points_size);
        let g_points_z = helper.read_results(&g_points_z_buf, g_points_size);

        helper.drop_all_buffers();
        Ok((g_points_x, g_points_y, g_points_z))
    }
}

/// Batch MSM: compute multiple MSMs sharing the same bases.
/// Each element of `all_scalars` is a set of scalars for one MSM.
/// Returns one result per scalar set.
pub fn batch_metal_msm_pallas(
    bases: &[pallas::Affine],
    all_scalars: &[&[pallas::Scalar]],
) -> Result<Vec<pallas::Point>, Box<dyn Error>> {
    if bases.is_empty() || all_scalars.is_empty() {
        return Ok(vec![]);
    }

    let pipeline = MetalMSMPipeline::with_default_config()?;
    let mut results = Vec::with_capacity(all_scalars.len());

    for scalars in all_scalars {
        let input_size = bases.len().min(scalars.len());
        let window_size = if input_size < 16384 { 8 } else { 13 };
        let scale_factor = if input_size <= 4096 { 1 } else { 2 };

        let result = pipeline.execute_pipeline(
            &bases[..input_size],
            &scalars[..input_size],
            input_size,
            window_size,
            scale_factor,
        )?;
        results.push(result);
    }

    Ok(results)
}

/// Public API: compute MSM on Pallas curve using Metal GPU.
pub fn metal_msm_pallas(
    bases: &[pallas::Affine],
    scalars: &[pallas::Scalar],
) -> Result<pallas::Point, Box<dyn Error>> {
    if bases.is_empty() || scalars.is_empty() {
        return Err("Empty input".into());
    }

    let input_size = bases.len().min(scalars.len());
    let bases = &bases[..input_size];
    let scalars = &scalars[..input_size];

    // Window size tuned for different input sizes
    let window_size = if input_size < 16384 {
        8
    } else if input_size < 524288 {
        13
    } else if input_size <= 16777216 {
        15
    } else {
        16
    };

    let scale_factor = if input_size <= 4096 {
        1
    } else if input_size <= 65536 {
        2
    } else if input_size <= 1048576 {
        4
    } else if input_size <= 16777216 {
        8
    } else {
        16
    };

    let pipeline = MetalMSMPipeline::with_default_config()?;
    pipeline.execute_pipeline(bases, scalars, input_size, window_size, scale_factor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::msm;
    use ff::Field;
    use group::Curve;
    use rand_core::OsRng;
    use std::time::Instant;

    #[test]
    fn test_batch_metal_msm_pallas() {
        use halo2curves::group::Curve;
        let n = 8192; // Hyrax commitment width
        let num_rows = 19; // typical non-zero rows
        let mut rng = OsRng;

        let bases: Vec<pallas::Affine> = (0..n)
            .map(|_| pallas::Point::random(&mut rng).to_affine())
            .collect();

        let all_scalars: Vec<Vec<pallas::Scalar>> = (0..num_rows)
            .map(|_| (0..n).map(|_| pallas::Scalar::random(&mut rng)).collect())
            .collect();

        let scalar_refs: Vec<&[pallas::Scalar]> = all_scalars.iter().map(|s| s.as_slice()).collect();

        // GPU batch
        println!("Running {} batched GPU MSMs of {} points...", num_rows, n);
        let start = Instant::now();
        let gpu_results = batch_metal_msm_pallas(&bases, &scalar_refs).unwrap();
        println!("GPU batch MSM: {:?} ({:?}/msm)", start.elapsed(), start.elapsed() / num_rows as u32);

        // CPU parallel
        println!("Running {} CPU MSMs of {} points (parallel)...", num_rows, n);
        let start = Instant::now();
        let cpu_results: Vec<_> = all_scalars
            .par_iter()
            .map(|s| msm::msm(s, &bases).unwrap())
            .collect();
        println!("CPU parallel MSM: {:?}", start.elapsed());

        // Verify
        for (i, (gpu, cpu)) in gpu_results.iter().zip(cpu_results.iter()).enumerate() {
            assert_eq!(gpu.to_affine(), cpu.to_affine(), "Mismatch at row {}", i);
        }
        println!("All {} MSMs verified correct!", num_rows);
    }

    #[test]
    fn test_metal_msm_large_scale() {
        use halo2curves::group::Curve;
        let mut rng = OsRng;

        for log_n in [14, 16, 17] {
            let n = 1usize << log_n;
            let bases: Vec<pallas::Affine> = (0..n)
                .map(|_| pallas::Point::random(&mut rng).to_affine())
                .collect();
            let scalars: Vec<pallas::Scalar> = (0..n)
                .map(|_| pallas::Scalar::random(&mut rng))
                .collect();

            let start = Instant::now();
            let gpu_result = metal_msm_pallas(&bases, &scalars).unwrap();
            let gpu_time = start.elapsed();

            let start = Instant::now();
            let cpu_result = msm::msm(&scalars, &bases).unwrap();
            let cpu_time = start.elapsed();

            assert_eq!(gpu_result.to_affine(), cpu_result.to_affine(), "Mismatch at N=2^{}", log_n);
            println!("N=2^{} ({}): GPU={:?}, CPU={:?}, ratio={:.2}x",
                log_n, n, gpu_time, cpu_time, gpu_time.as_secs_f64() / cpu_time.as_secs_f64());
        }
    }

    #[test]
    fn test_metal_msm_pallas_debug_raw() {
        // Debug: check raw GPU output
        use halo2curves::group::Curve;
        let generator = pallas::Point::generator().to_affine();
        let n = 256;
        let bases: Vec<pallas::Affine> = vec![generator; n];
        let scalars: Vec<pallas::Scalar> = vec![pallas::Scalar::ONE; n];

        // Run just the first few stages and inspect
        let config = super::MetalMSMConfig::default();
        let (coords, scals) = super::super::utils::limbs_conversion::pack_affine_and_scalars(&bases, &scalars, config.num_limbs);
        println!("coords len: {}, first 16: {:?}", coords.len(), &coords[..16]);
        println!("scals len: {}, first 8: {:?}", scals.len(), &scals[..8]);

        // Check that the packing is non-zero
        assert!(coords.iter().any(|&x| x != 0), "coords should not be all zero");
        assert!(scals.iter().any(|&x| x != 0), "scalars should not be all zero");
        println!("Packing verified non-zero");
    }

    #[test]
    fn test_metal_msm_pallas_small() {
        // Test with a small known input to debug
        use halo2curves::group::Curve;
        let generator = pallas::Point::generator().to_affine();
        // 256 points (minimum for GPU pipeline to not crash on thread counts)
        let n = 256;
        let bases: Vec<pallas::Affine> = vec![generator; n];
        let scalars: Vec<pallas::Scalar> = (0..n)
            .map(|i| pallas::Scalar::from(i as u64 + 1))
            .collect();

        println!("Running small GPU MSM with {} points...", n);
        let gpu_result = metal_msm_pallas(&bases, &scalars);
        match gpu_result {
            Ok(r) => {
                let r_affine = r.to_affine();
                println!("GPU result: {:?}", r_affine);
                // Expected: sum of i*G for i=1..n = n*(n+1)/2 * G
                let expected_scalar = pallas::Scalar::from((n as u64 * (n as u64 + 1)) / 2);
                let expected = generator * expected_scalar;
                println!("Expected: {:?}", expected);
                assert_eq!(r_affine, expected.to_affine(), "Small MSM mismatch");
            }
            Err(e) => {
                println!("GPU MSM error: {}", e);
                panic!("GPU MSM failed");
            }
        }
    }

    #[test]
    fn test_metal_msm_pallas_correctness() {
        let n = 1 << 13; // 8192 — matches Hyrax commitment width
        let mut rng = OsRng;

        let scalars: Vec<pallas::Scalar> = (0..n).map(|_| pallas::Scalar::random(&mut rng)).collect();
        let bases: Vec<pallas::Affine> = (0..n)
            .map(|_| (pallas::Point::random(&mut rng)).to_affine())
            .collect();

        println!("Running GPU MSM with {} points (cold)...", n);
        let start = Instant::now();
        let gpu_result = metal_msm_pallas(&bases, &scalars).unwrap();
        println!("GPU MSM (cold): {:?}", start.elapsed());

        // Warm run
        let start = Instant::now();
        let _gpu_result2 = metal_msm_pallas(&bases, &scalars).unwrap();
        println!("GPU MSM (warm): {:?}", start.elapsed());

        println!("Running CPU MSM with {} points...", n);
        let start = Instant::now();
        let cpu_result = msm::msm(&scalars, &bases).unwrap();
        println!("CPU MSM: {:?}", start.elapsed());

        assert_eq!(
            gpu_result.to_affine(),
            cpu_result.to_affine(),
            "GPU and CPU MSM results must match"
        );
        println!("Correctness verified!");
    }
}
