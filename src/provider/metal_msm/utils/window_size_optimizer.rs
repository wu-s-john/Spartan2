use metal::*;

include!(concat!(env!("OUT_DIR"), "/built_shaders.rs"));

/// Compile shader source and fetch GPU info
pub fn fetch_gpu_core_count_and_simd_width_from_device(device: &Device) -> (usize, usize) {
  let options = CompileOptions::new();
  let library = device
    .new_library_with_source(MSM_SHADER_SOURCE, &options)
    .expect("Failed to compile Metal shader source for GPU info");
  let kernel = library.get_function("smvp", None).unwrap();
  let pipeline = device
    .new_compute_pipeline_state_with_function(&kernel)
    .expect("Failed to create pipeline");

  let simd_width = pipeline.thread_execution_width() as usize;
  let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
  let estimated_cores = max_threads / simd_width;

  (estimated_cores, simd_width)
}
