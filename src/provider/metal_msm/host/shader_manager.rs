use crate::provider::metal_msm::host::{
  gpu::get_default_device,
  metal_wrapper::{MSMConstants, MetalConfig, get_or_calc_constants},
};
use metal::*;
use once_cell::sync::Lazy;
use std::{collections::HashMap, sync::Mutex};

include!(concat!(env!("OUT_DIR"), "/built_shaders.rs"));

static COMPILED_LIBRARY: Lazy<Mutex<Option<Library>>> = Lazy::new(|| Mutex::new(None));

/// Compile shader source at runtime and cache the library
fn get_or_compile_library(device: &Device) -> Library {
  let mut cache = COMPILED_LIBRARY.lock().unwrap();
  if let Some(ref lib) = *cache {
    return lib.clone();
  }

  let options = CompileOptions::new();
  let library = device
    .new_library_with_source(MSM_SHADER_SOURCE, &options)
    .expect("Failed to compile Metal shader source");
  *cache = Some(library.clone());
  library
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ShaderType {
  ConvertPointAndDecompose,
  Transpose,
  SMVP,
  BPRStage1,
  BPRStage2,
}

impl ShaderType {
  pub fn kernel_name(&self) -> &str {
    match self {
      ShaderType::ConvertPointAndDecompose => "convert_point_coords_and_decompose_scalars",
      ShaderType::Transpose => "transpose",
      ShaderType::SMVP => "smvp",
      ShaderType::BPRStage1 => "bpr_stage_1",
      ShaderType::BPRStage2 => "bpr_stage_2",
    }
  }

  pub fn get_config(&self, num_limbs: usize, log_limb_size: u32) -> MetalConfig {
    MetalConfig {
      log_limb_size,
      num_limbs,
      shader_file: String::new(),
      kernel_name: self.kernel_name().to_string(),
    }
  }
}

#[derive(Clone, Debug)]
pub struct ShaderManagerConfig {
  pub num_limbs: usize,
  pub log_limb_size: u32,
}

impl Default for ShaderManagerConfig {
  fn default() -> Self {
    Self {
      num_limbs: 16,
      log_limb_size: 16,
    }
  }
}

#[derive(Clone)]
pub struct PrecompiledShader {
  pub pipeline_state: ComputePipelineState,
  pub config: MetalConfig,
  pub constants: MSMConstants,
}

#[derive(Clone)]
pub struct ShaderManager {
  device: Device,
  config: ShaderManagerConfig,
  shaders: HashMap<ShaderType, PrecompiledShader>,
  constants: MSMConstants,
}

impl ShaderManager {
  pub fn new(config: ShaderManagerConfig) -> Result<Self, Box<dyn std::error::Error>> {
    let device = get_default_device();
    let constants = get_or_calc_constants(config.num_limbs, config.log_limb_size);
    let mut manager = Self {
      device,
      config: config.clone(),
      shaders: HashMap::new(),
      constants,
    };

    let library = get_or_compile_library(&manager.device);

    for shader_type in [
      ShaderType::ConvertPointAndDecompose,
      ShaderType::Transpose,
      ShaderType::SMVP,
      ShaderType::BPRStage1,
      ShaderType::BPRStage2,
    ] {
      let conf = shader_type.get_config(manager.config.num_limbs, manager.config.log_limb_size);
      let kernel = library.get_function(&conf.kernel_name, None)?;
      let ps = manager
        .device
        .new_compute_pipeline_state_with_function(&kernel)?;
      manager.shaders.insert(
        shader_type,
        PrecompiledShader {
          pipeline_state: ps,
          config: conf,
          constants: manager.constants.clone(),
        },
      );
    }

    Ok(manager)
  }

  pub fn with_default_config() -> Result<Self, Box<dyn std::error::Error>> {
    Self::new(ShaderManagerConfig::default())
  }

  pub fn get_shader(&self, shader_type: &ShaderType) -> Option<&PrecompiledShader> {
    self.shaders.get(shader_type)
  }

  pub fn device(&self) -> &Device {
    &self.device
  }

  pub fn config(&self) -> &ShaderManagerConfig {
    &self.config
  }
}
