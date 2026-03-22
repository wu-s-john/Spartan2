use crate::provider::metal_msm::host::gpu::{
    create_buffer, create_empty_buffer, get_default_device, read_buffer,
};
use crate::provider::metal_msm::utils::barrett_params::calc_barrett_mu;
use crate::provider::metal_msm::utils::mont_params::{calc_mont_radix, calc_nsafe, calc_rinv_and_n0};

use metal::*;
use num_bigint::BigUint;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::str::FromStr;
use std::sync::Mutex;

static CONSTANTS_CACHE: Lazy<Mutex<HashMap<(usize, u32), MSMConstants>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

#[derive(Clone)]
pub struct MetalConfig {
    pub log_limb_size: u32,
    pub num_limbs: usize,
    pub shader_file: String,
    pub kernel_name: String,
}

#[derive(Clone)]
pub struct MSMConstants {
    pub p: BigUint,
    pub r: BigUint,
    pub rinv: BigUint,
    pub n0: u32,
    pub nsafe: usize,
    pub mu: BigUint,
}

impl Default for MetalConfig {
    fn default() -> Self {
        Self {
            log_limb_size: 16,
            num_limbs: 16,
            shader_file: String::new(),
            kernel_name: String::new(),
        }
    }
}

pub struct MetalHelper {
    pub device: Device,
    pub command_queue: CommandQueue,
    pub buffers: Vec<Buffer>,
}

impl MetalHelper {
    pub fn new() -> Self {
        let device = get_default_device();
        let command_queue = device.new_command_queue();
        Self {
            device,
            command_queue,
            buffers: Vec::new(),
        }
    }

    pub fn with_device(device: Device) -> Self {
        let command_queue = device.new_command_queue();
        Self {
            device,
            command_queue,
            buffers: Vec::new(),
        }
    }

    pub fn create_buffer(&mut self, data: &[u32]) -> Buffer {
        let buffer = create_buffer(&self.device, data);
        self.buffers.push(buffer.clone());
        buffer
    }

    pub fn create_empty_buffer(&mut self, size: usize) -> Buffer {
        let buffer = create_empty_buffer(&self.device, size);
        self.buffers.push(buffer.clone());
        buffer
    }

    pub fn create_thread_group_size(&self, width: u64, height: u64, depth: u64) -> MTLSize {
        MTLSize {
            width,
            height,
            depth,
        }
    }

    pub fn execute_shader_with_pipeline(
        &self,
        pipeline_state: &ComputePipelineState,
        buffers: &[&Buffer],
        thread_group_count: &MTLSize,
        threads_per_threadgroup: &MTLSize,
    ) {
        let command_buffer = self.command_queue.new_command_buffer();
        let compute_pass_descriptor = ComputePassDescriptor::new();
        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

        encoder.set_compute_pipeline_state(pipeline_state);
        for (i, buffer) in buffers.iter().enumerate() {
            encoder.set_buffer(i as u64, Some(buffer), 0);
        }

        encoder.dispatch_thread_groups(*thread_group_count, *threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    pub fn read_results(&self, buffer: &Buffer, size: usize) -> Vec<u32> {
        read_buffer(buffer, size)
    }

    pub fn drop_all_buffers(&mut self) {
        self.buffers.clear();
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// Pallas base field modulus
fn pallas_modulus() -> BigUint {
    BigUint::from_str(
        "28948022309329048855892746252171976963363056481941560715954676764349967630337",
    )
    .unwrap()
}

pub fn get_or_calc_constants(num_limbs: usize, log_limb_size: u32) -> MSMConstants {
    let mut cache = CONSTANTS_CACHE.lock().unwrap();
    let key = (num_limbs, log_limb_size);

    if !cache.contains_key(&key) {
        let constants = calc_constants(num_limbs, log_limb_size);
        cache.insert(key, constants.clone());
        constants
    } else {
        cache.get(&key).unwrap().clone()
    }
}

fn calc_constants(num_limbs: usize, log_limb_size: u32) -> MSMConstants {
    let p = pallas_modulus();
    let r = calc_mont_radix(num_limbs, log_limb_size);
    let (rinv, n0) = calc_rinv_and_n0(&p, &r, log_limb_size);
    let nsafe = calc_nsafe(log_limb_size);
    let mu = calc_barrett_mu(&p);
    MSMConstants {
        p,
        r,
        rinv,
        n0,
        nsafe,
        mu,
    }
}
