use metal::*;

pub fn get_default_device() -> metal::Device {
    Device::system_default().expect("No Metal device found")
}

#[allow(unsafe_code)]
pub fn create_buffer(device: &Device, data: &[u32]) -> metal::Buffer {
    device.new_buffer_with_data(
        data.as_ptr().cast(),
        (data.len() * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::CPUCacheModeDefaultCache,
    )
}

#[allow(unsafe_code)]
pub fn read_buffer(result_buf: &metal::Buffer, num_u32s: usize) -> Vec<u32> {
    let ptr = result_buf.contents() as *const u32;
    assert!(!ptr.is_null(), "Metal buffer pointer is null");
    unsafe { std::slice::from_raw_parts(ptr, num_u32s) }.to_vec()
}

pub fn create_empty_buffer(device: &Device, size: usize) -> metal::Buffer {
    let data = vec![0u32; size];
    create_buffer(device, &data)
}
