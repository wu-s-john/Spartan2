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

#[allow(unsafe_code)]
pub fn create_empty_buffer(device: &Device, size: usize) -> metal::Buffer {
  let byte_len = (size * std::mem::size_of::<u32>()) as u64;
  let buffer = device.new_buffer(
    byte_len,
    MTLResourceOptions::StorageModeShared | MTLResourceOptions::CPUCacheModeDefaultCache,
  );
  let ptr = buffer.contents() as *mut u8;
  assert!(!ptr.is_null(), "Metal buffer pointer is null");
  unsafe { std::ptr::write_bytes(ptr, 0, byte_len as usize) };
  buffer
}

pub fn create_uninitialized_buffer(device: &Device, size: usize) -> metal::Buffer {
  let byte_len = (size * std::mem::size_of::<u32>()) as u64;
  device.new_buffer(
    byte_len,
    MTLResourceOptions::StorageModeShared | MTLResourceOptions::CPUCacheModeDefaultCache,
  )
}
