use std::{
  env, fs,
  io::Write,
  path::{Path, PathBuf},
};

fn main() -> std::io::Result<()> {
  if env::var("CARGO_FEATURE_GPU").is_ok() {
    embed_shader_source()?;
  } else {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let dest = out_dir.join("built_shaders.rs");
    let mut f = fs::File::create(&dest)?;
    writeln!(f, "// GPU feature not enabled — no Metal shaders")?;
  }
  Ok(())
}

/// Instead of pre-compiling Metal shaders (which requires Xcode),
/// embed the shader source as a string and compile at runtime.
fn embed_shader_source() -> std::io::Result<()> {
  let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
  let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

  let shader_root = manifest_dir
    .join("src")
    .join("provider")
    .join("metal_msm")
    .join("shader");

  // Collect all shader files and concatenate into one source string
  // Order matters: constants → types → bigint → field → mont → curve → cuzk kernels
  let shader_files = [
    "constants.metal",
    "misc/types.metal",
    "misc/get_constant.metal",
    "bigint/bigint.metal",
    "field/ff.metal",
    "mont_backend/mont.metal",
    "curve/utils.metal",
    "curve/jacobian.metal",
    "cuzk/extract_word_from_bytes_le.metal",
    "cuzk/barrett_reduction.metal",
    "cuzk/convert_point_coords_and_decompose_scalars.metal",
    "cuzk/transpose.metal",
    "cuzk/smvp.metal",
    "cuzk/pbpr.metal",
  ];

  let mut combined_source = String::new();
  combined_source.push_str("#include <metal_stdlib>\n");
  combined_source.push_str("#include <metal_math>\n");
  combined_source.push_str("using namespace metal;\n\n");

  for file in &shader_files {
    let path = shader_root.join(file);
    let content =
      fs::read_to_string(&path).unwrap_or_else(|e| panic!("Failed to read shader {}: {}", file, e));

    // Strip #include directives and #pragma once (we're concatenating manually)
    let filtered: String = content
      .lines()
      .filter(|line| {
        let trimmed = line.trim();
        !trimmed.starts_with("#include")
          && !trimmed.starts_with("#pragma once")
          && !trimmed.starts_with("using namespace metal")
      })
      .collect::<Vec<_>>()
      .join("\n");

    combined_source.push_str(&format!("// === {} ===\n", file));
    combined_source.push_str(&filtered);
    combined_source.push_str("\n\n");

    println!("cargo:rerun-if-changed={}", path.to_string_lossy());
  }

  // Write the combined source to a file
  let source_path = out_dir.join("msm_combined.metal");
  fs::write(&source_path, &combined_source)?;

  // Emit built_shaders.rs that includes the source as a string constant
  let dest = out_dir.join("built_shaders.rs");
  let mut f = fs::File::create(&dest)?;
  writeln!(
    f,
    r#"pub const MSM_SHADER_SOURCE: &str = include_str!(concat!(env!("OUT_DIR"), "/msm_combined.metal"));"#
  )?;

  Ok(())
}
