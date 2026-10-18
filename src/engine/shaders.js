export const PREPROCESS_WGSL = /* wgsl */ `
struct Dims {
  width: u32,
  height: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var sourceTexture: texture_2d<f32>;
@group(0) @binding(1) var sourceSampler: sampler;
@group(0) @binding(2) var<storage, read_write> inputTensor: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= dims.width || id.y >= dims.height) {
    return;
  }

  let uv = (vec2<f32>(id.xy) + vec2<f32>(0.5)) / vec2<f32>(f32(dims.width), f32(dims.height));
  let color = textureSampleLevel(sourceTexture, sourceSampler, uv, 0.0);
  let pixel = id.y * dims.width + id.x;
  let plane = dims.width * dims.height;

  inputTensor[pixel] = color.r;
  inputTensor[plane + pixel] = color.g;
  inputTensor[2u * plane + pixel] = color.b;
}
`;

export const EXTERNAL_PREPROCESS_WGSL = PREPROCESS_WGSL
  .replace('texture_2d<f32>', 'texture_external')
  .replace('@compute @workgroup_size',
    'struct Crop { rect: vec4f, bounds: vec4f };\n@group(0) @binding(4) var<uniform> crop: Crop;\n@compute @workgroup_size')
  .replace('textureSampleLevel(sourceTexture, sourceSampler, uv, 0.0)',
    'textureSampleBaseClampToEdge(sourceTexture, sourceSampler, clamp(crop.rect.xy + uv * crop.rect.zw, crop.bounds.xy, crop.bounds.zw))');

export const POSTPROCESS_WGSL = /* wgsl */ `
struct Dims {
  width: u32,
  height: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> outputTensor: array<f32>;
@group(0) @binding(1) var outputTexture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> dims: Dims;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= dims.width || id.y >= dims.height) {
    return;
  }

  let pixel = id.y * dims.width + id.x;
  let plane = dims.width * dims.height;
  let color = clamp(vec3<f32>(
    outputTensor[pixel],
    outputTensor[plane + pixel],
    outputTensor[2u * plane + pixel]
  ), vec3<f32>(0.0), vec3<f32>(1.0));

  textureStore(outputTexture, vec2<i32>(id.xy), vec4<f32>(color, 1.0));
}
`;

export const RENDER_WGSL = /* wgsl */ `
struct VertexOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vertexMain(@builtin(vertex_index) index: u32) -> VertexOut {
  var positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(3.0, -1.0),
    vec2<f32>(-1.0, 3.0)
  );
  let p = positions[index];
  var out: VertexOut;
  out.position = vec4<f32>(p, 0.0, 1.0);
  out.uv = vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
  return out;
}

@group(0) @binding(0) var imageTexture: texture_2d<f32>;
@group(0) @binding(1) var imageSampler: sampler;

@fragment
fn fragmentMain(in: VertexOut) -> @location(0) vec4<f32> {
  return textureSample(imageTexture, imageSampler, in.uv);
}
`;
