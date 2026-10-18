import test from 'node:test';
import assert from 'node:assert/strict';
import { GpuPrePost } from '../src/engine/gpu-prepost.js';

test('resolution changes await asynchronous pipelines and reuse them for later sizes', async (t) => {
  const oldTexture = globalThis.GPUTextureUsage;
  const oldBuffer = globalThis.GPUBufferUsage;
  const oldGpu = globalThis.navigator.gpu;
  Object.defineProperty(globalThis.navigator, 'gpu', { configurable: true, value: { getPreferredCanvasFormat: () => 'bgra8unorm' } });
  globalThis.GPUTextureUsage = globalThis.GPUBufferUsage = {};
  t.after(() => {
    Object.defineProperty(globalThis.navigator, 'gpu', { configurable: true, value: oldGpu });
    globalThis.GPUTextureUsage = oldTexture; globalThis.GPUBufferUsage = oldBuffer;
  });
  let pipelines = 0, destroyed = 0;
  const resource = () => ({ createView: () => ({}), destroy() { destroyed++; } });
  const pipeline = () => { pipelines++; return { getBindGroupLayout: () => ({}) }; };
  let finishCompile;
  const compilation = new Promise(resolve => { finishCompile = resolve; });
  const asyncPipeline = async () => { await compilation; return pipeline(); };
  const device = {
    createTexture: resource, createBuffer: resource, createSampler: () => ({}),
    createShaderModule: () => ({}), createComputePipeline: pipeline, createRenderPipeline: pipeline,
    createComputePipelineAsync: asyncPipeline, createRenderPipelineAsync: asyncPipeline,
    createBindGroup: () => ({}), queue: { writeBuffer() {} },
  };
  const canvas = { getContext: () => ({ configure() {}, unconfigure() {} }) };
  const gpu = new GpuPrePost({ device, canvas });
  let configured = false;
  const first = Promise.resolve(gpu.configure({ sourceWidth: 1920, sourceHeight: 1080, inputWidth: 320, inputHeight: 180 })).then(() => {configured = true;});
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(configured, false, 'must not expose rendering resources before compilation finishes');
  finishCompile();
  await first;
  await gpu.configure({ sourceWidth: 1920, sourceHeight: 1080, inputWidth: 240, inputHeight: 132 });
  assert.equal(pipelines, 3);
  assert.equal(destroyed, 5, 'direct path does not allocate the full-sized fallback source');
  assert.deepEqual(gpu.getOutputShape(), [1, 3, 264, 480]);
  await gpu.configure({sourceWidth:1280,sourceHeight:720,inputWidth:1280,inputHeight:720,displayWidth:1920,displayHeight:1080});
  assert.equal(canvas.width,1920);
  assert.equal(canvas.height,1080);
  assert.deepEqual(gpu.getOutputShape(),[1,3,1440,2560], 'display scaling must not shrink the 2x model output buffer');
  gpu.destroy();
});
