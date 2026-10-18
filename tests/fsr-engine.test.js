import test from 'node:test';
import assert from 'node:assert/strict';
import { FsrEngine } from '../src/engine/fsr-engine.js';
import { FSR_EASU_WGSL, FSR_RCAS_WGSL } from '../src/engine/fsr-shaders.js';

test('EASU shader avoids reserved WGSL identifiers', () => {
  assert.doesNotMatch(FSR_EASU_WGSL, /\bvar set\b/);
});

test('EASU preserves the signed reconstruction weight when normalizing taps', () => {
  assert.doesNotMatch(FSR_EASU_WGSL, /max\(abs\(accumulated\.weight\)/);
  assert.match(FSR_EASU_WGSL, /let normalized=select\(f,accumulated\.color\/accumulated\.weight/);
});

test('RCAS includes the center pixel when it computes local contrast bounds', () => {
  assert.match(FSR_RCAS_WGSL, /let hitMin=localMin\/max\(4\.\*localMax/);
  assert.match(FSR_RCAS_WGSL, /let hitMax=\(vec3f\(1\.\)-localMax\)/);
});

function makeDevice() {
  const calls = { copies: [], submits: 0, passes: 0, destroyed: 0, writes: [], textures: [], bindings: [], gpuCopies:0 };
  const texture = (descriptor) => { calls.textures.push(descriptor); return { createView: () => ({}), destroy: () => { calls.destroyed += 1; } }; };
  const device = {
    adapterInfo: { vendor: 'amd', architecture: 'rdna1', description: 'Radeon RX 5500 XT' },
    queue: {
      copyExternalImageToTexture: (...args) => calls.copies.push(args),
      writeBuffer: (_buffer, _offset, values) => calls.writes.push([...values]),
      submit: () => { calls.submits += 1; },
      onSubmittedWorkDone: async () => {},
    },
    createTexture: texture,
    createBuffer: () => ({ destroy: () => { calls.destroyed += 1; } }),
    createShaderModule: () => ({}),
    createRenderPipelineAsync: async () => ({ getBindGroupLayout: () => ({}) }),
    createBindGroup: (descriptor) => { calls.bindings.push(descriptor.entries.map(entry => entry.binding)); return {}; },
    createSampler: () => ({}),
    createCommandEncoder: () => ({
      copyTextureToTexture: () => { calls.gpuCopies++; },
      beginRenderPass: () => ({ setPipeline() {}, setBindGroup() {}, draw() {}, end() { calls.passes += 1; } }),
      finish: () => ({}),
    }),
    destroy() {},
  };
  return { device, calls };
}

function makeCanvas() {
  return {
    width: 0, height: 0,
    getContext: () => ({ configure() {}, unconfigure() {}, getCurrentTexture: () => ({ createView: () => ({}) }) }),
  };
}

test('FSR engine uses two GPU render passes with no CPU pixel readback', async () => {
  const { device, calls } = makeDevice();
  const engine = new FsrEngine({ requestAdapter: async () => ({ requestDevice: async () => device }) });
  const canvas = makeCanvas();
  const video = { videoWidth: 1280, videoHeight: 720 };
  await engine.initialize({ canvas, video, settings: { model: 'fsr1-spatial', preset: '1080p', spatialDenoise: 'off' } });
  const stats = await engine.processFrame(video);
  assert.deepEqual(
    { inputWidth: stats.inputWidth, inputHeight: stats.inputHeight, outputWidth: stats.outputWidth, outputHeight: stats.outputHeight },
    { inputWidth: 1280, inputHeight: 720, outputWidth: 1920, outputHeight: 1080 },
  );
  assert.equal(stats.backend, 'webgpu');
  assert.equal(stats.modelLabel, 'AMD FSR 1 Spatial');
  assert.equal(calls.copies.length, 1);
  assert.equal(calls.passes, 2);
  assert.equal(calls.submits, 1);
  assert.deepEqual(calls.writes[0], [1280, 720, 1920, 1080]);
  assert.deepEqual(calls.bindings[0], [0,2], 'EASU auto layout excludes unused samplers');
  assert.ok((calls.textures[0].usage & 4) !== 0, 'source texture accepts external-image copies');
  await engine.destroy();
});

test('FSR accepts canvas-style sources that expose width and height', async () => {
  const { device, calls } = makeDevice();
  const engine = new FsrEngine({ requestAdapter: async () => ({ requestDevice: async () => device }) });
  const source = { width: 1280, height: 720 };
  await engine.initialize({ canvas: makeCanvas(), video: source, settings: { model: 'fsr1-spatial', preset: '1080p' } });
  await engine.processFrame(source);
  assert.deepEqual(calls.writes[0], [1280, 720, 1920, 1080]);
  await engine.destroy();
});

test('FSR rejects initialization when GPU resource validation fails', async () => {
  const { device } = makeDevice();
  let popped = 0;
  device.pushErrorScope = () => {};
  device.popErrorScope = async () => { popped++; return { message: 'Invalid bind group' }; };
  const engine = new FsrEngine({ requestAdapter: async () => ({ requestDevice: async () => device }) });
  await assert.rejects(engine.initialize({ canvas: makeCanvas(), video: { width: 640, height: 360 }, settings: { model: 'fsr1-spatial', preset: '720p' } }), error => error.code === 'GPU_VALIDATION_ERROR' && error.message.includes('Invalid bind group'));
  assert.equal(popped, 1);
  await engine.destroy();
});

test('FSR surfaces asynchronous GPU errors instead of continuing to report rendered frames', async () => {
  const { device, calls } = makeDevice();
  let reportError;
  device.addEventListener = (type, callback) => { if (type === 'uncapturederror') reportError = callback; };
  const engine = new FsrEngine({ requestAdapter: async () => ({ requestDevice: async () => device }) });
  const source = { width: 640, height: 360 };
  await engine.initialize({ canvas: makeCanvas(), video: source, settings: { model: 'fsr1-spatial', preset: '720p' } });
  reportError({ error: { message: 'Invalid command buffer' }, preventDefault() {} });
  await assert.rejects(engine.processFrame(source), error => error.code === 'GPU_VALIDATION_ERROR');
  assert.equal(calls.submits, 0);
  await engine.destroy();
});

test('FSR denoises on GPU before upscale and can change strength without reallocating textures', async () => {
  const { device, calls } = makeDevice();
  const engine = new FsrEngine({ requestAdapter: async () => ({ requestDevice: async () => device }) });
  const source = { width: 640, height: 360 };
  await engine.initialize({ canvas: makeCanvas(), video: source, settings: { model: 'fsr1-spatial', preset: '720p', spatialDenoise: 'balanced' } });
  await engine.processFrame(source);
  assert.equal(calls.passes, 3);
  const allocations = calls.textures.length;
  await engine.updateSettings({ model: 'fsr1-spatial', preset: '720p', spatialDenoise: 'off' }, source);
  await engine.processFrame(source);
  assert.equal(calls.passes, 5);
  await engine.updateSettings({ model: 'fsr1-spatial', preset: '720p', spatialDenoise: 'strong' }, source);
  const stats = await engine.processFrame(source);
  assert.equal(calls.passes, 8);
  assert.equal(calls.submits, 3);
  assert.equal(calls.textures.length, allocations);
  assert.equal(stats.spatialDenoise, 'strong');
  await engine.destroy();
});

test('FSR performance preset resizes internally on GPU while keeping the requested output', async () => {
  const { device, calls } = makeDevice();
  const engine = new FsrEngine({ requestAdapter: async () => ({ requestDevice: async () => device }) });
  const source = { width:1920,height:1080 };
  await engine.initialize({canvas:makeCanvas(),video:source,settings:{model:'fsr1-spatial',preset:'1080p',fsrQuality:'performance',spatialDenoise:'off'}});
  const stats=await engine.processFrame(source);
  assert.equal(stats.inputWidth,960);
  assert.equal(stats.outputWidth,1920);
  assert.deepEqual(calls.copies[0][2],[1920,1080]);
  assert.equal(calls.passes,3);
  assert.equal(calls.submits,1);
  await engine.destroy();
});

test('FSR bounds work in flight and reports a frame only after GPU completion', async () => {
  const {device,calls}=makeDevice();
  let finish;
  device.queue.onSubmittedWorkDone=()=>new Promise(resolve=>{finish=resolve;});
  const engine=new FsrEngine({requestAdapter:async()=>({requestDevice:async()=>device})});
  const source={width:640,height:360};
  await engine.initialize({canvas:makeCanvas(),video:source,settings:{model:'fsr1-spatial',preset:'720p',spatialDenoise:'off'}});
  let completed=false;
  const first=engine.processFrame(source).then(stats=>{completed=true;return stats;});
  const second=engine.processFrame(source);
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(completed,false,'submission is not a completed frame');
  assert.equal(calls.submits,1,'second GPU frame cannot accumulate behind the first');
  finish();const stats=await first;
  assert.equal(stats.timingScope,'gpu-completed-frame');
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(calls.submits,2);finish();await second;await engine.destroy();
});

test('FSR native 1:1 uses only cleanup and sharpening, without EASU reconstruction',async()=>{
  const {device,calls}=makeDevice();
  const engine=new FsrEngine({requestAdapter:async()=>({requestDevice:async()=>device})});
  const source={width:1280,height:720};
  await engine.initialize({canvas:makeCanvas(),video:source,settings:{model:'fsr1-spatial',preset:'720p',fsrQuality:'native-aa',spatialDenoise:'off'}});
  const stats=await engine.processFrame(source);
  assert.equal(calls.passes,1);assert.equal(stats.easuApplied,false);
  await engine.updateSettings({model:'fsr1-spatial',preset:'720p',fsrQuality:'native-aa',spatialDenoise:'balanced'},source);
  await engine.processFrame(source);assert.equal(calls.passes,3);
  await engine.destroy();
});

test('FSR does not report successful rendering when GPU completion fails',async()=>{
  const {device}=makeDevice();
  device.queue.onSubmittedWorkDone=async()=>{throw Error('Device lost');};
  const engine=new FsrEngine({requestAdapter:async()=>({requestDevice:async()=>device})});
  const source={width:640,height:360};
  await engine.initialize({canvas:makeCanvas(),video:source,settings:{model:'fsr1-spatial',preset:'720p'}});
  await assert.rejects(engine.processFrame(source),error=>error.code==='GPU_COMPLETION_ERROR');
  await assert.rejects(engine.processFrame(source),error=>error.code==='GPU_COMPLETION_ERROR');
  await engine.destroy();
});

test('ASR temporal alternates GPU history and resets on seek, crop and noise changes', async () => {
  const {device,calls}=makeDevice();
  const engine=new FsrEngine({requestAdapter:async()=>({requestDevice:async()=>device})});
  const source={width:640,height:360,currentTime:1};
  const settings={model:'asr-anime-temporal',preset:'720p',spatialDenoise:'off'};
  await engine.initialize({canvas:makeCanvas(),video:source,settings});
  assert.equal((await engine.processFrame(source)).temporalHistoryAvailable,false);
  source.currentTime+=1/24;
  assert.equal((await engine.processFrame(source)).temporalHistoryAvailable,true);
  const allocations=calls.textures.length;
  source.currentTime=8;
  assert.equal((await engine.processFrame(source)).temporalHistoryAvailable,false);
  source.currentTime+=1/24;
  assert.equal((await engine.processFrame(source,{x:2,y:0,width:640,height:360})).temporalHistoryAvailable,false);
  await engine.updateSettings({...settings,spatialDenoise:'strong'},source);
  source.currentTime+=1/24;
  assert.equal((await engine.processFrame(source)).temporalHistoryAvailable,false);
  await engine.resetHistory();
  source.currentTime+=1/24;
  assert.equal((await engine.processFrame(source)).temporalHistoryAvailable,false);
  assert.equal(calls.textures.length,allocations);
  assert.equal(calls.gpuCopies,6);
  assert.equal(calls.submits,6);
  assert.equal(calls.passes,20);
  await engine.destroy();
});

test('FSR device loss interrupts completion and forbids resource reconfiguration', async () => {
  const { device, calls } = makeDevice();
  let lose, finish;
  device.lost = new Promise(resolve => { lose = resolve; });
  device.queue.onSubmittedWorkDone = () => new Promise(resolve => { finish = resolve; });
  const engine = new FsrEngine({ requestAdapter: async () => ({ requestDevice: async () => device }) });
  const source = { width: 640, height: 360 };
  await engine.initialize({ canvas: makeCanvas(), video: source, settings: { model: 'fsr1-spatial', preset: '720p' } });
  const frame = engine.processFrame(source);
  let failure;
  frame.catch(error => { failure = error; });
  await new Promise(resolve => setImmediate(resolve));
  lose({ message: 'device lost', reason: 'unknown' });
  await new Promise(resolve => setImmediate(resolve));
  // Complete the fake queue even when the assertion fails, so the CPU test never hangs.
  finish();
  await frame.catch(() => {});
  assert.equal(failure?.code, 'GPU_DEVICE_LOST');
  const allocations = calls.textures.length;
  await assert.rejects(engine.reconfigure(source), { code: 'GPU_DEVICE_LOST' });
  assert.equal(calls.textures.length, allocations);
  const destroying = engine.destroy();
  assert.equal(engine.destroy(), destroying);
  await destroying;
});

test('FSR GPU errors during submitted work terminate frame and cleanup exactly once', async () => {
  const { device } = makeDevice();
  let reportError, destroyed = 0, detached = 0;
  device.addEventListener = (_, listener) => { reportError = listener; };
  device.removeEventListener = () => { detached++; };
  device.destroy = () => { destroyed++; };
  device.queue.onSubmittedWorkDone = () => new Promise(() => {});
  const engine = new FsrEngine({ requestAdapter: async () => ({ requestDevice: async () => device }) });
  const source = { width: 640, height: 360 };
  await engine.initialize({ canvas: makeCanvas(), video: source, settings: { model: 'fsr1-spatial', preset: '720p' } });
  const frame = engine.processFrame(source);
  const rejection = assert.rejects(frame, { code: 'GPU_VALIDATION_ERROR' });
  await new Promise(resolve => setImmediate(resolve));
  reportError({ error: { message: 'OOM' }, preventDefault() {} });
  await rejection;
  await assert.rejects(engine.updateSettings(engine.settings, source), { code: 'GPU_VALIDATION_ERROR' });
  await engine.destroy();
  await engine.destroy();
  assert.equal(destroyed, 1);
  assert.equal(detached, 1);
});
