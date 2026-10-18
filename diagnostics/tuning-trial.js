import * as ort from '../vendor/ort.webgpu.bundle.min.mjs';
import { AdoreEngine } from '../src/engine/adore-engine.js';
import { GpuPrePost } from '../src/engine/gpu-prepost.js';
import { getModelCandidates } from '../src/engine/model-config.js';
import { DiagnosticLog } from '../src/core/diagnostic-log.js';
import { sharedPattern } from './shared-pattern.js';

const config = JSON.parse(decodeURIComponent(location.hash.slice(1)));
const engine = new AdoreEngine(), operators = new Map();
engine.diagnostics = new DiagnosticLog({send() {}});
let staging;
const result = {};
try {
  const adapter = await navigator.gpu.requestAdapter({powerPreference: 'high-performance'});
  if (!adapter) throw new Error('WebGPU indisponível');
  const info = adapter.info ?? {};
  result.gpu = {vendor: info.vendor, architecture: info.architecture, description: info.description,
    shaderF16: adapter.features.has('shader-f16'), timestampQuery: adapter.features.has('timestamp-query')};
  if (config.precision === 'fp16' && !result.gpu.shaderF16) throw new Error('shader-f16 indisponível');
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = new URL('../vendor/', location.href).href;
  ort.env.webgpu.powerPreference = 'high-performance';
  ort.env.webgpu.profiling = {mode: config.profile ? 'default' : 'off', ondata(data) {
    const key = `${data.kernelName}/${data.programName}`;
    const item = operators.get(key) ?? {name: key, calls: 0, totalMs: 0};
    item.calls++; item.totalMs += (data.endTime - data.startTime) / 1e6;
    operators.set(key, item);
  }};
  const source = document.createElement('canvas');
  source.width = source.videoWidth = config.preset === '1080p' ? 1280 : 640;
  source.height = source.videoHeight = config.preset === '1080p' ? 720 : 360;
  const ctx = source.getContext('2d');
  const pattern = sharedPattern(source.width, source.height);
  ctx.putImageData(new ImageData(pattern.rgba, source.width, source.height), 0, 0);
  result.inputSha256 = [...new Uint8Array(await crypto.subtle.digest('SHA-256', pattern.nchw.buffer))]
    .map(value => value.toString(16).padStart(2, '0')).join('');
  result.inputPattern = 'integer-grid-v1';
  engine.ort = ort;
  engine.model = getModelCandidates(path => '../' + path, config.model ?? 'animejanai').find(m => m.precision === config.precision);
  engine.session = await ort.InferenceSession.create(engine.model.url, {
    executionProviders: [{name: 'webgpu', preferredLayout: config.layout}], graphOptimizationLevel: 'all', logSeverityLevel: 3,
    ...(config.graphCapture ? {enableGraphCapture: true, freeDimensionOverrides: {batch_size: 1, height: source.height, width: source.width}} : {}),
  });
  engine.device = ort.env.webgpu.device;
  engine.inputName = engine.session.inputNames[0]; engine.outputName = engine.session.outputNames[0];
  engine.settings = {preset: config.preset};
  engine.gpu = new GpuPrePost({device: engine.device, canvas: document.getElementById('output'), workgroup: config.workgroup});
  await engine.reconfigure(source);
  const times = [];
  for (let i = 0; i < 25; i++) {
    const stats = await engine.processFrame(source);
    if (i >= 5) times.push(stats.totalMs);
  }
  times.sort((a, b) => a - b);
  result.medianMs = (times[9] + times[10]) / 2; result.p95Ms = times[18];
  result.within30FpsPipelineBudget = result.p95Ms < 1000 / 30;
  result.size = engine.currentSize;
  // One diagnostic readback, outside measured frames; production keeps tensors on GPU.
  const buffer = engine.gpu.getOutputBuffer();
  staging = engine.device.createBuffer({size: buffer.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
  const encoder = engine.device.createCommandEncoder(); encoder.copyBufferToBuffer(buffer, 0, staging, 0, buffer.size);
  engine.device.queue.submit([encoder.finish()]); await staging.mapAsync(GPUMapMode.READ);
  const values = new Float32Array(staging.getMappedRange());
  const count = engine.currentSize.modelOutputWidth * engine.currentSize.modelOutputHeight * 3;
  result.samples = Array.from({length: 4096}, (_, i) => values[Math.floor(i * count / 4096)]);
  staging.unmap();
  result.operators = [...operators.values()].sort((a, b) => b.totalMs - a.totalMs);
  result.profilingStatus = !config.profile ? 'disabled' : operators.size ? 'available' : 'unavailable-no-callbacks';
  result.operatorScope = 'Chamadas recebidas, incluindo aquecimento. Lista vazia não significa custo zero: o backend instalado pode não expor o callback, mesmo com timestamp-query.';
} catch (error) { result.error = error.message; }
finally {
  staging?.destroy();
  try { await engine.destroy(); } catch { /* failed trial */ }
  parent.postMessage({token: config.token, result}, location.origin);
}
