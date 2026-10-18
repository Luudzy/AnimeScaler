import test from 'node:test';
import assert from 'node:assert/strict';
import { AdoreEngine } from '../src/engine/adore-engine.js';
import { Tensor } from 'onnxruntime-common';

const tick = () => new Promise(resolve => setImmediate(resolve));
function deferred() { let resolve; const promise = new Promise(done => { resolve = done; }); return { promise, resolve }; }
const metadata = (shape) => ({ isTensor: true, type: 'float32', shape });
function fixture() {
  const engine = new AdoreEngine();
  const events = [];
  const lost = deferred();
  const listeners = new Map();
  engine.device = { features: new Set(), lost: lost.promise, queue: { async onSubmittedWorkDone() { events.push('completed'); } },
    addEventListener(type, listener) { listeners.set(type, listener); }, removeEventListener(type) { listeners.delete(type); } };
  engine.settings = { model: 'adore', preset: 'performance' };
  engine.session = { async run() { events.push('run'); }, async release() { events.push('release'); },
    inputMetadata: [metadata([1, 3, 'h', 'w'])], outputMetadata: [metadata([1, 3, 'oh', 'ow'])] };
  engine.gpu = { uploadVideoFrame() {}, preprocess() { events.push('preprocess'); }, render() { events.push('render'); },
    destroy() { events.push('destroy'); }, async configure() { events.push('configure'); },
    getInputBuffer: () => ({}), getOutputBuffer: () => ({}),
    getInputShape: () => [1, 3, 360, 640], getOutputShape: () => [1, 3, 720, 1280] };
  engine.ort = { Tensor: { fromGpuBuffer: () => ({ dispose() { events.push('dispose'); } }) } };
  engine.inputTensor = { dispose() { events.push('dispose'); } };
  engine.outputTensor = { dispose() { events.push('dispose'); } };
  return { engine, events, lost, listeners, video: { videoWidth: 1280, videoHeight: 720 } };
}

test('FP16 capability is awaited and checked before creating candidate', async () => {
  const { engine } = fixture();
  engine.device = null;
  const urls = [];
  const session = { inputNames: ['input'], outputNames: ['output'],
    inputMetadata: [metadata([1, 3, 'h', 'w'])], outputMetadata: [metadata([1, 3, 'oh', 'ow'])], async release() {} };
  engine.ort = { env: { webgpu: { device: Promise.resolve({ features: new Set() }) } },
    InferenceSession: { async create(url) { urls.push(url); return session; } } };
  await engine.createSession();
  assert.equal(urls.length, 1);
  assert.match(urls[0], /webgpu\.onnx$/);
  assert.equal(engine.model.precision, 'fp32');
});

test('inference timeout is terminal but keeps buffers until run and queue settle', async t => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const { engine, events, video } = fixture();
  const run = deferred(), queue = deferred();
  engine.session.run = () => run.promise;
  engine.device.queue.onSubmittedWorkDone = () => queue.promise;
  const frame = engine.processFrame(video);
  let failure;
  frame.catch(error => { failure = error; });
  t.after(async () => { run.resolve(); queue.resolve(); await frame.catch(() => {}); await engine.destroy(); });
  await tick();
  t.mock.timers.tick(30000);
  await tick();
  assert.equal(failure?.code, 'INFERENCE_TIMEOUT');
  await assert.rejects(engine.reconfigure(video), { code: 'INFERENCE_TIMEOUT' });
  const destroying = engine.destroy();
  await tick();
  assert.equal(events.includes('destroy'), false);
  assert.equal(events.includes('release'), false);
  run.resolve();
  await tick();
  assert.equal(events.includes('render'), false);
  assert.equal(events.includes('destroy'), false);
  queue.resolve();
  await destroying;
  assert.equal(events.filter(value => value === 'destroy').length, 1);
  assert.equal(events.filter(value => value === 'release').length, 1);
});

test('resize disposes borrowed tensor wrappers before replacing GPU buffers', async () => {
  const { engine, events, video } = fixture();
  await engine.reconfigure(video);
  assert.deepEqual(events.slice(0, 3), ['dispose', 'dispose', 'configure']);
  await engine.destroy();
});

for (const mode of ['lost', 'uncapturederror']) test(`neural ${mode} terminates in-flight inference and prevents reuse`, async () => {
  const { engine, events, video, lost, listeners } = fixture();
  const run = deferred();
  engine.session.run = () => run.promise;
  engine.watchDevice();
  const code = mode === 'lost' ? 'GPU_DEVICE_LOST' : 'GPU_VALIDATION_ERROR';
  const frame = engine.processFrame(video);
  const rejected = assert.rejects(frame, { code });
  await tick();
  if (mode === 'lost') lost.resolve({ reason: 'unknown', message: 'lost' });
  else listeners.get('uncapturederror')({ error: { message: 'out of memory' }, preventDefault() {} });
  await rejected;
  await assert.rejects(engine.reconfigure(video), { code });
  await assert.rejects(engine.updateSettings(engine.settings, video), { code });
  const destroying = engine.destroy();
  run.resolve();
  await destroying;
  assert.equal(events.includes('render'), false);
  assert.equal(listeners.size, 0);
  const fresh = fixture();
  await fresh.engine.processFrame(fresh.video);
  assert.equal(fresh.events.includes('render'), true);
  await fresh.engine.destroy();
});

test('static metadata mismatch rejects resize before disposing or allocating buffers', async () => {
  const { engine, events, video } = fixture();
  engine.session.inputMetadata = [metadata([1, 3, 8, 8])];
  engine.session.outputMetadata = [metadata([1, 3, 16, 16])];
  await assert.rejects(engine.reconfigure(video), { code: 'MODEL_SHAPE_UNSUPPORTED' });
  assert.deepEqual(events, []);
  await engine.destroy();
});

test('extra model inputs reject candidate and release each rejected session', async () => {
  const { engine } = fixture();
  let released = 0;
  engine.ort = { InferenceSession: { async create() { return {
    inputNames: ['image', 'extra'], outputNames: ['output'],
    inputMetadata: [metadata([1, 3, 'h', 'w']), metadata([1, 3, 'h', 'w'])],
    outputMetadata: [metadata([1, 3, 'oh', 'ow'])], async release() { released++; },
  }; } } };
  await assert.rejects(engine.createSession(), { code: 'MODEL_INIT_FAILED' });
  assert.equal(released, 1);
});

test('resize disposes real ORT wrappers without destroying borrowed GPU buffers', async () => {
  const { engine, video } = fixture();
  let bufferDestroyed = 0;
  const buffer = { destroy() { bufferDestroyed++; } };
  const input = Tensor.fromGpuBuffer(buffer, { dataType: 'float32', dims: [1, 3, 8, 8] });
  const output = Tensor.fromGpuBuffer(buffer, { dataType: 'float32', dims: [1, 3, 16, 16] });
  engine.inputTensor = input;
  engine.outputTensor = output;
  await engine.reconfigure(video);
  assert.throws(() => input.gpuBuffer, /disposed/);
  assert.throws(() => output.gpuBuffer, /disposed/);
  assert.equal(bufferDestroyed, 0);
  await engine.destroy();
});

test('GPU completion timeout also quarantines neural buffers', async t => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const { engine, events, video } = fixture();
  const queue = deferred();
  engine.device.queue.onSubmittedWorkDone = () => queue.promise;
  const frame = engine.processFrame(video);
  let failure;
  frame.catch(error => { failure = error; });
  t.after(async () => { queue.resolve(); await frame.catch(() => {}); await engine.destroy(); });
  await tick();
  assert.equal(events.includes('render'), true);
  t.mock.timers.tick(30000);
  await tick();
  assert.equal(failure?.code, 'INFERENCE_TIMEOUT');
  const destroying = engine.destroy();
  await tick();
  assert.equal(events.includes('destroy'), false);
  queue.resolve();
  await destroying;
});
