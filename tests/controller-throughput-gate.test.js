import test from 'node:test';
import assert from 'node:assert/strict';
import { AnimeScalerController } from '../src/content/controller.js';
import { SustainedFpsGate } from '../src/core/sustained-fps-gate.js';

function setupController(backend, t) {
  const previous = Object.fromEntries(['document', 'window', 'MutationObserver'].map(key => [key, globalThis[key]]));
  globalThis.document = { querySelectorAll() { return []; }, documentElement: {} };
  globalThis.window = { removeEventListener() {} };
  globalThis.MutationObserver = class { disconnect() {} };
  t.after(() => { for (const [key, value] of Object.entries(previous)) globalThis[key] = value; });
  const source = {
    isConnected: true,
    videoWidth: 640,
    videoHeight: 360,
    readyState: 4,
    paused: false,
    ended: false,
    currentTime: 0,
  };
  const events = [];
  let pauses = 0;
  source.pause = () => { pauses += 1; source.paused = true; };

  const controller = new AnimeScalerController();
  controller.state = 'running';
  controller.settings = { ...controller.settings, backend, maxFps: 24, adaptive: false };
  controller.video = controller.displayTarget = source;
  controller.throughputGate = backend === 'webgpu' ? null : new SustainedFpsGate({ targetFps: 24, windowMs: 5000 });
  controller.diagnostics = {
    sourceFrame() {},
    error(error) { events.push({ type: 'error', code: error.code }); },
    event(type, data) { events.push({ type, ...data }); },
  };
  controller.overlay = { show() {}, hide() { events.push({ type: 'overlay_hidden' }); } };
  controller.hud = { update() {} };
  controller.mutationObserver = { disconnect() {} };
  controller.scheduleNextFrame = () => {};
  controller.syncGeometry = () => {};
  controller.framePacer.take = () => true;
  controller.engine = {
    needsReconfigure: () => false,
    async processFrame() { return { warmup: false, totalMs: 50 }; },
    async destroy() {},
  };

  return { controller, source, events, get pauses() { return pauses; } };
}

test('sustained native throughput below target stops enhancement and leaves source playing', async t => {
  const { controller, source, events, pauses } = setupController('rocm', t);
  controller.observedSourceFps = 24;

  for (let now = 0; now <= 5000; now += 250) {
    source.currentTime = now / 1000;
    await controller.onVideoFrame(now, { mediaTime: now / 6000 });
  }

  assert.equal(controller.state, 'error');
  assert.equal(controller.error.code, 'THROUGHPUT_BELOW_TARGET');
  assert.equal(source.paused, false);
  assert.equal(pauses, 0);
  assert.ok(events.some(event => event.type === 'throughput_guard' && event.targetFps === 24));
  assert.ok(events.some(event => event.type === 'overlay_hidden'));
});

test('OpenCL throughput guard stops stale enhancement and leaves source playing', async t => {
  const { controller, source, events, pauses } = setupController('opencl', t);
  controller.observedSourceFps = 24;

  for (let now = 0; now <= 5000; now += 250) {
    source.currentTime = now / 1000;
    await controller.onVideoFrame(now, { mediaTime: now / 6000 });
  }

  assert.equal(controller.state, 'error');
  assert.equal(controller.error.code, 'THROUGHPUT_BELOW_TARGET');
  assert.equal(source.paused, false);
  assert.equal(pauses, 0);
  assert.ok(events.some(event => event.type === 'throughput_guard' && event.backend === 'opencl'));
  assert.ok(events.some(event => event.type === 'overlay_hidden'));
});

test('throughput guard does not change WebGPU playback', async t => {
  const { controller, source, events, pauses } = setupController('webgpu', t);

  for (let now = 0; now <= 5000; now += 250) {
    source.currentTime = now / 1000;
    await controller.onVideoFrame(now, { mediaTime: source.currentTime });
  }

  assert.equal(controller.state, 'running');
  assert.equal(source.paused, false);
  assert.equal(pauses, 0);
  assert.equal(events.some(event => event.type === 'throughput_guard'), false);
});

test('ROCm guard caps minimum cadence to source frame rate', async t => {
  const { controller, source, events } = setupController('rocm', t);
  const interval = 1000 / 12;

  for (let index = 0; index <= 60; index += 1) {
    const now = index * interval;
    source.currentTime = now / 1000;
    await controller.onVideoFrame(now, { mediaTime: source.currentTime });
  }

  assert.equal(controller.state, 'running');
  assert.equal(events.some(event => event.type === 'throughput_guard'), false);
});
