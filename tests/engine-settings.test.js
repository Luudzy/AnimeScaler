import test from 'node:test';
import assert from 'node:assert/strict';
import { AdoreEngine } from '../src/engine/adore-engine.js';

test('model switch waits for inference and releases old session before creating new one', async () => {
  const engine = new AdoreEngine();
  const events = [];
  let finish;
  engine.settings = { model: 'adore', preset: 'performance' };
  engine.session = { run: () => new Promise((r) => { finish = r; }), release: async () => { events.push('released'); } };
  engine.gpu = { uploadVideoFrame() {}, preprocess() {}, render() { events.push('rendered'); } };
  engine.inputTensor = {}; engine.outputTensor = {};
  engine.createSession = async () => { events.push(engine.settings.model); engine.session = {}; };
  engine.reconfigureNow = async () => { events.push('configured'); };
  const frame = engine.processFrame({});
  await new Promise((r) => setImmediate(r));
  assert.equal(typeof engine.updateSettings, 'function');
  const change = engine.updateSettings({ model: 'animejanai', preset: 'performance' }, {});
  await new Promise((r) => setImmediate(r));
  assert.deepEqual(events, []);
  finish(); await frame; await change;
  assert.deepEqual(events, ['rendered', 'released', 'animejanai', 'configured']);
});

test('frame completion and resource disposal wait for GPU queue completion', async () => {
  const engine = new AdoreEngine();
  let finishGpu;
  let destroyed = false;
  const gpuWork = new Promise((r) => { finishGpu = r; });
  engine.device = { queue: { onSubmittedWorkDone: () => gpuWork } };
  engine.session = { run: async () => {}, release: async () => {} };
  engine.gpu = { uploadVideoFrame() {}, preprocess() {}, render() {}, destroy() { destroyed = true; } };
  engine.inputTensor = {}; engine.outputTensor = {};
  let completed = false;
  const frame = engine.processFrame({}).then(() => { completed = true; });
  const teardown = engine.destroy();
  await new Promise((r) => setImmediate(r));
  assert.equal(completed, false);
  assert.equal(destroyed, false);
  finishGpu();
  await frame; await teardown;
  assert.equal(destroyed, true);
});
