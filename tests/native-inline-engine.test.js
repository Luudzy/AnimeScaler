import test from 'node:test';
import assert from 'node:assert/strict';
import { NativeInlineEngine } from '../src/engine/native-inline-engine.js';
import { sessionTag } from '../src/core/session-diagnostics.js';

const png = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII=';

function fixture({ captureError } = {}) {
  const calls = [];
  const client = {
    async start(options) { calls.push(['start', options]); this.sessionId = 'session-1'; },
    async processFrame(frame) { calls.push(['frame', frame]); return { sequence: frame.sequence,
      mediaTime: frame.mediaTime, width: 2, height: 2, mime: 'image/png', dataUrl: png }; },
    async configure(options) { calls.push(['configure', options]); },
    async stop(reason) { calls.push(['stop', reason]); },
  };
  const context = { drawImage(video) { calls.push(['draw', video]); } };
  const canvas = { width: 0, height: 0, getContext: () => context,
    toDataURL: () => { if (captureError) throw captureError; return png; } };
  const presenter = { async presentFrame(frame, isCurrent) { calls.push(['present', frame, isCurrent()]); return true; } };
  const engine = new NativeInlineEngine({ createClient: () => client, createCanvas: () => canvas });
  const video = { videoWidth: 1, videoHeight: 1, currentTime: 2, paused: false };
  return { engine, calls, client, canvas, presenter, video };
}

const settings = { backend: 'rocm', model: 'zenkai-1-0', preset: 'performance',
  adaptive: true, exclusiveViewer: false };

test('inline stats preserve known metrics and separate presenter latency', async t => {
  const state = fixture();
  let now = 0;
  t.mock.method(performance, 'now', () => now);
  state.canvas.toDataURL = () => { now += 2; return png; };
  state.client.processFrame = async frame => { now += 7; return { ...frame, stats: {
    inferenceMs: 3, captureMs: 1, warmup: true, reused: true, modelPrecision: 'fp16', layout: 'NCHW',
    modelLabel: 'Zenkai 1.0 ROCm', sourceMethod: 'texture-copy',
    transportMs: Infinity, token: 'secret', url: 'http://private', temporal: { acceptedResults: 4, token: 'secret' },
  } }; };
  state.presenter.presentFrame = async () => { now += 11; return true; };
  await state.engine.initialize({ video: state.video, settings, presenter: state.presenter });
  const stats = await state.engine.processFrame(state.video);
  assert.equal(stats.inferenceMs, 3);
  assert.equal(stats.warmup, true);
  assert.equal(stats.reused, true);
  assert.equal(stats.roundTripMs, 7);
  assert.equal(stats.presentationMs, 11);
  assert.equal(stats.totalMs, 20);
  assert.equal(stats.modelPrecision, 'fp16');
  assert.equal(stats.modelLabel, 'Zenkai 1.0 ROCm');
  assert.equal(stats.sourceMethod, 'texture-copy');
  assert.equal(stats.captureMs, 2);
  assert.equal(stats.nativeCaptureMs, 1);
  assert.deepEqual(stats.temporal, { acceptedResults: 4 });
  for (const key of ['token', 'url', 'transportMs']) assert.equal(key in stats, false);
  await state.engine.destroy();
});

test('missing native stats remain unknown', async () => {
  const state = fixture();
  await state.engine.initialize({ video: state.video, settings });
  const stats = await state.engine.processFrame(state.video);
  for (const key of ['warmup', 'reused', 'modelPrecision', 'layout', 'inferenceMs']) assert.equal(key in stats, false);
  await state.engine.destroy();
});

test('failed initialization releases capture and client once', async () => {
  const state = fixture();
  state.client.start = async () => { throw Object.assign(new Error('offline'), { code: 'HOST_UNAVAILABLE' }); };
  await assert.rejects(state.engine.initialize({ video: state.video, settings }), { code: 'HOST_UNAVAILABLE' });
  assert.equal(state.calls.filter(row => row[0] === 'stop').length, 1);
  assert.equal(state.canvas.width, 1);
  assert.equal(state.engine.captureContext, null);
  await state.engine.destroy();
  assert.equal(state.calls.filter(row => row[0] === 'stop').length, 1);
});

test('destroy during presentation prevents late stats', async () => {
  const state = fixture();
  let release;
  state.presenter.presentFrame = () => new Promise(resolve => { release = resolve; });
  await state.engine.initialize({ video: state.video, settings, presenter: state.presenter });
  const processing = state.engine.processFrame(state.video);
  const rejection = assert.rejects(processing, error => {
    assert.equal(error.code, 'STALE_SESSION');
    assert.deepEqual(error.diagnostic, {
      staleOrigin: 'inline_engine', sessionTag: sessionTag(state.engine.sessionId),
      messageType: 'FRAME_IN', generation: 1, closed: true, sequence: 1,
    });
    return true;
  });
  await new Promise(resolve => setImmediate(resolve));
  const destroying = state.engine.destroy();
  release(true);
  await rejection;
  await destroying;
});

for (const synchronous of [false, true]) test(`destroy preserves stop failure and frees capture, synchronous=${synchronous}`, async () => {
  const state = fixture();
  await state.engine.initialize({ video: state.video, settings });
  const error = Object.assign(new Error('stop failed'), { code: 'HOST_UNAVAILABLE' });
  state.client.stop = synchronous ? () => { throw error; } : async () => { throw error; };
  const destroying = state.engine.destroy();
  assert.equal(state.engine.destroy(), destroying);
  await assert.rejects(destroying, { code: 'HOST_UNAVAILABLE' });
  assert.equal(state.canvas.width, 1);
  assert.equal(state.engine.captureContext, null);
});

test('inline operation backlog is bounded and destroy drains queued calls', async () => {
  const state = fixture();
  await state.engine.initialize({ video: state.video, settings });
  let release;
  state.client.processFrame = () => new Promise(resolve => { release = resolve; });
  const first = state.engine.processFrame(state.video);
  const queued = state.engine.processFrame(state.video);
  const queuedRejected = assert.rejects(queued, { code: 'STALE_SESSION' });
  const excess = assert.rejects(state.engine.processFrame(state.video), { code: 'NATIVE_SESSION_BUSY' });
  await new Promise(resolve => setImmediate(resolve));
  const destroying = state.engine.destroy();
  release({ width: 1, height: 1 });
  await assert.rejects(first, { code: 'STALE_SESSION' });
  await queuedRejected;
  await excess;
  await destroying;
  assert.equal(state.canvas.width, 1);
});

test('NativeInlineEngine checks direct capture, starts host, relays PNG and presents returned output', async () => {
  const state = fixture();
  await state.engine.initialize({ canvas: {}, video: state.video, settings, captureMode: 'direct', presenter: state.presenter });
  assert.equal(state.calls.find(row => row[0] === 'start')[1].sessionId, state.engine.sessionId);
  const stats = await state.engine.processFrame(state.video);
  assert.equal(state.calls.find(row => row[0] === 'start')[1].captureMode, 'direct-png');
  const frame = state.calls.find(row => row[0] === 'frame')[1];
  assert.deepEqual({ sequence: frame.sequence, mediaTime: frame.mediaTime, width: frame.width, height: frame.height,
    mime: frame.mime, dataUrl: frame.dataUrl }, { sequence: 1, mediaTime: 2, width: 1, height: 1, mime: 'image/png', dataUrl: png });
  const presented = state.calls.find(row => row[0] === 'present');
  assert.equal(presented[1].dataUrl, png);
  assert.equal(presented[2], true);
  assert.equal(stats.backend, 'rocm');
  assert.equal(stats.totalMs >= 0, true);
});

test('NativeInlineEngine reports CORS before opening native host', async () => {
  const state = fixture({ captureError: Object.assign(new Error('tainted'), { name: 'SecurityError' }) });
  await assert.rejects(state.engine.initialize({ canvas: {}, video: state.video, settings, captureMode: 'direct' }),
    { code: 'CROSS_ORIGIN_VIDEO' });
  assert.equal(state.calls.some(row => row[0] === 'start'), false);
});

test('NativeInlineEngine selects restricted capture label and reconfigures current host session', async () => {
  const state = fixture();
  await state.engine.initialize({ canvas: {}, video: state.video, settings, captureMode: 'tab', presenter: state.presenter });
  await state.engine.updateSettings({ ...settings, preset: 'quality' }, state.video, 2);
  assert.deepEqual(state.calls.find(row => row[0] === 'start')[1].captureMode, 'restricted-png');
  assert.equal(state.calls.find(row => row[0] === 'configure')[1].adaptiveLevel, 2);
});
