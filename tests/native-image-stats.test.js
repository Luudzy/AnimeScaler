import test from 'node:test';
import assert from 'node:assert/strict';
import { NativeEngine } from '../src/engine/native-engine.js';

const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII=', 'base64');
function fixture(t, timings) {
  const engine = new NativeEngine();
  engine.settings = { model: 'adore', fluidVideo: true };
  engine.nativeInfo = { backend: 'rocm', precision: 'fp16' };
  engine.currentSize = { sourceWidth: 1, sourceHeight: 1, inputWidth: 1, inputHeight: 1, outputWidth: 1, outputHeight: 1, adaptiveLevel: 0 };
  engine.worker = { postMessage(message) {
    queueMicrotask(() => {
      const request = engine.requests.get(message.id);
      engine.requests.delete(message.id); clearTimeout(request.timer);
      request.resolve({ ...timings, sequence: 7, mediaTime: 2.5, width: 1, height: 1, mime: 'image/png', pngBytes: Uint8Array.from(png).buffer });
    });
  }, terminate() {} };
  t.after(() => engine.destroy());
  return { engine, bitmap: { width: 1, height: 1, close() {} } };
}

test('PNG output carries actual worker measurements and negotiated precision', async t => {
  const { engine, bitmap } = fixture(t, { inferenceMs: 11, captureMs: 2, transportMs: 13, workerMs: 17,
    renderMs: 1, warmup: true, reused: true, responseBytes: 24, compositor: 'webgpu',
    temporal: { acceptedResults: 3, aiCoverage: .4, historyAgeMs: 9, unchangedSource: false } });
  const output = await engine.processImage(bitmap, { sequence: 7, mediaTime: 2.5 });
  assert.equal(output.mime, 'image/png');
  assert.equal(output.dataUrl, `data:image/png;base64,${png.toString('base64')}`);
  assert.equal(output.stats?.inferenceMs, 11);
  assert.equal(output.stats.captureMs, 2);
  assert.equal(output.stats.workerMs, 17);
  assert.equal(output.stats.warmup, true);
  assert.equal(output.stats.reused, true);
  assert.equal(output.stats.backend, 'rocm');
  assert.equal(output.stats.modelPrecision, 'fp16');
  assert.deepEqual(output.stats.temporal, { acceptedResults: 3, aiCoverage: .4, historyAgeMs: 9, unchangedSource: false });
});

test('PNG stats exclude credentials, payloads, invalid measurements and arbitrary temporal text', async t => {
  const { engine, bitmap } = fixture(t, { inferenceMs: 5, captureMs: -1, workerMs: Infinity, warmup: 'yes',
    token: 'secret', url: 'http://private', compositor: 'http://private', responseBytes: NaN,
    temporal: { aiCoverage: .2, historyAgeMs: NaN, token: 'secret', arbitrary: 'http://private' } });
  const output = await engine.processImage(bitmap, { sequence: 7, mediaTime: 2.5 });
  assert.equal(output.stats?.inferenceMs, 5);
  for (const key of ['captureMs', 'workerMs', 'warmup', 'token', 'url', 'pngBytes', 'compositor', 'responseBytes']) {
    assert.equal(Object.hasOwn(output.stats, key), false, key);
  }
  assert.deepEqual(output.stats.temporal, { aiCoverage: .2 });
  assert.equal(JSON.stringify(output.stats).includes('secret'), false);
});

test('missing worker telemetry preserves legacy PNG contract without fabricated measurements', async t => {
  const { engine, bitmap } = fixture(t, {});
  const output = await engine.processImage(bitmap, { sequence: 7, mediaTime: 2.5 });
  assert.equal(Object.hasOwn(output, 'stats'), false);
});

test('inconsistent worker timing does not fabricate zero outside-worker time', async t => {
  const { engine, bitmap } = fixture(t, { inferenceMs: 4, workerMs: 100000 });
  const output = await engine.processImage(bitmap, { sequence: 7, mediaTime: 2.5 });
  assert.ok(Number.isFinite(output.stats.workerRoundTripMs));
  assert.equal(Object.hasOwn(output.stats, 'workerOverheadMs'), false);
});
