import test from 'node:test';
import assert from 'node:assert/strict';
import { AdoreEngine } from '../src/engine/adore-engine.js';

test('reconfigure uses manual capture crop dimensions instead of the full tab', async () => {
  let configured;
  const engine = new AdoreEngine();
  engine.settings = { preset: 'performance' };
  engine.ort = {
    Tensor: { fromGpuBuffer(_buffer, opts) { return opts; } },
  };
  engine.gpu = {
    configure(value) { configured = value; },
    getInputBuffer() { return {}; },
    getOutputBuffer() { return {}; },
    getInputShape() { return [1, 3, 360, 640]; },
    getOutputShape() { return [1, 3, 720, 1280]; },
  };
  const video = { videoWidth: 1920, videoHeight: 1080 };
  const crop = { x: 100, y: 50, width: 1280, height: 720 };

  const result = await engine.reconfigure(video, 0, crop);

  assert.equal(configured.sourceWidth, 1280);
  assert.equal(configured.sourceHeight, 720);
  assert.equal(result.sourceWidth, 1280);
  assert.equal(result.sourceHeight, 720);
  assert.deepEqual(engine.sourceRect, crop);
});

test('needsReconfigure reacts when manual crop size changes but not when only its origin moves', () => {
  const engine = new AdoreEngine();
  engine.currentSize = { sourceWidth: 1280, sourceHeight: 720, adaptiveLevel: 0 };
  const video = { videoWidth: 1920, videoHeight: 1080 };
  assert.equal(engine.needsReconfigure(video, 0, { x: 10, y: 20, width: 1280, height: 720 }), false);
  assert.equal(engine.needsReconfigure(video, 0, { x: 10, y: 20, width: 1000, height: 600 }), true);
});
