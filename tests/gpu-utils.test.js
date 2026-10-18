import test from 'node:test';
import assert from 'node:assert/strict';
import { alignTo, nchwFloatBytes, outputShape2x } from '../src/engine/gpu-utils.js';

test('alignTo rounds byte sizes up to alignment', () => {
  assert.equal(alignTo(1, 16), 16);
  assert.equal(alignTo(16, 16), 16);
  assert.equal(alignTo(17, 16), 32);
});

test('nchwFloatBytes returns aligned float32 RGB tensor storage', () => {
  assert.equal(nchwFloatBytes(2, 2), 48);
  assert.equal(nchwFloatBytes(3, 3, 16), 112);
});

test('outputShape2x describes a 2x NCHW RGB output', () => {
  assert.deepEqual(outputShape2x(960, 540), [1, 3, 1080, 1920]);
});
