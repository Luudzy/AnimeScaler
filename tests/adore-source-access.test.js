import test from 'node:test';
import assert from 'node:assert/strict';
import { AdoreEngine } from '../src/engine/adore-engine.js';

test('verifySource converts WebGPU cross-origin frame access into CROSS_ORIGIN_VIDEO before the render loop starts', () => {
  const engine = new AdoreEngine();
  engine.gpu = {
    uploadVideoFrame() {
      const error = new Error('tainted source');
      error.name = 'SecurityError';
      throw error;
    },
  };

  assert.throws(
    () => engine.verifySource({}),
    (error) => error.code === 'CROSS_ORIGIN_VIDEO',
  );
});
