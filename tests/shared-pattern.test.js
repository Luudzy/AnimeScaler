import test from 'node:test';
import assert from 'node:assert/strict';
import { sharedPattern } from '../diagnostics/shared-pattern.js';
test('browser pattern packs identical RGB values into RGBA and NCHW', () => {
  const {rgba, nchw} = sharedPattern(8, 4);
  for (let i = 0; i < 32; i++) {
    assert.equal(rgba[i * 4 + 3], 255);
    for (let c = 0; c < 3; c++) assert.equal(nchw[c * 32 + i], Math.fround(rgba[i * 4 + c] / 255));
  }
});
