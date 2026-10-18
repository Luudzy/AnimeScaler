import test from 'node:test';
import assert from 'node:assert/strict';

test('temporal reuse excludes changed cells and their neighbors', async () => {
  const { stableMask } = await import('../src/engine/temporal-mask.js');
  const before = new Uint8ClampedArray(9*9*4).fill(100), after = before.slice();
  after[(4*9+4)*4] = 255;
  const result = stableMask(before,after,9,9);
  assert.equal(result[(4*9+4)*4+3],0);
  assert.equal(result[(3*9+3)*4+3],0);
  assert.equal(result[(0*9+0)*4+3],255);
});

test('a scene cut discards the previous AI image completely', async () => {
  const { stableMask } = await import('../src/engine/temporal-mask.js');
  const before = new Uint8ClampedArray(8*8*4), after = new Uint8ClampedArray(8*8*4).fill(255);
  assert.equal(stableMask(before,after,8,8),null);
  assert.equal(stableMask(null,after,8,8),null);
});
