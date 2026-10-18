import test from 'node:test';
import assert from 'node:assert/strict';
import { getModelCandidates, supportsFloat32GpuIo } from '../src/engine/model-config.js';

test('selects AnimeJaNai without silently falling back to the heavier Adore', () => {
  const models = getModelCandidates((path) => path, 'animejanai');
  assert.equal(models.length, 2);
  assert.equal(models[0].precision, 'fp16');
  assert.equal(models[1].precision, 'fp32');
  assert.match(models[0].url, /SuperUltraCompact_webgpu_fp16.onnx$/);
  assert.ok(models.every(model => model.label === 'AnimeJaNai SuperUltraCompact'));
  assert.equal(models[0].label, 'AnimeJaNai SuperUltraCompact');
});

test('getModelCandidates prefers FP16 and falls back to FP32', () => {
  const candidates = getModelCandidates((path) => `chrome-extension://id/${path}`);
  assert.equal(candidates.length, 2);
  assert.equal(candidates[0].precision, 'fp16');
  assert.equal(candidates[1].precision, 'fp32');
  assert.match(candidates[0].url, /models\/2x_Adore_renarchi_webgpu_fp16\.onnx$/);
  assert.match(candidates[1].url, /models\/2x_Adore_renarchi_webgpu\.onnx$/);
});

test('model candidates retain verified prepared and release SHA-256 digests', () => {
  const [fp16, fp32] = getModelCandidates((path) => path);
  assert.equal(fp16.sha256, 'e503b357fe6d6ce34c3db9980cc2ed37dd1f23d4772b352c8a4f2befb151f2b5');
  assert.equal(fp32.sha256, 'b0a76dff7816816dee2c9a19f58bb1a81dd68178547f9518b64bc0257205749d');
});


test('supportsFloat32GpuIo accepts only float32 tensor I/O', () => {
  assert.equal(supportsFloat32GpuIo(
    { isTensor: true, type: 'float32', shape: [1, 3, 'h', 'w'] },
    { isTensor: true, type: 'float32', shape: [1, 3, 'oh', 'ow'] },
  ), true);
  assert.equal(supportsFloat32GpuIo(
    { isTensor: true, type: 'float16' },
    { isTensor: true, type: 'float16' },
  ), false);
  assert.equal(supportsFloat32GpuIo(undefined, undefined), false);
});

for (const [name, input, output] of [
  ['rank', [1, 3, 8], [1, 3, 16, 16]],
  ['channel', [1, 4, 8, 8], [1, 3, 16, 16]],
  ['batch', [2, 3, 8, 8], [2, 3, 16, 16]],
  ['scale', [1, 3, 8, 8], [1, 3, 8, 8]],
  ['same symbols', [1, 3, 'h', 'w'], [1, 3, 'h', 'w']],
  ['negative dimension', [1, 3, -1, 8], [1, 3, 16, 16]],
]) test(`GPU I/O rejects incompatible ${name}`, () => {
  const tensor = shape => ({ isTensor: true, type: 'float32', shape });
  assert.equal(supportsFloat32GpuIo(tensor(input), tensor(output)), false);
});
