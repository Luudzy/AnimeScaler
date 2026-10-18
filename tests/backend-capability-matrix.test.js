import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolveModelCompatibility } from '../src/core/local-models.js';

const backends = ['webgpu', 'rocm', 'tensorrt', 'opencl', 'd3d12'];
const routes = {
  'adore': ['rocm', 'tensorrt'],
  'animejanai': ['rocm', 'tensorrt'],
  'adore-lite': ['rocm', 'tensorrt'],
  'adore-lite-v3': ['rocm', 'tensorrt'],
  'fsr1-spatial': ['webgpu'],
  'asr-anime-spatial': ['webgpu'],
  'asr-anime-temporal': [],
  'zenkai-1-0': ['rocm', 'tensorrt'],
  'zenkai-realesr-clean5k': ['rocm'],
  'zenkai-animevideo-30k': ['rocm'],
  'hfa2k-cugan': ['rocm', 'tensorrt'],
  'realesr-anime': ['rocm', 'tensorrt'],
  'anime4k-s': ['rocm'],
  'anime4k-m': ['rocm'],
  'realesr-anime4k-s-360': ['rocm'],
  'span-mssim': ['rocm', 'tensorrt'],
  'anime4kcpp-acnet-f8b8-hdn': ['opencl'],
  'anime4kcpp-acnet-f8b4': ['opencl'],
  'anime4kcpp-arnet-f8b8': ['opencl'],
  'anime4kcpp-artcnn-c4f16': ['opencl'],
  'fsr3-native': ['d3d12'],
};
const webgpuPendingModels = new Set(['adore', 'animejanai', 'asr-anime-temporal']);

test('every public model has explicit compatibility for each backend', () => {
  const popup = readFileSync(new URL('../popup/popup.html', import.meta.url), 'utf8');
  const options = popup.match(/<select id="model"[^>]*>([\s\S]*?)<\/select>/)?.[1] ?? '';
  const publicIds = [...options.matchAll(/<option value="([^"]+)"/g)].map((match) => match[1]);
  assert.deepEqual(new Set(publicIds), new Set(Object.keys(routes)));

  for (const model of publicIds) {
    for (const backend of backends) {
      const result = resolveModelCompatibility(model, backend);
      const expectedRoute = routes[model].includes(backend);
      const expectedAvailable = expectedRoute;
      assert.equal(result.available, expectedAvailable, `${model}/${backend}`);
      assert.ok(result.status, `${model}/${backend} must have an explicit status`);
      if (!expectedAvailable) assert.ok(result.reason, `${model}/${backend} must explain the block`);
      else assert.equal(result.reason, null, `${model}/${backend}`);
    }
  }
});

test('WebGPU enables browser-qualified shader models and keeps other engines pending', () => {
  for (const model of ['fsr1-spatial', 'asr-anime-spatial']) {
    const result = resolveModelCompatibility(model, 'webgpu');
    assert.equal(result.available, true, model);
    assert.equal(result.status, 'route-available', model);
    assert.equal(result.reason, null, model);
  }

  for (const model of webgpuPendingModels) {
    const result = resolveModelCompatibility(model, 'webgpu');
    assert.equal(result.available, false, model);
    assert.equal(result.status, 'qualification-pending', model);
    assert.match(result.reason, new RegExp(model, 'i'));
  }

  for (const model of ['zenkai-1-0', 'hfa2k-cugan', 'anime4kcpp-acnet-f8b4', 'fsr3-native']) {
    const result = resolveModelCompatibility(model, 'webgpu');
    assert.equal(result.available, false, model);
    assert.equal(result.status, 'unsupported', model);
    assert.match(result.reason, /WebGPU.*executor|executor.*WebGPU/i);
  }
});

test('TensorRT excludes Torch-only Zenkai and ROCm-only Anime4K routes', () => {
  for (const model of ['zenkai-realesr-clean5k', 'zenkai-animevideo-30k', 'anime4k-s', 'anime4k-m', 'realesr-anime4k-s-360']) {
    const result = resolveModelCompatibility(model, 'tensorrt');
    assert.equal(result.available, false, model);
    assert.equal(result.status, 'unsupported', model);
    assert.match(result.reason, /ROCm/);
  }
});

test('pending WebGPU models explain that activation awaits qualification', () => {
  const result = resolveModelCompatibility('adore', 'webgpu');
  assert.equal(result.available, false);
  assert.match(result.reason, /WebGPU ainda não foi qualificado/i);
});

test('unknown model and backend fail closed with distinct reasons', () => {
  const unknownModel = resolveModelCompatibility('missing-model', 'rocm');
  assert.equal(unknownModel.available, false);
  assert.equal(unknownModel.status, 'unknown-model');

  const unknownBackend = resolveModelCompatibility('adore', 'metal');
  assert.equal(unknownBackend.available, false);
  assert.equal(unknownBackend.status, 'unknown-backend');
});
