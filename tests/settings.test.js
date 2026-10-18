import test from 'node:test';
import assert from 'node:assert/strict';
import { normalizeSettings, resolveInputSize, resolveFsrInputSize } from '../src/core/settings.js';
import { isFsrSpatialModel } from '../src/core/local-models.js';

test('720p and 1080p targets select 360p and 720p model inputs', () => {
  assert.equal(normalizeSettings({preset:'720p'}).preset, '720p');
  assert.deepEqual(resolveInputSize(1920,1080,'720p'), {width:640,height:360});
  assert.deepEqual(resolveInputSize(1920,1080,'1080p'), {width:960,height:540});
  assert.deepEqual(resolveInputSize(1280,720,'1080p-fast'), {width:960,height:540});
  assert.equal(normalizeSettings({model:'adore-lite'}).model,'adore-lite');
});

test('normalizeSettings returns safe defaults', () => {
  assert.deepEqual(normalizeSettings(), {
    backend: 'webgpu',
    exclusiveViewer: false,
    fluidVideo: false,
    enabled: false,
    preset: 'performance',
    model: 'animejanai',
    maxFps: 24,
    adaptive: true,
    showHud: true,
    spatialDenoise: 'balanced',
    fsrQuality: 'native-aa',
  });
});

test('normalizeSettings rejects unknown values and preserves valid values', () => {
  assert.deepEqual(
    normalizeSettings({ enabled: 1, preset: 'potato', adaptive: false, showHud: false }),
    { backend: 'webgpu', exclusiveViewer:false, fluidVideo:false,
    enabled: false, preset: 'performance', model: 'animejanai', maxFps: 24, adaptive: false, showHud: false, spatialDenoise: 'balanced', fsrQuality: 'native-aa' },
  );
  assert.equal(normalizeSettings({ preset: 'quality' }).preset, 'quality');
});

test('spatial noise reduction preserves supported strengths and rejects malformed values', () => {
  for (const spatialDenoise of ['off', 'balanced', 'strong']) {
    assert.equal(normalizeSettings({ spatialDenoise }).spatialDenoise, spatialDenoise);
  }
  assert.equal(normalizeSettings({ spatialDenoise: '100%' }).spatialDenoise, 'balanced');
});

test('FSR quality selects internal dimensions independently of output and never invents source pixels', () => {
  assert.deepEqual(resolveFsrInputSize(3840,2160,1920,1080,'performance'), {width:960,height:540});
  assert.deepEqual(resolveFsrInputSize(3840,2160,1920,1080,'quality'), {width:1280,height:720});
  assert.deepEqual(resolveFsrInputSize(1280,720,3840,2160,'quality'), {width:1280,height:720});
  assert.deepEqual(resolveFsrInputSize(1920,1080,3840,2160,'native-aa'), {width:1920,height:1080});
  assert.equal(normalizeSettings({fsrQuality:'performance'}).fsrQuality,'performance');
  assert.equal(normalizeSettings({fsrQuality:'garbage'}).fsrQuality,'native-aa');
});

test('model and frame cap persist and invalid values use safe defaults', () => {
  const saved = normalizeSettings({ model: 'adore', maxFps: 12 });
  assert.equal(normalizeSettings(JSON.parse(JSON.stringify(saved))).model, 'adore');
  assert.equal(saved.maxFps, 12);
  assert.equal(normalizeSettings({ model: 'unknown', maxFps: 1000 }).model, 'unknown');
  assert.equal(normalizeSettings({ maxFps: 1000 }).maxFps, 24);
});

test('FSR spatial remains a WebGPU stream mode and is selectable', () => {
  const fsr = normalizeSettings({ model: 'fsr1-spatial', backend: 'webgpu' });
  assert.equal(fsr.model, 'fsr1-spatial');
  assert.equal(fsr.backend, 'webgpu');
  assert.equal(isFsrSpatialModel(fsr.model), true);
});

test('legacy model and explicit backend intent survive normalization and roundtrip',()=>{
  const saved=normalizeSettings({model:'adore-lite-v2',backend:'webgpu'});
  assert.equal(saved.model,'adore-lite-v2');
  assert.equal(saved.backend,'webgpu');
  assert.deepEqual(normalizeSettings(JSON.parse(JSON.stringify(saved))),saved);
});

test('resolveInputSize preserves aspect ratio and returns even dimensions', () => {
  assert.deepEqual(resolveInputSize(1920, 1080, 'balanced', 0), { width: 960, height: 540 });
  assert.deepEqual(resolveInputSize(1280, 720, 'quality', 0), { width: 1280, height: 720 });
  const size = resolveInputSize(853, 480, 'balanced', 0);
  assert.equal(size.width % 2, 0);
  assert.equal(size.height % 2, 0);
  assert.ok(Math.abs(size.width / size.height - 853 / 480) < 0.01);
});

test('resolveInputSize steps down progressively without exceeding the source size', () => {
  assert.deepEqual(resolveInputSize(1920, 1080, 'quality', 0), { width: 1280, height: 720 });
  assert.deepEqual(resolveInputSize(1920, 1080, 'quality', 1), { width: 960, height: 540 });
  assert.deepEqual(resolveInputSize(1920, 1080, 'quality', 2), { width: 640, height: 360 });
  assert.deepEqual(resolveInputSize(640, 360, 'native', 0), { width: 640, height: 360 });
});

test('Adore input dimensions stay divisible by four at every adaptive step', () => {
  assert.deepEqual(resolveInputSize(1920, 1080, 'native', 4), { width: 480, height: 268 });
  for (const [w, h] of [[1920, 1080], [853, 480], [1080, 1920], [854, 478]]) {
    for (const preset of ['native', 'quality', 'balanced', 'performance']) {
      for (let level = 0; level < 5; level++) {
        const size = resolveInputSize(w, h, preset, level);
        assert.equal(size.width % 4, 0);
        assert.equal(size.height % 4, 0);
      }
    }
  }
});

test('performance starts small and adaptive can reach a low GPU cost floor', () => {
  assert.deepEqual(resolveInputSize(1920, 1080, 'performance', 0), { width: 320, height: 180 });
  assert.deepEqual(resolveInputSize(1920, 1080, 'performance', 2), { width: 160, height: 88 });
  assert.deepEqual(resolveInputSize(1920, 1080, 'native', 7), { width: 160, height: 88 });
});
