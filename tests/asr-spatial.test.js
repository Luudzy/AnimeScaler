import test from 'node:test';
import assert from 'node:assert/strict';
import { ASR_ANIME_WGSL } from '../src/engine/fsr-shaders.js';
import { createProcessingEngine, requiresEngineRecreation } from '../src/engine/engine-factory.js';
import { FsrEngine } from '../src/engine/fsr-engine.js';
import { isSpatialUpscalerModel } from '../src/core/local-models.js';
import { normalizeSettings } from '../src/core/settings.js';

test('ASR is a WebGPU-only anime spatial upscaler', () => {
  const settings = normalizeSettings({ model: 'asr-anime-spatial', backend: 'webgpu' });
  assert.equal(isSpatialUpscalerModel(settings.model), true);
  assert.equal(settings.backend, 'webgpu');
  assert.ok(createProcessingEngine(settings) instanceof FsrEngine);
});

test('ASR gets its own refinement shader and reinitializes on FSR switch', () => {
  assert.match(ASR_ANIME_WGSL, /flatness/);
  assert.match(ASR_ANIME_WGSL, /lineStrength/);
  assert.equal(requiresEngineRecreation({ model: 'fsr1-spatial' }, { model: 'asr-anime-spatial' }), true);
});

test('ASR temporal is a separate opt-in WebGPU mode and does not replace spatial defaults', () => {
  const settings=normalizeSettings({model:'asr-anime-temporal',backend:'webgpu'});
  assert.equal(settings.model,'asr-anime-temporal');
  assert.equal(settings.backend,'webgpu');
  assert.throws(()=>createProcessingEngine(settings),{code:'MODEL_BACKEND_UNAVAILABLE'});
  assert.equal(requiresEngineRecreation({model:'asr-anime-spatial'},settings),true);
  assert.equal(normalizeSettings({model:'asr-anime-spatial'}).model,'asr-anime-spatial');
});
