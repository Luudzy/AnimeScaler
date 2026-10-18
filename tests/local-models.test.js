import test from 'node:test';
import assert from 'node:assert/strict';
import {ADDED_MODELS,requiresNative,isFsrSpatialModel,resolveModelCompatibility} from '../src/core/local-models.js';
import {normalizeSettings} from '../src/core/settings.js';
import {getModelCandidates} from '../src/engine/model-config.js';
import {readFileSync} from 'node:fs';

test('Zenkai 1.0 is a selectable native x2 model without changing the default',()=>{
  assert.equal(ADDED_MODELS['zenkai-1-0']?.label,'Zenkai 1.0');
  assert.equal(ADDED_MODELS['zenkai-1-0'].scale,2);
  assert.equal(normalizeSettings({model:'zenkai-1-0',backend:'rocm'}).model,'zenkai-1-0');
  assert.equal(normalizeSettings().model,'animejanai');
  assert.equal(requiresNative('zenkai-1-0'),true);
  assert.throws(()=>getModelCandidates(undefined,'zenkai-1-0'),/ROCm ou TensorRT/);
  const popup=readFileSync(new URL('../popup/popup.html',import.meta.url),'utf8');
  assert.match(popup,/<option value="zenkai-1-0">Zenkai 1\.0.*experimental.*<\/option>/);
  assert.equal(popup.match(/<select id="model"[^>]*>\s*<option value="([^"]+)"/)[1],'animejanai');
});
test('user models retain selection and cannot silently run Adore on WebGPU',()=>{
  for(const [id,model] of Object.entries(ADDED_MODELS)) {
    assert.equal(normalizeSettings({model:id}).model,id);
    assert.equal(requiresNative(id),true);
    assert.ok((id==='fsr3-native' ? [1] : [2,4]).includes(model.scale));
    assert.throws(()=>getModelCandidates(undefined,id),id==='fsr3-native' ? /DirectX 12/ : model.openclOnly ? /OpenCL/ : model.rocmOnly ? /apenas ROCm/ : /ROCm ou TensorRT/);
  }
});

test('Anime4K experiments explicitly use ROCm and preserve existing defaults',async()=>{
  const {isRocmOnly}=await import('../src/core/local-models.js');
  for(const id of ['anime4k-s','anime4k-m','realesr-anime4k-s-360']){
    assert.equal(isRocmOnly(id),true);
    assert.equal(normalizeSettings({model:id,backend:'tensorrt'}).backend,'tensorrt');
    assert.equal(normalizeSettings({model:id}).backend,'rocm');
    assert.equal(normalizeSettings({model:id,backend:'rocm'}).model,id);
    assert.match(readFileSync(new URL('../popup/popup.html',import.meta.url),'utf8'),new RegExp('option value="'+id+'"'));
  }
  assert.equal(normalizeSettings({model:'realesr-anime',backend:'tensorrt'}).backend,'tensorrt');
});

test('FSR spatial is exposed as a GPU stream option without native service requirements',()=>{
  const popup=readFileSync(new URL('../popup/popup.html',import.meta.url),'utf8');
  assert.equal(isFsrSpatialModel('fsr1-spatial'),true);
  assert.equal(requiresNative('fsr1-spatial'),false);
  assert.ok(popup.includes('value="fsr1-spatial"'));
});

test('compatibility matrix keeps requested model/backend and gives WebGPU block reason',()=>{
  const blocked=resolveModelCompatibility('adore','webgpu');
  assert.equal(blocked.available,false);
  assert.match(blocked.reason,/WebGPU.*qualificado/i);
  assert.equal(resolveModelCompatibility('zenkai-1-0','rocm').available,true);
  assert.equal(resolveModelCompatibility('zenkai-1-0','tensorrt').available,true);
  assert.equal(resolveModelCompatibility('anime4k-s','tensorrt').available,false);
});

test('Anime4KCPP models select standalone OpenCL without making neural models available there',()=>{
  for(const id of Object.keys(ADDED_MODELS).filter(id=>id.startsWith('anime4kcpp-'))){
    assert.equal(normalizeSettings({model:id}).backend,'opencl');
    assert.equal(normalizeSettings({model:id,backend:'opencl'}).backend,'opencl');
    assert.equal(resolveModelCompatibility(id,'opencl').available,true);
    assert.equal(resolveModelCompatibility('adore','opencl').available,false);
  }
  const popup=readFileSync(new URL('../popup/popup.html',import.meta.url),'utf8');
  assert.match(popup,/<option value="opencl">Anime4KCPP local · OpenCL<\/option>/);
});

test('30k and FSR3 keep experimental status in catalog metadata',()=>{
  assert.equal(ADDED_MODELS['zenkai-animevideo-30k'].releaseStatus,'experimental');
  assert.equal(ADDED_MODELS['fsr3-native'].releaseStatus,'experimental');
});

test('popup catalog entries resolve in the shared registry and expose linked live status',()=>{
  const popup=readFileSync(new URL('../popup/popup.html',import.meta.url),'utf8');
  const modelOptions=popup.match(/<select id="model"[^>]*>([\s\S]*?)<\/select>/)[1];
  const ids=[...modelOptions.matchAll(/<option value="([^"]+)"/g)].map(match=>match[1]);
  for(const id of ids) {
    const compatibility=['webgpu','rocm','tensorrt','opencl','d3d12'].map(backend=>resolveModelCompatibility(id,backend));
    assert.ok(compatibility.some(result=>!result.reason?.includes('não consta no catálogo')),`missing registry entry: ${id}`);
  }
  assert.match(popup,/<select id="model"[^>]*aria-describedby="modelNote"/);
  assert.match(popup,/<select id="backend"[^>]*aria-describedby="modelNote"/);
  assert.match(popup,/<small id="modelNote" role="status" aria-live="polite"><\/small>/);
});
