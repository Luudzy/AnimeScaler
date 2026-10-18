import test from 'node:test';
import assert from 'node:assert/strict';
import { resolveBackendDescriptor } from '../src/core/backend-descriptor.js';

test('qualified spatial models retain and execute their WebGPU route', () => {
  for (const model of ['fsr1-spatial', 'asr-anime-spatial']) {
    const descriptor=resolveBackendDescriptor({ model, backend: 'webgpu' });
    assert.equal(descriptor.id,'webgpu');
    assert.equal(descriptor.modelId,model);
    assert.equal(descriptor.available,undefined);
    assert.equal(descriptor.reason,undefined);
  }
});

test('ASR temporal stays blocked until its own qualification', () => {
  const descriptor=resolveBackendDescriptor({model:'asr-anime-temporal',backend:'webgpu'});
  assert.equal(descriptor.available,false);
  assert.match(descriptor.reason,/WebGPU.*qualificado/i);
});

test('native-only model/backend pairs retain selection and expose mismatch', () => {
  assert.equal(resolveBackendDescriptor({ model: 'fsr3-native', backend: 'd3d12' }).id, 'd3d12');
  assert.notEqual(resolveBackendDescriptor({ model: 'anime4k-s', backend: 'rocm' }).available, false);
  assert.equal(resolveBackendDescriptor({ model: 'anime4k-s', backend: 'tensorrt' }).available, false);
});

test('native-capable models respect selected ROCm or TensorRT backend', () => {
  for (const id of ['rocm', 'tensorrt']) {
    assert.deepEqual(resolveBackendDescriptor({ model: 'zenkai-1-0', backend: id }), {
      id, modelId: 'zenkai-1-0', executionHost: 'extension-offscreen', outputFormat: 'rgba8', requiresLocalService: true,
    });
  }
});

test('OpenCL descriptor is limited to the standalone Anime4KCPP service',()=>{
  const descriptor=resolveBackendDescriptor({model:'anime4kcpp-acnet-f8b4',backend:'opencl'});
  assert.deepEqual(descriptor,{id:'opencl',modelId:'anime4kcpp-acnet-f8b4',executionHost:'extension-offscreen',outputFormat:'rgba8',requiresLocalService:true});
  assert.equal(resolveBackendDescriptor({model:'adore',backend:'opencl'}).available,false);
});

test('unsupported D3D12 model combination keeps request and explains block', () => {
  const descriptor=resolveBackendDescriptor({ model: 'adore', backend: 'd3d12' });
  assert.equal(descriptor.id,'d3d12');
  assert.equal(descriptor.modelId,'adore');
  assert.equal(descriptor.available,false);
  assert.match(descriptor.reason,/exige ROCm local ou TensorRT local/i);
});

test('descriptor exposes matrix reason without replacing model or backend',()=>{
  const descriptor=resolveBackendDescriptor({model:'zenkai-animevideo-30k',backend:'webgpu'});
  assert.equal(descriptor.modelId,'zenkai-animevideo-30k');
  assert.equal(descriptor.id,'webgpu');
  assert.equal(descriptor.available,false);
  assert.match(descriptor.reason,/exige ROCm local/i);
  assert.equal(descriptor.releaseStatus,'experimental');
});
