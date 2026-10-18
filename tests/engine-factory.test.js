import test from 'node:test';
import assert from 'node:assert/strict';
import { createProcessingEngine, requiresEngineRecreation } from '../src/engine/engine-factory.js';
import { FsrEngine } from '../src/engine/fsr-engine.js';
import { AdoreEngine } from '../src/engine/adore-engine.js';
import { NativeInlineEngine } from '../src/engine/native-inline-engine.js';

test('engine factory routes qualified FSR to WebGPU and blocks unqualified models', () => {
  assert.ok(createProcessingEngine({model:'fsr1-spatial'}) instanceof FsrEngine);
  for(const model of ['animejanai']) {
    assert.throws(()=>createProcessingEngine({ model }),{code:'MODEL_BACKEND_UNAVAILABLE'});
  }
});

test('switching between AI and FSR recreates the processing engine', () => {
  assert.equal(requiresEngineRecreation({ model: 'animejanai' }, { model: 'fsr1-spatial' }), true);
  assert.equal(requiresEngineRecreation({ model: 'fsr1-spatial' }, { model: 'adore' }), true);
  assert.equal(requiresEngineRecreation({ model: 'adore' }, { model: 'animejanai' }), false);
});

test('factory routes ROCm, TensorRT, OpenCL and FSR3 to the inline native engine', () => {
  assert.ok(createProcessingEngine({ model: 'zenkai-1-0', backend: 'rocm' }) instanceof NativeInlineEngine);
  assert.ok(createProcessingEngine({ model: 'zenkai-1-0', backend: 'tensorrt' }) instanceof NativeInlineEngine);
  assert.ok(createProcessingEngine({ model: 'anime4kcpp-acnet-f8b4', backend: 'opencl' }) instanceof NativeInlineEngine);
  assert.ok(createProcessingEngine({ model: 'fsr3-native', backend: 'd3d12' }) instanceof NativeInlineEngine);
});

test('native backend and model changes reuse adapter for host reconfiguration', () => {
  assert.equal(requiresEngineRecreation({ model: 'zenkai-1-0', backend: 'rocm' }, { model: 'zenkai-1-0', backend: 'tensorrt' }), false);
  assert.equal(requiresEngineRecreation({ model: 'zenkai-1-0', backend: 'rocm' }, { model: 'adore', backend: 'rocm' }), false);
  assert.equal(requiresEngineRecreation({ model: 'adore', backend: 'webgpu' }, { model: 'adore', backend: 'rocm' }), true);
});

test('factory blocks unqualified WebGPU pair with the matrix reason',()=>{
  assert.throws(()=>createProcessingEngine({model:'animejanai',backend:'webgpu'}),error=>
    error.code==='MODEL_BACKEND_UNAVAILABLE' && /WebGPU.*qualificado/i.test(error.message));
});
