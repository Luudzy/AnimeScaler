import test from 'node:test';
import assert from 'node:assert/strict';
import { AdoreEngine } from '../src/engine/adore-engine.js';

for (const model of ['animejanai', 'adore']) {
test(`${model} requests NHWC internally and retries NCHW if session creation rejects it`, async () => {
  const engine = new AdoreEngine(), layouts = [];
  engine.settings = {model};
  engine.ort = {
    env: {webgpu: {device: {features: new Set(['shader-f16'])}}},
    InferenceSession: {create: async (_, options) => {
      const layout = options.executionProviders[0].preferredLayout; layouts.push(layout);
      if (layout === 'NHWC') throw new Error('layout unsupported');
      return {inputNames: ['input'], outputNames: ['output'], inputMetadata: [{isTensor: true, type: 'float32', shape: [1,3,'h','w']}],
        outputMetadata: [{isTensor: true, type: 'float32', shape: [1,3,'oh','ow']}], release: async () => {}};
    }},
  };
  await engine.createSession();
  assert.deepEqual(layouts, ['NHWC', 'NCHW']);
  assert.equal(engine.model.layout, 'NCHW');
});
}

for (const model of ['animejanai', 'adore']) for (const supported of [true, false]) {
  test(`${model} selects ${supported ? 'FP16' : 'FP32'} when shader-f16 is ${supported ? 'available' : 'unavailable'}`, async () => {
    const engine = new AdoreEngine();
    engine.settings = {model};
    let released = 0;
    engine.ort = {
      env: {webgpu: {device: {features: new Set(supported ? ['shader-f16'] : [])}}},
      InferenceSession: {create: async () => ({
        inputNames: ['input'], outputNames: ['output'],
        inputMetadata: [{isTensor: true, type: 'float32', shape: [1,3,'h','w']}],
        outputMetadata: [{isTensor: true, type: 'float32', shape: [1,3,'oh','ow']}],
        release: async () => { released++; },
      })},
    };
    await engine.createSession();
    assert.equal(engine.model.precision, supported ? 'fp16' : 'fp32');
    assert.equal(released, 0, 'unsupported FP16 is skipped before session allocation');
  });
}
