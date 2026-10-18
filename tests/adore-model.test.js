import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import protobuf from '../node_modules/onnxruntime-web/lib/onnxjs/ort-schema/protobuf/onnx.js';
import { getModelCandidates } from '../src/engine/model-config.js';

const hasAdoreModels = existsSync('models/2x_Adore_renarchi_webgpu.onnx')
  && existsSync('models/2x_Adore_renarchi_webgpu_fp16.onnx');

test('prepared Adore FP16 retains float32 GPU I/O and half precision weights', { skip: !hasAdoreModels && 'download Adore models first' }, async () => {
  const candidate = getModelCandidates(path => path, 'adore')[0];
  const bytes = await readFile(candidate.path);
  assert.equal(createHash('sha256').update(bytes).digest('hex'), candidate.sha256);
  const model = protobuf.onnx.ModelProto.decode(bytes);
  assert.equal(model.graph.input[0].type.tensorType.elemType, 1);
  assert.equal(model.graph.output[0].type.tensorType.elemType, 1);
  assert.ok(model.graph.initializer.some(t => t.dataType === 10));
  assert.ok(model.graph.node.some(n => n.opType === 'Cast'));
  const types = new Map(model.graph.valueInfo.map(v => [v.name, v.type.tensorType.elemType]));
  const reductions = model.graph.node.filter(n => n.opType === 'ReduceMean');
  assert.equal(reductions.length, 4);
  for (const reduction of reductions) {
    assert.equal(types.get(reduction.input[0]), 1, 'spatial reductions must accumulate in FP32');
    assert.equal(types.get(reduction.output[0]), 1);
  }
  for (const axis of [2, 3]) assert.notEqual(model.graph.input[0].type.tensorType.shape.dim[axis].dimParam,
    model.graph.output[0].type.tensorType.shape.dim[axis].dimParam);
});

test('prepared Adore FP32 changes only spatial dimension labels', { skip: !hasAdoreModels && 'download Adore models first' }, async () => {
  const candidate = getModelCandidates(path => path, 'adore')[1];
  const bytes = await readFile(candidate.path);
  assert.equal(createHash('sha256').update(bytes).digest('hex'), candidate.sha256);
  const prepared = protobuf.onnx.ModelProto.decode(bytes);
  const original = protobuf.onnx.ModelProto.decode(await readFile('models/2x_Adore_renarchi_fp32.onnx'));
  for (const group of ['input', 'output']) for (const axis of [2, 3]) {
    prepared.graph[group][0].type.tensorType.shape.dim[axis].dimParam = original.graph[group][0].type.tensorType.shape.dim[axis].dimParam;
  }
  assert.deepEqual(prepared, original);
});
