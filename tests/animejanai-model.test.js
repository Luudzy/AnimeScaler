import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import protobuf from '../node_modules/onnxruntime-web/lib/onnxjs/ort-schema/protobuf/onnx.js';
import { getModelCandidates } from '../src/engine/model-config.js';

const hasAnimeJaNaiModels = existsSync('models/2x_AnimeJaNai_HD_V3_SuperUltraCompact_webgpu.onnx')
  && existsSync('models/2x_AnimeJaNai_HD_V3_SuperUltraCompact_webgpu_fp16.onnx');

test('selected AnimeJaNai model gives 2x output its own spatial dimension symbols', { skip: !hasAnimeJaNaiModels && 'download and prepare AnimeJaNai models first' }, async () => {
  const candidate = getModelCandidates(path => path, 'animejanai')[0];
  const model = protobuf.onnx.ModelProto.decode(await readFile(candidate.path));
  const input = model.graph.input[0].type.tensorType.shape.dim;
  const output = model.graph.output[0].type.tensorType.shape.dim;
  assert.equal(input[0].dimParam, output[0].dimParam);
  for (const axis of [2, 3]) {
    assert.ok(output[axis].dimParam);
    assert.notEqual(input[axis].dimParam, output[axis].dimParam,
      '2x output must not claim the same spatial size as the input');
  }
});

test('prepared model preserves all operations and weights from the verified original', { skip: !hasAnimeJaNaiModels && 'download and prepare AnimeJaNai models first' }, async () => {
  const candidate = getModelCandidates(path => path, 'animejanai').find(model => model.precision === 'fp32');
  const bytes = await readFile(candidate.path);
  assert.equal(createHash('sha256').update(bytes).digest('hex'), candidate.sha256);
  const prepared = protobuf.onnx.ModelProto.decode(bytes);
  const original = protobuf.onnx.ModelProto.decode(await readFile(
    'models/2x_AnimeJaNai_HD_V3_SuperUltraCompact.onnx'));
  prepared.graph.output[0].type.tensorType.shape.dim[2].dimParam = 'height';
  prepared.graph.output[0].type.tensorType.shape.dim[3].dimParam = 'width';
  assert.deepEqual(prepared, original);
});

test('preferred AnimeJaNai uses half precision weights with float32 GPU I/O', { skip: !hasAnimeJaNaiModels && 'download and prepare AnimeJaNai models first' }, async () => {
  const candidate = getModelCandidates(path => path, 'animejanai')[0];
  const bytes = await readFile(candidate.path);
  const model = protobuf.onnx.ModelProto.decode(bytes);
  assert.equal(model.graph.input[0].type.tensorType.elemType, 1);
  assert.equal(model.graph.output[0].type.tensorType.elemType, 1);
  const weights = model.graph.initializer.filter(value => value.dims.length === 4);
  assert.ok(weights.length > 0);
  assert.ok(weights.every(value => value.dataType === 10), 'convolution weights must be FP16');
  assert.equal(createHash('sha256').update(bytes).digest('hex'), candidate.sha256);
});
