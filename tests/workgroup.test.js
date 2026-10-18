import test from 'node:test';
import assert from 'node:assert/strict';
import { validateWorkgroup, workgroupShader } from '../src/engine/workgroup.js';
import { PREPROCESS_WGSL } from '../src/engine/shaders.js';
test('candidate workgroups obey device limits and specialize the shader', () => {
  assert.deepEqual(validateWorkgroup([16, 8], {maxComputeInvocationsPerWorkgroup: 128}), [16, 8]);
  assert.throws(() => validateWorkgroup([16, 16], {maxComputeInvocationsPerWorkgroup: 128}));
  assert.throws(() => validateWorkgroup([7, 7]));
  assert.match(workgroupShader(PREPROCESS_WGSL, [16, 8]), /@workgroup_size\(16, 8\)/);
});
