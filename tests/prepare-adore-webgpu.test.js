import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { prepareAdoreWebgpu } from '../scripts/prepare-adore-webgpu.mjs';

const digest = (value) => createHash('sha256').update(value).digest('hex');
const outputs = { fp32: 'models/adore-webgpu.onnx', fp16: 'models/adore-webgpu-fp16.onnx' };

async function withRoot(run) {
  const root = await mkdtemp(join(tmpdir(), 'animescaler-adore-setup-'));
  await mkdir(join(root, 'models'));
  try { await run(root); } finally { await rm(root, { recursive: true, force: true }); }
}

test('prepareAdoreWebgpu preserves already verified models and skips conversion', async () => {
  await withRoot(async (root) => {
    await writeFile(join(root, outputs.fp32), 'good fp32');
    await writeFile(join(root, outputs.fp16), 'good fp16');
    let calls = 0;
    const result = await prepareAdoreWebgpu({
      root,
      outputs,
      expectedHashes: { fp32: digest('good fp32'), fp16: digest('good fp16') },
      generate: async () => { calls += 1; },
    });
    assert.equal(result.prepared, false);
    assert.equal(calls, 0);
  });
});

test('prepareAdoreWebgpu refuses to overwrite an existing hash-mismatched model', async () => {
  await withRoot(async (root) => {
    const existing = Buffer.from('user model');
    await writeFile(join(root, outputs.fp32), existing);
    let calls = 0;
    await assert.rejects(prepareAdoreWebgpu({
      root,
      outputs,
      expectedHashes: { fp32: digest('expected fp32'), fp16: digest('expected fp16') },
      generate: async () => { calls += 1; },
    }), /Existing Adore model hash mismatch/);
    assert.deepEqual(await readFile(join(root, outputs.fp32)), existing);
    assert.equal(calls, 0);
  });
});

test('prepareAdoreWebgpu verifies generated files before reporting ready', async () => {
  await withRoot(async (root) => {
    const expectedHashes = { fp32: digest('prepared fp32'), fp16: digest('prepared fp16') };
    const result = await prepareAdoreWebgpu({
      root,
      outputs,
      expectedHashes,
      generate: async () => {
        await writeFile(join(root, outputs.fp32), 'prepared fp32');
        await writeFile(join(root, outputs.fp16), 'prepared fp16');
      },
    });
    assert.equal(result.prepared, true);
  });
});

test('setup:models runs the Adore WebGPU preparation after downloading source weights', async () => {
  const packageJson = JSON.parse(await readFile(new URL('../package.json', import.meta.url), 'utf8'));
  assert.match(packageJson.scripts['setup:models'], /setup-models\.mjs/);
  assert.match(packageJson.scripts['setup:models'], /prepare-adore-webgpu\.mjs/);
  assert.match(packageJson.scripts['setup:fp16'], /setup:adore-fp16/);
  assert.doesNotMatch(packageJson.scripts['setup:fp16'], /python scripts\/convert-adore-fp16\.py/);
});
