import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { copyFile, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { prepareAnimejanai } from '../scripts/prepare-animejanai-artifact.mjs';

const sourceFixture = new URL('../models/2x_AnimeJaNai_HD_V3_SuperUltraCompact.onnx', import.meta.url);
const outputName = '2x_AnimeJaNai_HD_V3_SuperUltraCompact_webgpu.onnx';
const expectedOutputHash = '3eca0dee498cc0bfb311a6ed97e23bf225f7660de033ca9683ffefd155c0b164';
const hasAnimeJaNaiSource = existsSync(new URL('../models/2x_AnimeJaNai_HD_V3_SuperUltraCompact.onnx', import.meta.url));

async function withModelRoot(run) {
  const root = await mkdtemp(join(tmpdir(), 'animescaler-animejanai-'));
  const models = join(root, 'models');
  await mkdir(models);
  await copyFile(sourceFixture, join(models, '2x_AnimeJaNai_HD_V3_SuperUltraCompact.onnx'));
  try { await run(root, join(models, outputName)); }
  finally { await rm(root, { recursive: true, force: true }); }
}

test('AnimeJaNai preparation preserves an existing unverified output', { skip: !hasAnimeJaNaiSource && 'download the official AnimeJaNai source first' }, async () => {
  await withModelRoot(async (root, target) => {
    const existing = Buffer.from('preserve user artifact');
    await writeFile(target, existing);
    await assert.rejects(prepareAnimejanai({ root }), /Existing AnimeJaNai WebGPU output differs/);
    assert.deepEqual(await readFile(target), existing);
  });
});

test('AnimeJaNai preparation creates a verified output once and reuses it', { skip: !hasAnimeJaNaiSource && 'download the official AnimeJaNai source first' }, async () => {
  await withModelRoot(async (root, target) => {
    const first = await prepareAnimejanai({ root });
    const output = await readFile(target);
    assert.equal(first.prepared, true);
    assert.equal(first.sha256, expectedOutputHash);
    const second = await prepareAnimejanai({ root });
    assert.equal(second.prepared, false);
    assert.deepEqual(await readFile(target), output);
  });
});
