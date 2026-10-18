import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createHash } from 'node:crypto';
import { buildScopeManifest, writeScopeManifest } from '../scripts/scope-manifest.mjs';

async function withRoot(run) {
  const root = await mkdtemp(join(tmpdir(), 'animescaler-scope-'));
  try { await run(root); } finally { await rm(root, { recursive: true, force: true }); }
}

test('scope manifest sorts paths and records exact content hashes and timestamp', async () => {
  await withRoot(async (root) => {
    await mkdir(join(root, 'src'));
    await writeFile(join(root, 'src', 'b.js'), 'beta');
    await writeFile(join(root, 'a.js'), 'alpha');
    const manifest = await buildScopeManifest(['src/b.js', 'a.js'], { root, capturedAt: '2026-10-13T12:00:00.000Z' });
    assert.equal(manifest.capturedAt, '2026-10-13T12:00:00.000Z');
    assert.deepEqual(manifest.files.map((entry) => entry.path), ['a.js', 'src/b.js']);
    assert.equal(manifest.files[0].sha256, createHash('sha256').update('alpha').digest('hex'));
    const output = join(root, 'evidence', 'scope.json');
    await writeScopeManifest(output, ['src/b.js', 'a.js'], { root, capturedAt: manifest.capturedAt });
    assert.deepEqual(JSON.parse(await readFile(output, 'utf8')), manifest);
  });
});

test('scope manifest rejects paths that escape the project root', async () => {
  await withRoot(async (root) => {
    await assert.rejects(buildScopeManifest(['../outside.txt'], { root }), /outside project root/);
  });
});
