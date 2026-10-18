import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createRuntimeManifest, verifyRuntimeManifest } from '../scripts/ort-runtime-integrity.mjs';

async function withVendor(run) {
  const root = await mkdtemp(join(tmpdir(), 'animescaler-ort-'));
  const vendor = join(root, 'vendor');
  await mkdir(vendor);
  try { await run(vendor); } finally { await rm(root, { recursive: true, force: true }); }
}

async function writeRuntime(vendor) {
  const files = ['ort.webgpu.bundle.min.mjs', 'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm'];
  for (const name of files) await writeFile(join(vendor, name), `runtime:${name}`);
  return files;
}

test('ORT manifest binds runtime file hashes to the expected package version', async () => {
  await withVendor(async (vendor) => {
    const files = await writeRuntime(vendor);
    const manifest = await createRuntimeManifest(vendor, '1.29.0', files);
    await writeFile(join(vendor, 'ort-runtime-manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
    assert.equal(await verifyRuntimeManifest(vendor, '1.29.0'), true);
  });
});

test('ORT preflight rejects an altered runtime file', async () => {
  await withVendor(async (vendor) => {
    const files = await writeRuntime(vendor);
    const manifest = await createRuntimeManifest(vendor, '1.29.0', files);
    await writeFile(join(vendor, 'ort-runtime-manifest.json'), JSON.stringify(manifest));
    await writeFile(join(vendor, files[0]), 'altered runtime');
    await assert.rejects(verifyRuntimeManifest(vendor, '1.29.0'), /SHA-256 mismatch/);
  });
});

test('ORT preflight rejects a runtime package version mismatch', async () => {
  await withVendor(async (vendor) => {
    const files = await writeRuntime(vendor);
    const manifest = await createRuntimeManifest(vendor, '1.28.0', files);
    await writeFile(join(vendor, 'ort-runtime-manifest.json'), JSON.stringify(manifest));
    await assert.rejects(verifyRuntimeManifest(vendor, '1.29.0'), /version mismatch/);
  });
});

test('ORT preflight rejects runtime files absent from the manifest', async () => {
  await withVendor(async (vendor) => {
    const files = await writeRuntime(vendor);
    const manifest = await createRuntimeManifest(vendor, '1.29.0', files);
    await writeFile(join(vendor, 'ort-runtime-manifest.json'), JSON.stringify(manifest));
    await writeFile(join(vendor, 'ort-wasm-extra.wasm'), 'untracked runtime');
    await assert.rejects(verifyRuntimeManifest(vendor, '1.29.0'), /file set mismatch/);
  });
});
