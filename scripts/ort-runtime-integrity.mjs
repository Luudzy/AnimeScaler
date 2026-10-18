import { createHash, randomUUID } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { readFile, readdir, rename, rm, writeFile } from 'node:fs/promises';
import { basename, join } from 'node:path';

const MANIFEST = 'ort-runtime-manifest.json';
const isRuntimeFile = (name) => name === 'ort.webgpu.bundle.min.mjs' || /^ort-wasm.*\.(wasm|mjs)$/.test(name);

async function sha256(path) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest('hex');
}

function validateFileSet(names) {
  if (!names.includes('ort.webgpu.bundle.min.mjs') || !names.some((name) => name.endsWith('.wasm')) || !names.some((name) => name.endsWith('.mjs'))) {
    throw new Error('ORT runtime manifest is missing the WebGPU bundle or WASM runtime files');
  }
}

export async function createRuntimeManifest(directory, version, names) {
  if (typeof version !== 'string' || !version) throw new Error('ORT package version is required');
  const sortedNames = [...new Set(names)].sort();
  validateFileSet(sortedNames);
  const files = {};
  for (const name of sortedNames) {
    if (basename(name) !== name || !isRuntimeFile(name)) throw new Error(`Invalid ORT runtime filename: ${name}`);
    files[name] = await sha256(join(directory, name));
  }
  return { schemaVersion: 1, package: { name: 'onnxruntime-web', version }, files };
}

export async function writeRuntimeManifest(directory, version, names) {
  const manifest = await createRuntimeManifest(directory, version, names);
  const path = join(directory, MANIFEST);
  const temp = `${path}.${randomUUID()}.tmp`;
  try {
    await writeFile(temp, `${JSON.stringify(manifest, null, 2)}\n`, { flag: 'wx' });
    await rename(temp, path);
  } catch (error) {
    await rm(temp, { force: true });
    throw error;
  }
  return manifest;
}

export async function verifyRuntimeManifest(directory, expectedVersion) {
  const manifest = JSON.parse(await readFile(join(directory, MANIFEST), 'utf8'));
  if (manifest.schemaVersion !== 1 || manifest.package?.name !== 'onnxruntime-web') throw new Error('Unsupported ORT runtime manifest');
  if (manifest.package.version !== expectedVersion) throw new Error(`ORT package version mismatch: expected ${expectedVersion}, got ${manifest.package.version}`);
  if (!manifest.files || typeof manifest.files !== 'object' || Array.isArray(manifest.files)) throw new Error('Invalid ORT runtime manifest file list');
  const expectedNames = Object.keys(manifest.files).sort();
  validateFileSet(expectedNames);
  if (expectedNames.some((name) => basename(name) !== name || !isRuntimeFile(name))) throw new Error('Invalid ORT runtime manifest filename');
  const actualNames = (await readdir(directory)).filter(isRuntimeFile).sort();
  if (JSON.stringify(actualNames) !== JSON.stringify(expectedNames)) throw new Error('ORT runtime file set mismatch');
  for (const name of expectedNames) {
    const actual = await sha256(join(directory, name));
    if (actual !== manifest.files[name]) throw new Error(`ORT runtime SHA-256 mismatch: ${name}`);
  }
  return true;
}
