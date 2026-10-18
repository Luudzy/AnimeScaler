import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const root = new URL('../', import.meta.url);
const packageJson = JSON.parse(await readFile(new URL('package.json', root), 'utf8'));

test('extension package includes its inline host without private native connection files', () => {
  assert.ok(packageJson.files.includes('native-inline.html'));
  assert.ok(packageJson.files.includes('src/host/'));
  assert.equal(packageJson.files.some(entry => entry.startsWith('native/')), false);
});

test('extension package never bundles model weights', () => {
  assert.equal(packageJson.files.some(entry => /\.(onnx|pt|pth|npz|safetensors|glsl)$/i.test(entry)), false);
});
