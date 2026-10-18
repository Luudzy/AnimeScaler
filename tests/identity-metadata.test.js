import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const manifest = JSON.parse(await readFile(new URL('../manifest.json', import.meta.url), 'utf8'));
const packageJson = JSON.parse(await readFile(new URL('../package.json', import.meta.url), 'utf8'));

test('public extension identity names AnimeScaler and limits claims to gated routes', () => {
  assert.equal(manifest.name, 'AnimeScaler');
  assert.match(manifest.description, /WebGPU/i);
  assert.match(manifest.description, /experiment/i);
  assert.match(manifest.description, /runtime/i);
});

test('package metadata distinguishes neural models, shaders and experimental routes', () => {
  assert.match(packageJson.description, /neural/i);
  assert.match(packageJson.description, /shader/i);
  assert.match(packageJson.description, /experimental/i);
});
