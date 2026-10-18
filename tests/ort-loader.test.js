import test from 'node:test';
import assert from 'node:assert/strict';

test('rejected ORT import permits retry and concurrent callers share import', async t => {
  const previous = globalThis.chrome;
  t.after(() => { globalThis.chrome = previous; delete globalThis.ortImports; });
  let attempts = 0;
  globalThis.chrome = { runtime: { getURL(path) {
    if (path === 'vendor/') return 'vendor/';
    attempts++;
    return attempts === 1 ? 'data:text/javascript,throw new Error("import failed")' :
      `data:text/javascript,${encodeURIComponent('globalThis.ortImports=(globalThis.ortImports??0)+1; export const env={wasm:{},webgpu:{}};')}`;
  } } };
  const { loadOrt } = await import('../src/engine/ort-loader.js?retry-test');
  await assert.rejects(loadOrt(), /import failed/);
  const [first, second] = await Promise.all([loadOrt(), loadOrt()]);
  assert.equal(first, second);
  assert.equal(attempts, 2);
  assert.equal(globalThis.ortImports, 1);
  assert.equal(first.env.wasm.wasmPaths, 'vendor/');
});

test('ORT retry bypasses rejected ESM cache for the same resource URL', async t => {
  const previous = globalThis.chrome;
  t.after(() => { globalThis.chrome = previous; delete globalThis.allowOrtRetry; });
  const source = 'if(!globalThis.allowOrtRetry)throw Error("not ready"); export const env={wasm:{},webgpu:{}};';
  const url = `data:text/javascript,${encodeURIComponent(source)}`;
  globalThis.chrome = { runtime: { getURL: path => path === 'vendor/' ? 'vendor/' : url } };
  const { loadOrt } = await import('../src/engine/ort-loader.js?same-resource-test');
  await assert.rejects(loadOrt(), /not ready/);
  globalThis.allowOrtRetry = true;
  assert.equal((await loadOrt()).env.logLevel, 'warning');
});
