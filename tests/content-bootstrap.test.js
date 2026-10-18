import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import { buildContent } from '../scripts/build-content.mjs';

const source = await readFile(new URL('../content-bootstrap.js', import.meta.url), 'utf8');

test('distributed bootstrap is reproducible from current source modules using Node only', async () => {
  assert.equal(await buildContent({ write: false }), source);
});

function frame() {
  const listeners = [], errors = [];
  let imports = 0;
  const realm = vm.createContext({ performance, setTimeout, clearTimeout, URL,
    MutationObserver: class { observe() {} disconnect() {} },
    document: {}, console: { error: (...args) => errors.push(args) },
    chrome: { runtime: {
      getURL: path => `chrome-extension://test/${path}`,
      onMessage: { addListener: listener => listeners.push(listener) },
    } },
    browserImport() {
      imports++;
      return Promise.reject(new Error('Unsafe attempt to load URL chrome-extension://test/src/content/main.js'));
    },
  });
  // Adapt browser module fetching at the boundary; Node VM has no Chromium resource loader.
  const script = new vm.Script(source.replace(/\bimport\s*\(/g, 'browserImport('));
  return { realm, listeners, errors, imports: () => imports, run: () => script.runInContext(realm) };
}

test('registered bootstrap installs controller even when frame resource imports are denied', async () => {
  const state = frame();
  state.run();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(state.listeners.length, 1, 'controller must install without importing content modules through the frame');
  assert.equal(state.imports(), 0);
  assert.equal(state.errors.length, 0);
});

test('duplicate injection stays idempotent while different frames get independent controllers', async () => {
  const first = frame(), second = frame();
  first.run(); first.run(); second.run();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(first.listeners.length, 1);
  assert.equal(second.listeners.length, 1);
  assert.notEqual(first.realm.__animeScalerController, second.realm.__animeScalerController);
});

test('bootstrap reports constructor failure and permits a later explicit injection', () => {
  const state = frame();
  state.realm.MutationObserver = class { constructor() { throw new Error('observer unavailable'); } };
  state.run();
  assert.equal(state.errors.length, 1);
  assert.equal(state.listeners.length, 0);
  assert.equal(state.realm.__animeScalerBootstrapRequested, false);
  state.realm.MutationObserver = class { observe() {} disconnect() {} };
  state.run(); state.run();
  assert.equal(state.listeners.length, 1);
});

test('listener registration failure leaves bootstrap retryable', () => {
  const state = frame();
  let attempts = 0;
  state.realm.chrome.runtime.onMessage.addListener = listener => {
    attempts++;
    if (attempts === 1) throw new Error('extension context unavailable');
    state.listeners.push(listener);
  };

  state.run();
  assert.equal(state.errors.length, 1);
  assert.equal(state.realm.__animeScalerBootstrapRequested, false);
  assert.equal(state.realm.__animeScalerController, undefined);
  state.run();

  assert.equal(attempts, 2);
  assert.equal(state.listeners.length, 1);
  assert.ok(state.realm.__animeScalerController);
});

test('distributed bootstrap preserves asynchronous STATUS message handling', async () => {
  const state = frame();
  state.run();
  let response;
  const keepChannel = state.listeners[0]({ type: 'ANIMESCALER_STATUS' }, {}, value => { response = value; });
  assert.equal(keepChannel, true);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(response.state, 'stopped');
  assert.equal(response.captureMode, 'direct');
  assert.equal(state.listeners[0]({ type: 'UNRELATED' }, {}, () => {}), false);
});
