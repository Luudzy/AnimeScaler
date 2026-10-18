import test from 'node:test';
import assert from 'node:assert/strict';
import { setupDiagnostics } from '../popup/diagnostics.js';

function diagnosticsHarness(t, { reply = { ok: true, data: { schemaVersion: 1, events: [], dropped: 0 } }, click } = {}) {
  const original = { document: globalThis.document, chrome: globalThis.chrome, setTimeout: globalThis.setTimeout };
  const created = [];
  const revoked = [];
  const elements = Object.fromEntries(['diagnosticStatus', 'exportDiagnostics', 'clearDiagnostics'].map(id => [id, {
    addEventListener(_, callback) { this.handler = callback; },
  }]));
  const anchor = { click() { click?.(); } };
  globalThis.document = {
    getElementById: id => elements[id],
    createElement: () => anchor,
  };
  globalThis.chrome = {
    runtime: { getManifest: () => ({ version: 'test' }), sendMessage: async () => reply },
  };
  t.mock.method(URL, 'createObjectURL', () => {
    const url = `blob:diagnostics-${created.length + 1}`;
    created.push(url);
    return url;
  });
  t.mock.method(URL, 'revokeObjectURL', url => revoked.push(url));
  globalThis.setTimeout = callback => callback();
  t.after(() => Object.assign(globalThis, original));
  setupDiagnostics();
  return { elements, created, revoked, anchor };
}

test('popup exports valid diagnostic JSON and reports storage failures', async t => {
  const original = {document: globalThis.document, chrome: globalThis.chrome, setTimeout: globalThis.setTimeout};
  let blob, clicked = false;
  const elements = Object.fromEntries(['diagnosticStatus', 'exportDiagnostics', 'clearDiagnostics'].map(id => [id, {
    addEventListener(_, callback) { this.handler = callback; },
  }]));
  globalThis.document = {getElementById: id => elements[id], createElement: () => ({click() {clicked = true;}})};
  globalThis.chrome = {runtime: {getManifest: () => ({version: 'test'}), sendMessage: async () => ({ok: true, data: {
    schemaVersion: 1, events: [{type: 'performance'}], dropped: 0,
  }})}};
  t.mock.method(URL, 'createObjectURL', value => {blob = value; return 'blob:test';});
  t.mock.method(URL, 'revokeObjectURL', () => {});
  globalThis.setTimeout = callback => callback();
  t.after(() => Object.assign(globalThis, original));
  setupDiagnostics();
  await elements.exportDiagnostics.handler();
  assert.equal(clicked, true);
  assert.equal(JSON.parse(await blob.text()).events[0].type, 'performance');
  assert.equal(elements.exportDiagnostics.disabled, false);
  globalThis.chrome.runtime.sendMessage = async () => ({ok: false, error: 'storage failed'});
  await elements.exportDiagnostics.handler();
  assert.equal(elements.diagnosticStatus.textContent, 'storage failed');
});

test('successful export revokes each created URL exactly once', async t => {
  const harness = diagnosticsHarness(t);
  await harness.elements.exportDiagnostics.handler();

  assert.equal(harness.created.length, 1);
  assert.deepEqual(harness.revoked, harness.created);
});

test('download click error still revokes the created URL exactly once', async t => {
  const harness = diagnosticsHarness(t, { click() { throw new Error('download failed'); } });
  await harness.elements.exportDiagnostics.handler();

  assert.equal(harness.created.length, 1);
  assert.deepEqual(harness.revoked, harness.created);
  assert.equal(harness.elements.diagnosticStatus.textContent, 'download failed');
});

test('storage read error does not revoke a URL that was never created', async t => {
  const harness = diagnosticsHarness(t, { reply: { ok: false, error: 'storage failed' } });
  await harness.elements.exportDiagnostics.handler();

  assert.deepEqual(harness.created, []);
  assert.deepEqual(harness.revoked, []);
  assert.equal(harness.elements.diagnosticStatus.textContent, 'storage failed');
});
