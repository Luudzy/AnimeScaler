import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import * as models from '../src/core/local-models.js';
import * as frames from '../src/core/frame-routing.js';
import * as capture from '../src/core/tab-capture.js';
import { normalizeSettings } from '../src/core/settings.js';

const routeModule = await import('../src/core/activation-route.js').catch(error => {
  if (error.code !== 'ERR_MODULE_NOT_FOUND') throw error;
  return {};
});
const { resolveActivationRoute } = routeModule;
const popupSource = await readFile(new URL('../popup/popup.js', import.meta.url), 'utf8');
const popupHtml = await readFile(new URL('../popup/popup.html', import.meta.url), 'utf8');

test('popup explains that native backends run in a separate window', () => {
  assert.match(popupHtml, /janela separada|outra janela/i);
});

test('popup delegates native viewer startup and stop to the extension service worker', () => {
  assert.match(popupSource, /ANIMESCALER_START_VIEWER/);
  assert.match(popupSource, /ANIMESCALER_STOP_VIEWER/);
  assert.doesNotMatch(popupSource, /chrome\.windows|chrome\.tabs\.create|window\.open/);
});

test('processed FPS help explains that reused frames count toward the metric', () => {
  const popup = popupHarness();
  popup.render({ state: 'running', captureMode: 'direct', processedFps: 24, stats: {
    sourceWidth: 640, sourceHeight: 360, outputWidth: 1280, outputHeight: 720,
    totalMs: 24, processedFps: 24, modelLabel: 'Adore', modelPrecision: 'fp16',
  } });
  const visibleMetric = popup.element('metrics').textContent
    .normalize('NFD').replace(/\p{Diacritic}/gu, '').toLowerCase();
  assert.match(visibleMetric, /(?:fps.{0,100}(?:inclui|conta|considera|contabiliza).{0,60}quadros? (?:reutilizados?|reaproveitados?)|quadros? (?:reutilizados?|reaproveitados?).{0,60}(?:sao )?(?:incluidos?|contados?|considerados?|contabilizados?).{0,60}fps)/);
});

function popupHarness({ direct = { state: 'running', captureMode: 'direct' }, restricted, settings = {}, probeState = 'stopped', frameResponses, viewerStart, viewerStatus } = {}) {
  const messages = [];
  const elements = new Map();
  const element = id => {
    if (!elements.has(id)) elements.set(id, {
      value: '', checked: false, style: {}, attributes: {}, classList: { toggle() {} },
      setAttribute(name, value) { this.attributes[name] = value; }, addEventListener(type, callback) { this[type] = callback; },
      querySelector() { return {}; },
    });
    return elements.get(id);
  };
  const probe = { state: probeState, hasVideo: true, videoScore: 10, frameId: 0 };
  const unexpected = () => { throw new Error('Viewer/window/fullscreen operation is forbidden'); };
  const context = vm.createContext({
    ...models, ...frames, ...capture, normalizeSettings, resolveActivationRoute,
    DiagnosticLog: class { error() {} }, performance, setTimeout, clearTimeout,
    document: { getElementById: element, querySelectorAll: () => [], exitFullscreen: unexpected },
    chrome: {
      tabs: {
        query: async () => [{ id: 7 }],
        async sendMessage(tabId, message, options) {
          messages.push({ tabId, message, options });
          if (message.type === 'ANIMESCALER_PROBE') return frameResponses ? frameResponses[options.frameId] : probe;
          if (message.type === 'ANIMESCALER_START') return direct;
          if (message.type === 'ANIMESCALER_START_TAB_CAPTURE') return restricted;
          if (message.type === 'ANIMESCALER_STOP') return { state: 'stopped' };
          if (message.type === 'ANIMESCALER_SETTINGS') return { state: 'running' };
          throw new Error(`Unexpected message: ${message.type}`);
        },
      },
      webNavigation: { getAllFrames: async () => frameResponses
        ? Object.keys(frameResponses).map(id => ({ frameId: Number(id), parentFrameId: -1, url: 'https://player.test/' }))
        : [{ frameId: 0, parentFrameId: -1, url: 'https://player.test/' }] },
      tabCapture: { getMediaStreamId: async () => 'isolated-stream' },
      storage: { local: { set: async () => {} }, session: { get: unexpected, remove: unexpected } },
      runtime: { sendMessage: async message => {
        messages.push({ message, source: 'runtime' });
        if (message.type === 'ANIMESCALER_START_VIEWER') return viewerStart || { state: 'running', captureMode: 'viewer' };
        if (message.type === 'ANIMESCALER_STOP_VIEWER') return { state: 'stopped' };
        if (message.type === 'ANIMESCALER_VIEWER_STATUS') return viewerStatus || { state: 'stopped' };
        throw new Error(`Unexpected runtime message: ${message.type}`);
      } },
      windows: { create: unexpected, get: unexpected, remove: unexpected, update: unexpected },
    },
  });
  const source = popupSource.replace(/^import[\s\S]*?;\r?$/gm, '').split("const { setupDiagnostics }")[0];
  vm.runInContext(`${source}\nglobalThis.popup = { startScaler, render, refreshStatus, updateBackendOptions, probeFrames };`, context);
  element('backend').value = settings.backend || 'webgpu';
  element('model').value = settings.model || 'adore';
  return { ...context.popup, element, messages, chrome: context.chrome };
}

test('unavailable model explains disabled activation beside button without a loading cursor', () => {
  const popup = popupHarness({ settings: { model: 'animejanai', backend: 'webgpu' } });
  popup.updateBackendOptions();
  assert.equal(popup.element('toggle').disabled, true);
  assert.equal(popup.element('toggle').attributes['aria-busy'], 'false');
  assert.equal(popup.element('toggle').style.cursor, 'not-allowed');
  assert.equal(popup.element('activationReason').hidden, false);
  assert.match(popup.element('activationReason').textContent, /WebGPU.*qualificad/);
  popup.element('backend').value = 'rocm';
  popup.element('model').value = 'zenkai-1-0';
  popup.updateBackendOptions();
  assert.equal(popup.element('toggle').disabled, false);
  assert.equal(popup.element('activationReason').hidden, true);
});

test('running session keeps Stop available despite incompatible pending selection', () => {
  const popup = popupHarness();
  popup.render({ state: 'running' });
  popup.updateBackendOptions();
  assert.equal(popup.element('toggle').disabled, false);
  assert.equal(popup.element('toggle').attributes['aria-busy'], 'false');
});

test('activation busy state clears on failure and survives compatibility refresh', async () => {
  let reject;
  const direct = new Promise((_, failed) => { reject = failed; });
  const popup = popupHarness({ settings: { backend: 'rocm', model: 'zenkai-1-0' }, viewerStart: direct });
  popup.updateBackendOptions();
  const activating = popup.element('toggle').click();
  await new Promise(resolve => setImmediate(resolve));
  popup.updateBackendOptions();
  assert.equal(popup.element('toggle').disabled, true);
  assert.equal(popup.element('toggle').attributes['aria-busy'], 'true');
  reject(new Error('service unavailable'));
  await activating;
  assert.equal(popup.element('toggle').attributes['aria-busy'], 'false');
  assert.equal(popup.element('toggle').disabled, false);
  assert.equal(popup.element('status').textContent, 'Erro');
});

test('unresponsive iframe probe expires at the existing 4500 ms discovery deadline', async t => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const popup = popupHarness({ frameResponses: {
    0: new Promise(() => {}), 1: { state: 'stopped', hasVideo: true, videoScore: 100 },
  } });
  let result;
  const probing = popup.probeFrames({ id: 7 }).then(value => { result = value; });
  await new Promise(resolve => setImmediate(resolve));
  t.mock.timers.tick(4499);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(result, undefined);
  t.mock.timers.tick(1);
  await new Promise(resolve => setImmediate(resolve));
  assert.ok(result, 'one iframe must not keep the entire popup awaiting forever');
  await probing;
  assert.equal(result[0].ok, false);
  assert.equal(result[0].error.code, 'FRAME_RESPONSE_TIMEOUT');
  assert.equal(result[1].hasVideo, true);
});

test('activation helper exists without browser dependencies', () => {
  assert.equal(typeof resolveActivationRoute, 'function');
});

for (const backend of ['rocm', 'tensorrt', 'opencl', 'd3d12']) {
  test(`${backend} starts in the viewer window`, async () => {
    const settings = Object.freeze({ backend, exclusiveViewer: true });
    assert.equal(resolveActivationRoute(settings, { state: 'stopped' }), 'viewer');
    const popup = popupHarness({ settings });
    const result = await popup.startScaler({ id: 7 }, { frameId: 3 }, settings);
    assert.equal(result.state, 'running');
    const request = popup.messages.find(({ message }) => message.type === 'ANIMESCALER_START_VIEWER');
    assert.ok(request);
    assert.equal(request.message.settings.exclusiveViewer, false);
    assert.equal(request.message.settings.backend, backend);
    assert.equal(request.message.target.frameId, 3);
    assert.equal(request.message.tab.id, 7);
    assert.equal(settings.exclusiveViewer, true);
  });
}

test('WebGPU remains inline', async () => {
  const settings = { backend: 'webgpu' };
  assert.equal(resolveActivationRoute(settings, { state: 'stopped' }), 'inline');
  const popup = popupHarness({ settings });
  await popup.startScaler({ id: 7 }, { frameId: 3 }, settings);
  assert.ok(popup.messages.some(({ message }) => message.type === 'ANIMESCALER_START'));
  assert.ok(!popup.messages.some(({ message }) => message.type === 'ANIMESCALER_START_VIEWER'));
});

for (const state of ['running', 'starting']) {
  test(`${state} activation stops inline for every backend`, async () => {
    for (const backend of ['webgpu', 'rocm', 'tensorrt', 'opencl', 'd3d12']) {
      assert.equal(resolveActivationRoute({ backend, exclusiveViewer: true }, { state }), 'stop');
    }
    const popup = popupHarness({ probeState: state });
    popup.render({ state });
    await popup.element('toggle').click();
    assert.equal(popup.element('status').textContent, 'Parado');
    assert.ok(popup.messages.some(({ message }) => message.type === 'ANIMESCALER_STOP'));
  });
}

test('empty and failed statuses can retry using the selected backend route', () => {
  assert.equal(resolveActivationRoute(), 'inline');
  assert.equal(resolveActivationRoute({}, { state: 'error', error: { code: 'SOURCE_NOT_ISOLATED' } }), 'inline');
  assert.equal(resolveActivationRoute({ backend: 'rocm' }, { state: 'error', error: { code: 'VIEWER_START_FAILED' } }), 'viewer');
});

for (const code of ['SOURCE_NOT_ISOLATED', 'ELEMENT_CAPTURE_UNAVAILABLE', 'TAB_CAPTURE_NEEDS_VIEWER']) {
  test(`${code} WebGPU capture failure stays explicit`, async () => {
    const error = { state: 'error', captureMode: 'tab', error: { code, message: 'Capture restriction failed' } };
    const popup = popupHarness({ direct: { state: 'error', error: { code: 'CROSS_ORIGIN_VIDEO' } }, restricted: error });
    const result = await popup.startScaler({ id: 7 }, { frameId: 0 }, { backend: 'webgpu', exclusiveViewer: true });
    assert.equal(result, error);
    popup.render(result);
    const visible = popup.element('metrics').textContent;
    const messageIndex = visible.indexOf(error.error.message);
    const codeIndex = visible.indexOf(code);
    assert.notEqual(messageIndex, -1);
    assert.notEqual(codeIndex, -1);
    assert.ok(messageIndex < codeIndex);
    const start = popup.messages.find(({ message }) => message.type === 'ANIMESCALER_START_TAB_CAPTURE');
    assert.equal(start.message.settings.backend, 'webgpu');
    assert.equal(start.message.settings.exclusiveViewer, false);
  });
}

test('error status presents readable message before technical code and keeps both', () => {
  const popup = popupHarness();
  const message = 'Não foi possível isolar a captura do player.';
  const code = 'SOURCE_NOT_ISOLATED';
  popup.render({ state: 'error', error: { code, message } });

  const visible = popup.element('metrics').textContent;
  const messageIndex = visible.indexOf(message);
  const codeIndex = visible.indexOf(code);
  assert.notEqual(messageIndex, -1, 'readable error message remains visible');
  assert.notEqual(codeIndex, -1, 'technical error code remains visible');
  assert.ok(messageIndex < codeIndex, 'readable message appears before technical code');
});

test('capture API unavailable stays explicit', async () => {
  const popup = popupHarness({ direct: { state: 'error', error: { code: 'CROSS_ORIGIN_VIDEO' } } });
  delete popup.chrome.tabCapture;
  const result = await popup.startScaler({ id: 7 }, { frameId: 0 }, { backend: 'webgpu' });
  assert.equal(result.state, 'error');
  assert.equal(result.error.code, 'TAB_CAPTURE_API_UNAVAILABLE');
});

test('settings changes preserve inline session and disable legacy fullscreen preference', async () => {
  const popup = popupHarness({ settings: { backend: 'webgpu', exclusiveViewer: true } });
  await popup.startScaler({ id: 7 }, { frameId: 0 }, { backend: 'webgpu' });
  popup.render({ state: 'running', captureMode: 'direct' });
  await popup.element('backend').change();
  const update = popup.messages.find(({ message }) => message.type === 'ANIMESCALER_SETTINGS');
  assert.ok(update);
  assert.equal(update.message.settings.exclusiveViewer, false);
  assert.equal(update.message.settings.backend, 'webgpu');
});

test('status refresh uses original frames without querying viewer windows', async () => {
  const popup = popupHarness();
  await popup.refreshStatus();
  assert.equal(popup.element('status').textContent, 'Parado');
});

test('status refresh reports the native viewer window as active', async () => {
  const popup = popupHarness({ viewerStatus: { state: 'running', captureMode: 'viewer' } });
  await popup.refreshStatus();
  assert.equal(popup.element('status').textContent, 'Ativo');
  assert.match(popup.element('metrics').textContent, /Janela separada/);
  assert.equal(popup.messages.filter(({ message }) => message.type === 'ANIMESCALER_PROBE').length, 0);
});

test('deactivate closes the viewer window through the service worker', async () => {
  const popup = popupHarness({ viewerStatus: { state: 'running', captureMode: 'viewer' } });
  popup.render({ state: 'running', captureMode: 'viewer' });
  await popup.element('toggle').click();
  assert.ok(popup.messages.some(({ message }) => message.type === 'ANIMESCALER_STOP_VIEWER'));
  assert.equal(popup.element('status').textContent, 'Parado');
});
