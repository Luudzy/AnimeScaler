import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { mkdtemp, mkdir, cp, readFile, writeFile, rm } from 'node:fs/promises';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';

const root = fileURLToPath(new URL('../../', import.meta.url));
const executablePath = 'C:\\chrome-win\\chrome.exe';
const probePortName = 'animescaler-browser-api-probe-v1';
const offscreenPortName = 'animescaler-browser-api-offscreen-probe-v1';
const swOffscreenPortName = 'animescaler-browser-api-sw-offscreen-probe-v1';

function observedSenderFields(sender) {
  return Object.fromEntries(['id', 'url', 'tabId', 'documentId'].map(field => [field,
    sender[field] === undefined ? { state: 'absent' } : { state: 'observed', value: sender[field] }]));
}

function assertAbsentSenderFields(sender) {
  const fields = observedSenderFields(sender);
  assert.equal(fields.tabId.state, 'absent', 'Chromium 151 sender.tabId must be absent');
  assert.equal(fields.documentId.state, 'absent', 'Chromium 151 sender.documentId must be absent');
}

async function connectWorkerToOffscreen(worker, nonce) {
  return worker.evaluate(({ name, nonce }) => new Promise((resolve, reject) => {
    const events = globalThis.__browserApiSwPortEvents ??= [];
    const ports = globalThis.__browserApiSwPorts ??= {};
    const port = chrome.runtime.connect({ name });
    ports[nonce] = port;
    events.push({ type: 'sw-connect-called', name, nonce });
    let received = false;
    const timer = setTimeout(() => {
      events.push({ type: 'sw-port-timeout', name, nonce });
      port.disconnect();
      reject(new Error('SW_OFFSCREEN_PORT_REPLY_TIMEOUT'));
    }, 5000);
    port.onMessage.addListener(reply => {
      events.push({ type: 'sw-port-message', name, nonce, reply });
      clearTimeout(timer);
      received = true;
      resolve(reply);
    });
    port.onDisconnect.addListener(() => {
      const error = chrome.runtime.lastError?.message ?? null;
      events.push({ type: 'sw-port-disconnect', name, nonce, error });
      clearTimeout(timer);
      if (!received) reject(new Error(`SW_OFFSCREEN_PORT_DISCONNECTED: ${error ?? 'no reply'}`));
    });
    port.postMessage({ nonce });
  }), { name: swOffscreenPortName, nonce });
}

for (const field of ['tabId', 'documentId']) for (const value of [null, 'unexpected']) {
  test(`sender absence assertion rejects observed ${field}=${JSON.stringify(value)}`, () => {
    const sender = { [field]: value };
    assert.equal(observedSenderFields(sender)[field].state, 'observed');
    assert.throws(() => assertAbsentSenderFields(sender), { name: 'AssertionError' });
  });
}

async function buildProbeCopy(destination) {
  await mkdir(destination);
  // Only static extension code/assets are needed; models, credentials and caches are never copied.
  for (const entry of ['src', 'popup', 'assets', 'background.js', 'content-bootstrap.js', 'manifest.json']) {
    await cp(path.join(root, entry), path.join(destination, entry), { recursive: true });
  }
  const manifest = JSON.parse(await readFile(path.join(destination, 'manifest.json'), 'utf8'));
  manifest.permissions = [...new Set([...manifest.permissions, 'offscreen'])];
  manifest.background.service_worker = '__browser-api-probe-worker.mjs';
  manifest.content_scripts.find(script => script.js.includes('content-bootstrap.js')).js.push('__browser-api-probe-content.js');
  await writeFile(path.join(destination, '__browser-api-probe-worker.mjs'), `
import './background.js';
// Browser API instrumentation in a disposable copy, never a native inference relay.
globalThis.__browserApiProbeStartupMarker = 'Browser API probe worker startup';
console.info(globalThis.__browserApiProbeStartupMarker);
const events = globalThis.__browserApiProbeEvents = [];
const senderInfo = sender => ({ id: sender.id, url: sender.url, origin: sender.origin,
  tabId: sender.tab?.id, frameId: sender.frameId, documentId: sender.documentId });
const receiver = { id: chrome.runtime.id, url: self.location.href };
chrome.runtime.onMessage.addListener((message, sender, respond) => {
  if (message.type === 'BROWSER_API_OFFSCREEN_PORT_EVENT') {
    events.push({ type: 'offscreen-port-event', event: message.event, sender: senderInfo(sender) });
    respond({ received: true });
    return;
  }
  if (message.type === 'BROWSER_API_OFFSCREEN_FAILURE') {
    console.error('Browser API offscreen failed:', message.error);
    events.push({ type: 'offscreen-error', error: message.error, sender: senderInfo(sender) });
    respond({ received: true });
    return;
  }
  if (message.type !== 'BROWSER_API_PROBE_PING') return;
  const result = { kind: 'browser-api-probe', nonce: message.nonce, sender: senderInfo(sender), receiver, portReply: message.portReply };
  events.push({ type: 'message', ...result });
  respond(result);
});
chrome.runtime.onConnect.addListener(port => {
  if (port.name === '${swOffscreenPortName}') {
    // Do not echo or close this channel: only the offscreen onConnect may answer it.
    events.push({ type: 'probe-other-receiver-connect', name: port.name, sender: senderInfo(port.sender), receiver });
    return;
  }
  if (!['${probePortName}', '${offscreenPortName}'].includes(port.name)) { port.disconnect(); return; }
  const sender = senderInfo(port.sender);
  events.push({ type: 'connect', name: port.name, sender, receiver });
  port.onMessage.addListener(message => {
    events.push({ type: 'port-message', name: port.name, nonce: message.nonce, sender, receiver });
    port.postMessage({ kind: 'browser-api-probe', nonce: message.nonce, sender, receiver });
  });
  port.onDisconnect.addListener(() => events.push({ type: 'disconnect', name: port.name, sender }));
});
`);
  await writeFile(path.join(destination, '__browser-api-probe-content.js'), `
// This fixture lives in the isolated content world, using actual chrome.runtime APIs.
chrome.runtime.onMessage.addListener((message, sender, respond) => {
  if (message.type !== 'BROWSER_API_CONTENT_PING') return;
  respond({ nonce: message.nonce, sender: { id: sender.id, url: sender.url, tabId: sender.tab?.id },
    receiver: { id: chrome.runtime.id, url: location.href } });
});
chrome.runtime.sendMessage({ type: 'BROWSER_API_PROBE_PING', nonce: 'content-message' }).then(reply => {
  document.documentElement.dataset.browserApiProbeMessage = JSON.stringify(reply);
}).catch(error => console.error('Browser API content message failed:', error.message));
const port = chrome.runtime.connect({ name: '${probePortName}' });
port.onMessage.addListener(reply => { document.documentElement.dataset.browserApiProbePort = JSON.stringify(reply); });
port.onDisconnect.addListener(() => {
  if (chrome.runtime.lastError) console.error('Browser API content Port failed:', chrome.runtime.lastError.message);
});
port.postMessage({ nonce: 'content-port' });
`);
  await writeFile(path.join(destination, '__browser-api-probe-offscreen.html'),
    '<!doctype html><html><head><title>Disposable Browser API offscreen probe</title></head><body><script src="__browser-api-probe-offscreen.js"></script></body></html>');
  await writeFile(path.join(destination, '__browser-api-probe-offscreen.js'), `
// Offscreen Browser API fixture only; no content relay, capture, inference or credentials.
const reportFailure = error => chrome.runtime.sendMessage({ type: 'BROWSER_API_OFFSCREEN_FAILURE',
  error: String(error?.message ?? error) }).catch(() => {});
addEventListener('error', event => { void reportFailure(event.message); });
addEventListener('unhandledrejection', event => { void reportFailure(event.reason); });
const parsed = new DOMParser().parseFromString('<p>probe</p>', 'text/html').body.textContent === 'probe';
chrome.runtime.onMessage.addListener((message, sender, respond) => {
  if (message.type !== 'BROWSER_API_OFFSCREEN_PING') return;
  respond({ kind: 'browser-api-offscreen', nonce: message.nonce, parsed,
    sender: { id: sender.id, url: sender.url, tabId: sender.tab?.id, documentId: sender.documentId },
    receiver: { id: chrome.runtime.id, url: location.href } });
});
chrome.runtime.onConnect.addListener(port => {
  if (port.name !== '${swOffscreenPortName}') return;
  const connection = crypto.randomUUID();
  const sender = { id: port.sender?.id, url: port.sender?.url, tabId: port.sender?.tab?.id,
    documentId: port.sender?.documentId };
  const receiver = { id: chrome.runtime.id, url: location.href };
  const record = (type, details = {}) => chrome.runtime.sendMessage({ type: 'BROWSER_API_OFFSCREEN_PORT_EVENT',
    event: { type, connection, name: port.name, sender, receiver, ...details } }).catch(reportFailure);
  void record('connect');
  port.onMessage.addListener(message => {
    void record('message', { nonce: message.nonce });
    port.postMessage({ kind: 'browser-api-sw-offscreen-port', connection, name: port.name,
      nonce: message.nonce, sender, receiver });
  });
  port.onDisconnect.addListener(() => { void record('disconnect'); });
});
const port = chrome.runtime.connect({ name: '${offscreenPortName}' });
let acknowledged = false;
port.onMessage.addListener(reply => {
  if (acknowledged) return;
  acknowledged = true;
  chrome.runtime.sendMessage({ type: 'BROWSER_API_PROBE_PING', nonce: 'offscreen-port-received',
    portReply: reply }).catch(reportFailure);
});
port.onDisconnect.addListener(() => {
  if (chrome.runtime.lastError) void reportFailure(chrome.runtime.lastError);
});
port.postMessage({ nonce: 'offscreen-port' });
`);
  await writeFile(path.join(destination, 'manifest.json'), JSON.stringify(manifest, null, 2));
}

async function buildNativeRelayProbeCopy(destination, { stubEngine = false, slowFrames = false } = {}) {
  await mkdir(destination);
  for (const entry of ['src', 'popup', 'viewer', 'assets', 'background.js', 'content-bootstrap.js', 'manifest.json', 'native-inline.html']) {
    await cp(path.join(root, entry), path.join(destination, entry), { recursive: true });
  }
  const contentPath = path.join(destination, 'content-bootstrap.js');
  const contentBootstrap = await readFile(contentPath, 'utf8');
  await writeFile(contentPath, `${contentBootstrap}\n${nativeRelayContentProbe}`, 'utf8');

  const hostHtmlPath = path.join(destination, 'native-inline.html');
  const hostHtml = await readFile(hostHtmlPath, 'utf8');
  assert.match(hostHtml, /<\/body\s*>/i, 'production native-inline.html needs a body for passive E2E observation');
  await writeFile(hostHtmlPath, hostHtml.replace(/<\/body\s*>/i,
    '<script src="__native-relay-e2e-host-observer.js"></script></body>'), 'utf8');
  await writeFile(path.join(destination, '__native-relay-e2e-host-observer.js'), nativeRelayHostObserver, 'utf8');
  if (stubEngine) {
    await writeFile(path.join(destination, 'src/engine/native-engine.js'), `
const output = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII=';
export class NativeEngine {
  async initialize({ canvas }) { this.canvas = canvas; }
  async updateSettings() {}
  needsReconfigure() { return false; }
  async reconfigure() {}
  async processFrame(video, crop) {
    const width = Math.max(1, Math.round(crop.width * 2));
    const height = Math.max(1, Math.round(crop.height * 2));
    this.canvas.width = width; this.canvas.height = height;
    this.canvas.getContext('2d').drawImage(video, crop.x, crop.y, crop.width, crop.height, 0, 0, width, height);
    return { sourceWidth: crop.width, sourceHeight: crop.height, outputWidth: width, outputHeight: height,
      totalMs: 1, modelPrecision: 'stub', modelLabel: 'Viewer E2E', backend: 'rocm', warmup: false, reused: false };
  }
  async processImage(bitmap, { sequence, mediaTime }) {
    ${slowFrames ? 'await new Promise(resolve => setTimeout(resolve, 1600));' : ''}
    return { sequence, mediaTime, width: ${slowFrames ? '1' : 'bitmap.width'}, height: ${slowFrames ? '1' : 'bitmap.height'}, mime: 'image/png', dataUrl: output };
  }
  async destroy() {}
}
`, 'utf8');
  }
  // manifest.json, background.js, and src/host/native-inline-host.js remain byte-for-byte product copies.
}

async function buildViewerWindowProbeCopy(destination) {
  await mkdir(destination);
  for (const entry of ['src', 'popup', 'viewer', 'assets', 'background.js', 'content-bootstrap.js', 'manifest.json']) {
    await cp(path.join(root, entry), path.join(destination, entry), { recursive: true });
  }
  const backgroundPath = path.join(destination, 'background.js');
  await writeFile(backgroundPath, `${await readFile(backgroundPath, 'utf8')}\nchrome.tabCapture.getMediaStreamId = async () => 'viewer-window-e2e-stream';\n`);
  await writeFile(path.join(destination, 'src/engine/native-engine.js'), `
export class NativeEngine {
  async initialize({ canvas }) { this.canvas = canvas; }
  async updateSettings() {}
  needsReconfigure() { return false; }
  async reconfigure() {}
  async processFrame(video, crop) {
    const width = Math.max(1, Math.round(crop.width * 2));
    const height = Math.max(1, Math.round(crop.height * 2));
    this.canvas.width = width; this.canvas.height = height;
    this.canvas.getContext('2d').drawImage(video, crop.x, crop.y, crop.width, crop.height, 0, 0, width, height);
    return { sourceWidth: crop.width, sourceHeight: crop.height, outputWidth: width, outputHeight: height,
      totalMs: 1, modelPrecision: 'stub', modelLabel: 'Viewer E2E', backend: 'rocm', warmup: false, reused: false };
  }
  async destroy() {}
}
`, 'utf8');
  await writeFile(path.join(destination, 'viewer/viewer-window-e2e-shim.js'), `
const canvas = document.createElement('canvas');
canvas.width = 1280; canvas.height = 720;
const context = canvas.getContext('2d');
const paint = () => {
  context.fillStyle = '#18a84a'; context.fillRect(0, 0, 1280, 720);
  context.fillStyle = '#ffffff'; context.fillRect(24, 24, 80, 40);
};
paint(); setInterval(paint, 40);
navigator.mediaDevices.getUserMedia = async () => canvas.captureStream(24);
`, 'utf8');
  const viewerHtml = path.join(destination, 'viewer/viewer.html');
  const html = await readFile(viewerHtml, 'utf8');
  await writeFile(viewerHtml, html.replace('<script type="module" src="viewer.js"></script>',
    '<script src="viewer-window-e2e-shim.js"></script><script type="module" src="viewer.js"></script>'));
}

const nativeRelayContentProbe = `
(() => {
  let started = false;
  window.addEventListener('native-relay-e2e-start', event => {
    if (started) return;
    started = true;
    const options = event.detail ? JSON.parse(event.detail) : {};
    const sessionId = 'native-relay-e2e-' + crypto.randomUUID();
    const report = { sessionId, messages: [], disconnected: false };
    const publish = () => { document.documentElement.dataset.nativeRelayE2e = JSON.stringify(report); };
    const port = chrome.runtime.connect({ name: 'animescaler-native-inline-v1' });
    port.onMessage.addListener(message => {
      report.messages.push(message);
      if (message?.type === 'ERROR' && message.sessionId === sessionId) {
        report.error = message;
        publish();
      }
      if (message?.type === 'STARTED' && options.frame && !report.frameSent) {
        report.frameSent = true;
        port.postMessage({ protocol: 1, type: 'FRAME_IN', sessionId, ...options.frame });
      }
      if (message?.type === 'FRAME_OUT') report.output = message;
      if (message?.type === 'FRAME_ACK') (report.frameAcks ??= []).push(message);
      if (message?.type === 'STOPPED') report.stopped = true;
      publish();
    });
    port.onDisconnect.addListener(() => { report.disconnected = true; publish(); });
    window.addEventListener('native-relay-e2e-stop', () => {
      port.postMessage({ protocol: 1, type: 'STOP', sessionId, reason: 'chromium-e2e' });
    }, { once: true });
    const request = { protocol: 1, type: 'START', sessionId,
      backend: 'e2e-invalid-backend', model: 'e2e-invalid-model',
      settings: { backend: 'e2e-invalid-backend', model: 'e2e-invalid-model', preset: 'performance' },
      captureMode: 'direct-png', sourceWidth: 320, sourceHeight: 180, ...options.start };
    report.requestKeys = Object.keys(request).sort();
    report.settingsKeys = Object.keys(request.settings).sort();
    port.postMessage(request);
    publish();
  }, { once: true });
  document.documentElement.dataset.nativeRelayProbeReady = 'true';
})();
`;

const nativeRelayHostObserver = `
(() => {
  const portName = 'animescaler-native-inline-v1';
  const snapshot = sender => ({ id: sender?.id, url: sender?.url, origin: sender?.origin,
    tabId: sender?.tab?.id, documentId: sender?.documentId });
  const observe = observation => chrome.runtime.sendMessage({
    type: '__ANIMESCALER_NATIVE_RELAY_E2E_OBSERVATION', observation,
  }).catch(error => console.error('Native relay E2E observation failed:', error.message));
  chrome.runtime.onConnect.addListener(port => {
    if (port.name !== portName) return;
    const sender = snapshot(port.sender);
    let sessionId;
    port.onMessage.addListener(message => {
      if (message?.type !== 'START' || typeof message.sessionId !== 'string' ||
          !message.sessionId.startsWith('native-relay-e2e-')) return;
      sessionId = message.sessionId;
      observe({ kind: 'host-received-start', sessionId, portName, hostUrl: location.href, sender,
        message: { protocol: message.protocol, type: message.type, sessionId, backend: message.backend,
          model: message.model, captureMode: message.captureMode, sourceWidth: message.sourceWidth,
          sourceHeight: message.sourceHeight, messageKeys: Object.keys(message).sort(),
          settingsKeys: Object.keys(message.settings ?? {}).sort() } });
    });
    port.onDisconnect.addListener(() => {
      if (sessionId) observe({ kind: 'host-port-disconnect', sessionId, portName, hostUrl: location.href, sender });
    });
  });
})();
`;

async function eventually(operation, condition, label, timeoutMs = 10000) {
  const deadline = Date.now() + timeoutMs;
  let result;
  let error;
  while (Date.now() < deadline) {
    try { result = await operation(); if (condition(result)) return result; }
    catch (caught) { error = caught; }
    await new Promise(resolve => setTimeout(resolve, 50));
  }
  assert.fail(`${label}: ${error?.message ?? JSON.stringify(result) ?? 'gate unavailable'}`);
}

async function withExtension(t, derivedProbe, run) {
  assert.ok(existsSync(executablePath), `CHROMIUM_EXECUTABLE_MISSING: ${executablePath}`);
  const browserApiProbe = derivedProbe === true;
  const nativeRelayProbe = ['nativeRelay', 'nativeRelayFrames', 'nativeRelaySlowFrames'].includes(derivedProbe);
  const nativeRelayFramesProbe = derivedProbe === 'nativeRelayFrames' || derivedProbe === 'nativeRelaySlowFrames';
  const nativeRelaySlowFramesProbe = derivedProbe === 'nativeRelaySlowFrames';
  const viewerWindowProbe = derivedProbe === 'viewerWindow';
  const temporaryExtension = browserApiProbe || nativeRelayProbe || viewerWindowProbe;
  const temporary = await mkdtemp(path.join(tmpdir(), 'animescaler-chromium151-'));
  const userDataDir = path.join(temporary, 'profile');
  const extensionDir = temporaryExtension
    ? path.join(temporary, viewerWindowProbe ? 'viewer-window-probe' : nativeRelayProbe ? 'native-relay-probe' : 'extension-probe') : root;
  const reportPath = path.join(temporary, 'evidence.json');
  const report = { testName: t.name, classification: browserApiProbe ? 'Chromium Browser API probe; NOT AnimeScaler relay'
      : viewerWindowProbe ? 'Chromium viewer-window lifecycle with synthetic capture and NativeEngine'
        : nativeRelayProbe ? 'Chromium production NativeInline relay with passive temp-copy probes' : 'Chromium original AnimeScaler extension',
    executablePath, userDataDir, extensionDir, checks: [], console: [], workerConsole: [], pageErrors: [], webErrors: [],
    limitations: ['No inference, model loading, native service or GPU validation.',
      nativeRelayProbe ? 'Relay test uses an invalid backend to stop before engine initialization.'
        : 'No production offscreen host or native relay integration.', 'Brave not executed.',
      'Worker console hooks attach immediately after launch returns; logs before Worker object availability are not guaranteed.',
      'Popup HTML opened as extension page; action bubble and user gesture permissions are not proved.'] };
  let context;
  let server;
  let failure;
  try {
    if (browserApiProbe) await buildProbeCopy(extensionDir);
    else if (viewerWindowProbe) await buildViewerWindowProbeCopy(extensionDir);
    else if (nativeRelayProbe) await buildNativeRelayProbeCopy(extensionDir, {
      stubEngine: nativeRelayFramesProbe, slowFrames: nativeRelaySlowFramesProbe,
    });
    server = createServer((request, response) => {
      if (request.url === '/favicon.ico') { response.writeHead(204); response.end(); return; }
      if (request.url !== '/fixture') { response.writeHead(404); response.end(); return; }
      response.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
      response.end('<!doctype html><html><head><title>AnimeScaler runtime fixture</title></head><body><video id="source" width="320" height="180" controls></video></body></html>');
    });
    await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
    const fixtureUrl = `http://127.0.0.1:${server.address().port}/fixture`;
    report.fixtureUrl = fixtureUrl;
    context = await chromium.launchPersistentContext(userDataDir, {
      executablePath, headless: true, timeout: 20000, ignoreDefaultArgs: ['--disable-extensions'],
      args: [`--disable-extensions-except=${extensionDir}`, `--load-extension=${extensionDir}`, '--disable-gpu'],
    });
    const watchedWorkers = new WeakSet();
    const watchWorker = worker => {
      if (watchedWorkers.has(worker)) return;
      watchedWorkers.add(worker);
      worker.on('console', message => report.workerConsole.push({ workerUrl: worker.url(), type: message.type(), text: message.text() }));
    };
    context.on('serviceworker', watchWorker);
    for (const worker of context.serviceWorkers()) watchWorker(worker);
    context.on('console', message => report.console.push({ type: message.type(), text: message.text(), url: message.location().url }));
    context.on('weberror', event => report.webErrors.push(event.error().message));
    const watchPage = page => page.on('pageerror', error => report.pageErrors.push({ url: page.url(), message: error.message }));
    context.on('page', watchPage);
    for (const page of context.pages()) watchPage(page);
    const workerPath = browserApiProbe ? '/__browser-api-probe-worker.mjs' : '/background.js';
    const worker = context.serviceWorkers().find(item => item.url().endsWith(workerPath))
      ?? await context.waitForEvent('serviceworker', { predicate: item => item.url().endsWith(workerPath), timeout: 15000 });
    watchWorker(worker);
    const browser = context.browser();
    assert.ok(browser, 'persistent context must expose actual browser');
    report.version = browser.version();
    assert.match(report.version, /^151\./, 'CHROMIUM_VERSION_MISMATCH');
    const extensionId = new URL(worker.url()).hostname;
    report.workerUrl = worker.url();
    report.extensionId = extensionId;
    await worker.evaluate(() => console.info('AnimeScaler Chromium E2E worker capture ready'));
    const page = await context.newPage();
    await page.goto(fixtureUrl);
    const tabId = await eventually(() => worker.evaluate(async url => {
      const [tab] = await chrome.tabs.query({ url }); return tab?.id;
    }, fixtureUrl), Number.isInteger, 'loopback tab not registered');
    const send = message => worker.evaluate(({ tabId, message }) => chrome.tabs.sendMessage(tabId, message, { frameId: 0 }), { tabId, message });
    const productProbe = await eventually(() => send({ type: 'ANIMESCALER_PROBE' }), reply => reply?.hasVideo === true, 'content bootstrap/PROBE unavailable');
    assert.equal(productProbe.state, 'stopped');
    report.checks.push({ name: 'existing SW to content PROBE', result: productProbe });
    const popup = await context.newPage();
    await popup.goto(`chrome-extension://${extensionId}/popup/popup.html`);
    await popup.locator('#toggle').waitFor();
    assert.equal(await popup.title(), 'AnimeScaler');
    assert.equal(await popup.evaluate(() => chrome.runtime.id), extensionId);
    report.checks.push({ name: 'real popup HTML/context', passed: true });
    await run({ context, worker, page, popup, send, tabId, fixtureUrl, extensionId, report });
    await eventually(() => Promise.resolve(report.workerConsole), rows => rows.some(row => row.text.includes('worker capture ready')), 'worker console capture unavailable');
    assert.equal(report.workerConsole.filter(row => row.text === 'AnimeScaler Chromium E2E worker capture ready').length, 1,
      'post-hook marker must be captured exactly once; duplicate listeners are forbidden');
    if (browserApiProbe) {
      const marker = await worker.evaluate(() => globalThis.__browserApiProbeStartupMarker);
      assert.equal(marker, 'Browser API probe worker startup', 'wrapper startup did not execute');
      report.startupConsole = { marker, captured: report.workerConsole.some(row => row.text === marker),
        limit: 'Startup may execute before launch resolves or the Worker hook attaches; no claim of pre-availability log capture.' };
    }
    assert.deepEqual(report.pageErrors, [], 'pageerror captured during E2E');
    assert.deepEqual(report.webErrors, [], 'context error captured during E2E');
    assert.deepEqual(report.console.filter(row => row.type === 'error'), [], 'console errors captured during E2E');
    assert.deepEqual(report.workerConsole.filter(row => row.type === 'error'), [], 'worker console errors captured during E2E');
    report.state = 'passed';
  } catch (error) {
    failure = error;
    report.state = 'failed';
    report.failure = { message: error.message, stack: error.stack };
  } finally {
    try { await context?.close(); } catch (error) { report.cleanupError = error.message; failure ??= error; }
    if (server) await new Promise(resolve => { server.close(resolve); server.closeAllConnections(); });
    if (browserApiProbe) report.startupConsole ??= {
      marker: 'Browser API probe worker startup',
      captured: report.workerConsole.some(row => row.text === 'Browser API probe worker startup'),
      limit: 'Startup may execute before launch resolves or the Worker hook attaches; no claim of pre-availability log capture.',
    };
    // Every removal targets a child of this newly created temp directory.
    for (const directory of [userDataDir, ...(temporaryExtension ? [extensionDir] : [])]) {
      assert.equal(path.dirname(path.resolve(directory)), path.resolve(temporary));
      try { await rm(directory, { recursive: true, force: true }); }
      catch (error) { report.cleanupError = error.message; failure ??= error; }
    }
    report.cleanup = { profileRemoved: !existsSync(userDataDir),
      probeCopyRemoved: browserApiProbe ? !existsSync(extensionDir) : null,
      nativeRelayCopyRemoved: nativeRelayProbe ? !existsSync(extensionDir) : null };
    if (!failure) {
      try {
        assert.equal(report.cleanup.profileRemoved, true);
        if (browserApiProbe) assert.equal(report.cleanup.probeCopyRemoved, true);
        if (nativeRelayProbe) assert.equal(report.cleanup.nativeRelayCopyRemoved, true);
        assert.deepEqual(report.pageErrors, []);
        assert.deepEqual(report.webErrors, []);
        assert.deepEqual(report.console.filter(row => row.type === 'error'), []);
        assert.deepEqual(report.workerConsole.filter(row => row.type === 'error'), []);
      } catch (error) { failure = error; }
    }
    if (failure) { report.state = 'failed'; report.failure = { message: failure.message, stack: failure.stack }; }
    await writeFile(reportPath, JSON.stringify(report, null, 2));
    t.diagnostic(`Evidence: ${reportPath}`);
  }
  if (failure) throw failure;
}

test('Chromium 151 original extension: MV3, popup and existing bidirectional messages', { timeout: 60000 }, async t => {
  await withExtension(t, false, async ({ worker, page, popup, send, report }) => {
    const status = await send({ type: 'ANIMESCALER_STATUS' });
    assert.equal(status.state, 'stopped');
    const stopped = await send({ type: 'ANIMESCALER_STOP' });
    assert.equal(stopped.state, 'stopped');
    // No decoded video is supplied; this error path must fail before creating any engine.
    await page.locator('video').evaluate(video => video.remove());
    const blocked = await send({ type: 'ANIMESCALER_START', settings: { backend: 'rocm', model: 'zenkai-1-0' } });
    assert.equal(blocked.state, 'error');
    assert.equal(blocked.error.code, 'NO_VIDEO');
    const history = await eventually(() => worker.evaluate(async () => (await chrome.storage.local.get('animeScalerDiagnostics')).animeScalerDiagnostics),
      value => value?.events?.some(event => event.type === 'error' && event.data?.code === 'NO_VIDEO'), 'existing content to background diagnostic unavailable');
    const exported = await popup.evaluate(() => chrome.runtime.sendMessage({ type: 'ANIMESCALER_DIAGNOSTIC_EXPORT' }));
    assert.equal(exported.ok, true);
    assert.ok(exported.data.events.some(event => event.data?.code === 'NO_VIDEO'));
    report.checks.push({ name: 'existing content to SW diagnostic', history }, { name: 'popup to SW diagnostic export', reply: exported });
    report.limitations.push('Product native Port relay absent; BACKEND_HOST_UNAVAILABLE is unchanged.',
      'Fixture video has no decoded media. NO_VIDEO proves messaging/error handling, not playback.');
  });
});

test('Chromium 151 popup and viewer expose visible keyboard focus in native tab order', { timeout: 60000 }, async t => {
  await withExtension(t, false, async ({ popup, extensionId, report }) => {
    await popup.keyboard.press('Tab');
    const popupFirst = await popup.evaluate(() => ({ tag: document.activeElement.tagName, id: document.activeElement.id,
      className: document.activeElement.className, disabled: document.activeElement.disabled,
      text: document.activeElement.textContent.trim(),
      outline: getComputedStyle(document.activeElement).outlineStyle,
      outlineWidth: getComputedStyle(document.activeElement).outlineWidth }));
    assert.equal(popupFirst.tag, 'BUTTON');
    assert.equal(popupFirst.className, 'profile-btn active');
    assert.equal(popupFirst.disabled, false);
    assert.equal(popupFirst.outline, 'solid');
    assert.ok(Number.parseFloat(popupFirst.outlineWidth) >= 2);
    await popup.keyboard.press('Tab');
    assert.equal(await popup.evaluate(() => document.activeElement.dataset.preset), 'balanced');
    await popup.keyboard.press('Tab');
    assert.equal(await popup.evaluate(() => document.activeElement.dataset.preset), 'quality');

    await popup.goto(`chrome-extension://${extensionId}/viewer/viewer.html?session=keyboard-focus-probe`);
    await popup.locator('#back').waitFor();
    await popup.keyboard.press('Tab');
    const viewerFirst = await popup.evaluate(() => ({ id: document.activeElement.id,
      outline: getComputedStyle(document.activeElement).outlineStyle,
      outlineWidth: getComputedStyle(document.activeElement).outlineWidth }));
    assert.equal(viewerFirst.id, 'back');
    assert.equal(viewerFirst.outline, 'solid');
    assert.ok(Number.parseFloat(viewerFirst.outlineWidth) >= 2);
    await popup.keyboard.press('Tab');
    assert.equal(await popup.evaluate(() => document.activeElement.id), 'play');
    report.checks.push({ name: 'popup keyboard focus and native tab order', first: popupFirst, nextPresets: ['balanced', 'quality'] },
      { name: 'viewer keyboard focus and native tab order', first: viewerFirst, next: 'play' });
    report.limitations.push('Viewer opens without a capture session; this proves controls and focus behavior, not playback.');
  });
});

test('Chromium 151 derived fixture: actual sender/receiver and Ports, NOT native relay', { timeout: 60000 }, async t => {
  await withExtension(t, true, async ({ worker, page, popup, fixtureUrl, extensionId, tabId, report }) => {
    const reply = await popup.evaluate(async () => {
      try { return await chrome.runtime.sendMessage({ type: 'BROWSER_API_PROBE_PING', nonce: 'popup-message' }); }
      catch (error) { return { unavailable: error.message }; }
    });
    assert.equal(reply?.kind, 'browser-api-probe', 'Browser API probe receiver missing; do not substitute a mock');
    assert.equal(reply.sender.id, extensionId);
    assert.equal(reply.sender.url, `chrome-extension://${extensionId}/popup/popup.html`);
    assert.equal(reply.receiver.id, extensionId);
    report.checks.push({ name: 'actual popup sender/worker receiver', reply });
    await page.waitForFunction(() => document.documentElement.dataset.browserApiProbeMessage, undefined, { timeout: 10000 });
    const contentMessage = await page.evaluate(() => JSON.parse(document.documentElement.dataset.browserApiProbeMessage));
    assert.equal(contentMessage.nonce, 'content-message');
    assert.equal(contentMessage.sender.url, fixtureUrl);
    assert.equal(contentMessage.sender.id, extensionId);
    assert.equal(contentMessage.receiver.url, worker.url());
    const fromWorker = await worker.evaluate(tabId => chrome.tabs.sendMessage(tabId,
      { type: 'BROWSER_API_CONTENT_PING', nonce: 'worker-message' }, { frameId: 0 }), tabId);
    assert.equal(fromWorker.nonce, 'worker-message');
    assert.equal(fromWorker.sender.id, extensionId);
    assert.equal(fromWorker.receiver.url, fixtureUrl);
    assert.equal(fromWorker.receiver.id, extensionId);
    report.checks.push({ name: 'actual content message sender/worker receiver', reply: contentMessage },
      { name: 'actual worker message sender/content receiver', reply: fromWorker });
    await page.waitForFunction(() => document.documentElement.dataset.browserApiProbePort, undefined, { timeout: 10000 });
    const contentReply = await page.evaluate(() => JSON.parse(document.documentElement.dataset.browserApiProbePort));
    assert.equal(contentReply.nonce, 'content-port');
    assert.equal(contentReply.sender.id, extensionId);
    assert.equal(contentReply.sender.url, fixtureUrl);
    assert.equal(contentReply.sender.frameId, 0);
    assert.ok(Number.isInteger(contentReply.sender.tabId));
    assert.equal(contentReply.receiver.url, worker.url());
    const popupPort = await popup.evaluate(portName => new Promise((resolve, reject) => {
      const port = chrome.runtime.connect({ name: portName });
      const timer = setTimeout(() => { port.disconnect(); reject(new Error('Port receiver timeout')); }, 5000);
      port.onDisconnect.addListener(() => { clearTimeout(timer); if (chrome.runtime.lastError) reject(new Error(chrome.runtime.lastError.message)); });
      port.onMessage.addListener(message => { clearTimeout(timer); resolve(message); port.disconnect(); });
      port.postMessage({ nonce: 'popup-port' });
    }), probePortName);
    assert.equal(popupPort.nonce, 'popup-port');
    assert.equal(popupPort.sender.url, `chrome-extension://${extensionId}/popup/popup.html`);
    assert.equal(popupPort.sender.id, extensionId);
    assert.ok(Number.isInteger(popupPort.sender.tabId), 'extension page in a tab has tab sender; not offscreen');
    await page.close();
    const events = await eventually(() => worker.evaluate(() => globalThis.__browserApiProbeEvents),
      rows => rows?.filter(row => row.type === 'disconnect').length === 2, 'real Port disconnect events missing');
    assert.equal(events.filter(row => row.type === 'connect').length, 2);
    assert.equal(events.filter(row => row.type === 'port-message').length, 2);
    assert.equal(events.filter(row => row.type === 'connect').every(row => row.name === probePortName), true);
    report.checks.push({ name: 'actual content Port round trip', reply: contentReply },
      { name: 'actual popup Port round trip', reply: popupPort }, { name: 'real Port lifecycle', events });
    report.limitations.push('Probe listeners/scripts exist only in temp extension copy; product runtime.connect/onConnect are not implemented.',
      'Observed tab senders do not prove service-worker/offscreen sender topology or approve NativeInline host validation.');
  });
});

test('Chromium 151 Browser API probe: SW/offscreen messages and both Port initiation directions, NOT product/content relay', { timeout: 60000 }, async t => {
  await withExtension(t, true, async ({ worker, extensionId, report }) => {
    const capabilities = await worker.evaluate(() => ({
      createDocument: typeof chrome.offscreen?.createDocument,
      closeDocument: typeof chrome.offscreen?.closeDocument,
      getContexts: typeof chrome.runtime.getContexts,
    }));
    report.checks.push({ name: 'real offscreen API gates', capabilities });
    assert.equal(capabilities.createDocument, 'function', 'OFFSCREEN_API_UNAVAILABLE');
    assert.equal(capabilities.closeDocument, 'function', 'OFFSCREEN_CLOSE_UNAVAILABLE');
    assert.equal(capabilities.getContexts, 'function', 'RUNTIME_CONTEXTS_UNAVAILABLE');
    const documentPath = '__browser-api-probe-offscreen.html';
    const documentUrl = `chrome-extension://${extensionId}/${documentPath}`;
    const contexts = () => worker.evaluate(() => chrome.runtime.getContexts({ contextTypes: ['OFFSCREEN_DOCUMENT'] }));
    assert.deepEqual(await contexts(), [], 'unexpected pre-existing offscreen context');
    let created = false;
    try {
      await worker.evaluate(url => chrome.offscreen.createDocument({ url, reasons: ['DOM_PARSER'],
        justification: 'Disposable Browser API probe parses a fixture document and verifies extension messaging.' }), documentPath);
      created = true;
      const active = await eventually(contexts, rows => rows.length === 1, 'OFFSCREEN_DOCUMENT not observed');
      report.checks.push({ name: 'runtime.getContexts OFFSCREEN_DOCUMENT', contexts: active });
      assert.equal(active[0].contextType, 'OFFSCREEN_DOCUMENT');
      assert.equal(active[0].documentUrl, documentUrl);
      assert.ok(active[0].documentId, 'offscreen context has no documentId');
      const reply = await worker.evaluate(() => chrome.runtime.sendMessage({ type: 'BROWSER_API_OFFSCREEN_PING', nonce: 'sw-offscreen-message' }));
      report.checks.push({ name: 'real SW to offscreen message round trip', reply,
        senderFields: observedSenderFields(reply?.sender ?? {}) });
      assert.equal(reply?.kind, 'browser-api-offscreen');
      assert.equal(reply.nonce, 'sw-offscreen-message');
      assert.equal(reply.sender.id, extensionId);
      assert.equal(reply.sender.url, worker.url(), 'offscreen must observe actual SW sender URL');
      assertAbsentSenderFields(reply.sender);
      assert.equal(reply.receiver.id, extensionId);
      assert.equal(reply.receiver.url, documentUrl);
      assert.equal(reply.parsed, true);
      const received = await eventually(() => worker.evaluate(() => globalThis.__browserApiProbeEvents),
        rows => rows?.some(row => row.type === 'message' && row.nonce === 'offscreen-port-received'), 'offscreen did not receive SW Port reply');
      const receipt = received.find(row => row.type === 'message' && row.nonce === 'offscreen-port-received');
      report.checks.push({ name: 'real offscreen Port received SW reply', receipt,
        messageSenderFields: observedSenderFields(receipt.sender),
        portSenderFields: observedSenderFields(receipt.portReply?.sender ?? {}) });
      assert.equal(receipt.sender.id, extensionId);
      assert.equal(receipt.sender.url, documentUrl);
      assertAbsentSenderFields(receipt.sender);
      assert.equal(receipt.portReply.nonce, 'offscreen-port');
      assert.equal(receipt.portReply.sender.id, extensionId);
      assert.equal(receipt.portReply.sender.url, documentUrl);
      assertAbsentSenderFields(receipt.portReply.sender);
      assert.equal(receipt.portReply.receiver.url, worker.url());
      assert.equal(receipt.portReply.receiver.id, extensionId);
      for (const nonce of ['sw-port-explicit-close', 'sw-port-document-close']) {
        const portReply = await connectWorkerToOffscreen(worker, nonce);
        report.checks.push({ name: 'actual SW-initiated Port round trip at offscreen onConnect', nonce, portReply,
          senderFields: observedSenderFields(portReply?.sender ?? {}) });
        assert.equal(portReply.kind, 'browser-api-sw-offscreen-port');
        assert.equal(portReply.nonce, nonce);
        assert.equal(portReply.name, swOffscreenPortName);
        assert.equal(portReply.sender.id, extensionId);
        assert.equal(portReply.sender.url, worker.url());
        assertAbsentSenderFields(portReply.sender);
        assert.equal(portReply.receiver.id, extensionId);
        assert.equal(portReply.receiver.url, documentUrl, 'reply must come from offscreen, not another receiver');
        const traces = await eventually(() => worker.evaluate(() => globalThis.__browserApiProbeEvents),
          rows => rows?.some(row => row.type === 'offscreen-port-event' && row.event.type === 'connect' &&
            row.event.connection === portReply.connection), 'offscreen runtime.onConnect not observed');
        const connected = traces.find(row => row.type === 'offscreen-port-event' && row.event.type === 'connect' &&
          row.event.connection === portReply.connection);
        assert.equal(connected.sender.url, documentUrl, 'onConnect trace reporter must be offscreen');
        assert.equal(connected.sender.id, extensionId);
        assert.equal(connected.event.sender.url, worker.url());
        assert.equal(connected.event.sender.id, extensionId);
        assertAbsentSenderFields(connected.event.sender);
        assert.equal(connected.event.name, swOffscreenPortName);
        report.checks.push({ name: 'actual offscreen onConnect sender/receiver', nonce, connected,
          senderFields: observedSenderFields(connected.event.sender) });
        if (nonce === 'sw-port-explicit-close') {
          await worker.evaluate(nonce => globalThis.__browserApiSwPorts[nonce].disconnect(), nonce);
          const disconnected = await eventually(() => worker.evaluate(() => globalThis.__browserApiProbeEvents),
            rows => rows?.some(row => row.type === 'offscreen-port-event' && row.event.type === 'disconnect' &&
              row.event.connection === portReply.connection), 'offscreen did not observe SW Port.disconnect');
          report.checks.push({ name: 'offscreen receives disconnect after SW closes Port', nonce,
            event: disconnected.find(row => row.type === 'offscreen-port-event' && row.event.type === 'disconnect' &&
              row.event.connection === portReply.connection) });
        }
      }
      await worker.evaluate(() => chrome.offscreen.closeDocument());
      created = false;
      const afterClose = await eventually(contexts, rows => rows.length === 0, 'offscreen context survived closeDocument');
      const events = await eventually(() => worker.evaluate(() => globalThis.__browserApiProbeEvents),
        rows => rows?.some(row => row.type === 'disconnect' && row.sender.url === documentUrl), 'offscreen Port disconnect not observed');
      const offscreenEvents = events.filter(row => row.sender?.url === documentUrl);
      assert.equal(offscreenEvents.filter(row => row.type === 'connect').length, 1);
      assert.equal(offscreenEvents.filter(row => row.type === 'port-message').length, 1);
      assert.equal(offscreenEvents.filter(row => row.type === 'disconnect').length, 1);
      assert.equal(offscreenEvents.filter(row => ['connect', 'port-message', 'disconnect'].includes(row.type))
        .every(row => row.name === offscreenPortName), true);
      report.checks.push({ name: 'closeDocument removes context and disconnects real Port', contexts: afterClose, events: offscreenEvents });
      const swEvents = await eventually(() => worker.evaluate(() => globalThis.__browserApiSwPortEvents),
        rows => rows?.some(row => row.type === 'sw-port-disconnect' && row.nonce === 'sw-port-document-close'),
        'SW did not observe Port disconnect after closeDocument');
      assert.equal(swEvents.filter(row => row.type === 'sw-connect-called').length, 2);
      assert.equal(swEvents.filter(row => row.type === 'sw-port-message').length, 2);
      report.checks.push({ name: 'SW-initiated Port disconnect after offscreen closeDocument', events: swEvents });
      report.limitations.push('Offscreen permission/HTML/script exist ONLY in the disposable Browser API fixture.',
        'This proves direct SW/offscreen APIs, not production NativeInline relay or content-to-host relay topology.',
        'MessageSender optional fields are observed or absent. OFFSCREEN_DOCUMENT context documentId is not substituted into sender metadata. Identity gates require exact sender.id and sender.url.');
    } finally {
      report.swInitiatedPortTrace = await worker.evaluate(() => globalThis.__browserApiSwPortEvents ?? []);
      if (created) {
        await worker.evaluate(() => chrome.offscreen.closeDocument());
        const remaining = await eventually(contexts, rows => rows.length === 0, 'offscreen cleanup failed');
        const events = await eventually(() => worker.evaluate(() => globalThis.__browserApiProbeEvents),
          rows => rows?.some(row => row.type === 'disconnect' && row.sender.url === documentUrl), 'offscreen cleanup Port disconnect missing');
        report.checks.push({ name: 'closeDocument cleanup after failed gate', contexts: remaining,
          events: events.filter(row => row.sender?.url === documentUrl) });
      }
    }
  });
});

test('Chromium 151 NativeInline rejects secret-bearing START before offscreen host', { timeout: 60000 }, async t => {
  await withExtension(t, 'nativeRelay', async ({ worker, page, report }) => {
    const initial = await worker.evaluate(() => chrome.runtime.getContexts({ contextTypes: ['OFFSCREEN_DOCUMENT'] }));
    assert.deepEqual(initial, []);
    await worker.evaluate(() => {
      globalThis.__nativeRelayE2eEvents = [];
      chrome.runtime.onMessage.addListener((message, sender, respond) => {
        if (message?.type !== '__ANIMESCALER_NATIVE_RELAY_E2E_OBSERVATION') return;
        globalThis.__nativeRelayE2eEvents.push({ observation: message.observation, sender: { id: sender?.id, url: sender?.url } });
        respond({ observed: true });
      });
    });
    const secrets = { start: { streamId: 'synthetic-stream-id', token: 'synthetic-token',
      settings: { backend: 'e2e-invalid-backend', model: 'e2e-invalid-model', preset: 'performance', token: 'nested-secret' } } };
    await page.waitForFunction(() => document.documentElement.dataset.nativeRelayProbeReady === 'true');
    await page.evaluate(value => window.dispatchEvent(new CustomEvent('native-relay-e2e-start',
      { detail: JSON.stringify(value) })), secrets);
    await page.waitForFunction(() => {
      const value = document.documentElement.dataset.nativeRelayE2e;
      if (!value) return false;
      const result = JSON.parse(value);
      return result.error?.type === 'ERROR' && result.disconnected;
    }, undefined, { timeout: 15000 });
    const client = await page.evaluate(() => JSON.parse(document.documentElement.dataset.nativeRelayE2e));
    assert.equal(client.error.code, 'SOURCE_NOT_ISOLATED');
    assert.equal(client.error.sessionId, client.sessionId);
    const hostEvents = await worker.evaluate(() => globalThis.__nativeRelayE2eEvents ?? []);
    assert.equal(hostEvents.some(event => event.observation?.kind === 'host-received-start'), false,
      'service worker must reject secret-bearing input before forwarding it');
    const contexts = await eventually(() => worker.evaluate(() => chrome.runtime.getContexts({ contextTypes: ['OFFSCREEN_DOCUMENT'] })),
      rows => rows.length === 0, 'offscreen context survived rejected secret-bearing START');
    report.checks.push({ name: 'secret-bearing START rejected before host relay', error: client.error,
      forbiddenKeys: ['streamId', 'token', 'settings.token'], hostEvents, contexts });
    report.limitations.push('Secrets are synthetic test values; no real stream ID or service token was used.');
  });
});

test('Chromium 151 production relay round-trips PNG and STOP with temporary engine stub', { timeout: 60000 }, async t => {
  await withExtension(t, 'nativeRelayFrames', async ({ worker, page, report }) => {
    const png = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII=';
    const options = {
      start: { backend: 'rocm', model: 'zenkai-1-0', captureMode: 'direct-png', sourceWidth: 1, sourceHeight: 1,
        settings: { backend: 'rocm', model: 'zenkai-1-0', preset: 'performance' } },
      frame: { sequence: 1, mediaTime: 0.5, width: 1, height: 1, mime: 'image/png', dataUrl: png },
    };
    await page.waitForFunction(() => document.documentElement.dataset.nativeRelayProbeReady === 'true');
    await page.evaluate(value => window.dispatchEvent(new CustomEvent('native-relay-e2e-start',
      { detail: JSON.stringify(value) })), options);
    await page.waitForFunction(() => {
      const value = document.documentElement.dataset.nativeRelayE2e;
      if (!value) return false;
      const state = JSON.parse(value);
      return state.output && state.frameAcks?.some(ack => ack.sequence === 1 && ack.accepted === true);
    }, undefined, { timeout: 20000 });
    const client = await page.evaluate(() => JSON.parse(document.documentElement.dataset.nativeRelayE2e));
    assert.equal(client.error, undefined);
    assert.equal(client.output.sequence, 1);
    assert.equal(client.output.mediaTime, 0.5);
    assert.equal(client.output.width, 1);
    assert.equal(client.output.height, 1);
    assert.equal(client.output.mime, 'image/png');
    assert.equal(client.output.dataUrl, png, 'PNG must cross host and return unchanged');
    assert.deepEqual(client.messages.map(message => message.type), ['STARTED', 'FRAME_OUT', 'FRAME_ACK']);
    await page.evaluate(() => window.dispatchEvent(new Event('native-relay-e2e-stop')));
    await page.waitForFunction(() => {
      const value = document.documentElement.dataset.nativeRelayE2e;
      if (!value) return false;
      const state = JSON.parse(value);
      return state.stopped && state.disconnected;
    }, undefined, { timeout: 15000 });
    const stopped = await page.evaluate(() => JSON.parse(document.documentElement.dataset.nativeRelayE2e));
    assert.equal(stopped.messages.at(-1).type, 'STOPPED');
    const contexts = await eventually(() => worker.evaluate(() => chrome.runtime.getContexts({ contextTypes: ['OFFSCREEN_DOCUMENT'] })),
      rows => rows.length === 0, 'offscreen document survived STOP');
    report.checks.push({ name: 'production relay returned PNG and ACK', sequence: client.output.sequence,
      mediaTime: client.output.mediaTime, dimensions: [client.output.width, client.output.height],
      mime: client.output.mime, pngMatchesInput: client.output.dataUrl === png,
      messages: client.messages.map(message => message.type) },
      { name: 'production STOP cleanup', messages: stopped.messages.map(message => message.type), contexts });
    report.limitations.push('NativeEngine is replaced only in the temporary extension copy with a deterministic PNG stub; production service, Worker, and GPU are not run.');
  });
});

test('Chromium 151 closing the source tab closes its production relay and offscreen host', { timeout: 60000 }, async t => {
  await withExtension(t, 'nativeRelayFrames', async ({ worker, page, report }) => {
    await page.waitForFunction(() => document.documentElement.dataset.nativeRelayProbeReady === 'true');
    await page.evaluate(value => window.dispatchEvent(new CustomEvent('native-relay-e2e-start',
      { detail: JSON.stringify(value) })), {
      start: { backend: 'rocm', model: 'zenkai-1-0', captureMode: 'direct-png', sourceWidth: 1, sourceHeight: 1,
        settings: { backend: 'rocm', model: 'zenkai-1-0', preset: 'performance' } },
    });
    await page.waitForFunction(() => {
      const value = document.documentElement.dataset.nativeRelayE2e;
      return value && JSON.parse(value).messages.some(message => message.type === 'STARTED');
    }, undefined, { timeout: 15000 });
    const activeContexts = await eventually(() => worker.evaluate(() =>
      chrome.runtime.getContexts({ contextTypes: ['OFFSCREEN_DOCUMENT'] })),
      contexts => contexts.length === 1, 'production offscreen document did not start');
    await page.close();
    const closedContexts = await eventually(() => worker.evaluate(() =>
      chrome.runtime.getContexts({ contextTypes: ['OFFSCREEN_DOCUMENT'] })),
      contexts => contexts.length === 0, 'closing source tab left production offscreen document running');
    report.checks.push({ name: 'source tab close disconnects content Port and cleans offscreen host',
      contextsBeforeTabClose: activeContexts.length, contextsAfterTabClose: closedContexts.length });
    report.limitations.push('Temporary extension copy uses a deterministic NativeEngine stub; native service and GPU are not run.');
  });
});

test('Chromium 151 product controller presents native backend output inline with temporary engine stub', { timeout: 60000 }, async t => {
  await withExtension(t, 'nativeRelayFrames', async ({ worker, page, send, report }) => {
    const source = await page.evaluate(async () => {
      const video = document.querySelector('video');
      const sourceCanvas = document.createElement('canvas');
      sourceCanvas.width = 1;
      sourceCanvas.height = 1;
      const context = sourceCanvas.getContext('2d');
      let tick = 0;
      const paint = () => { context.fillStyle = tick++ % 2 ? '#ff0000' : '#00ff00'; context.fillRect(0, 0, 1, 1); };
      paint();
      const timer = setInterval(paint, 100);
      video.muted = true;
      video.controls = true;
      video.srcObject = sourceCanvas.captureStream(24);
      await video.play();
      await new Promise(resolve => video.requestVideoFrameCallback(resolve));
      return { width: video.videoWidth, height: video.videoHeight, readyState: video.readyState,
        timerStarted: Boolean(timer) };
    });
    assert.deepEqual({ width: source.width, height: source.height }, { width: 1, height: 1 });
    const settings = { backend: 'rocm', model: 'zenkai-1-0', preset: 'performance', enabled: true,
      adaptive: false, maxFps: 24, showHud: false };
    const started = await send({ type: 'ANIMESCALER_START', settings });
    assert.equal(started.state, 'running');
    const active = await eventually(() => send({ type: 'ANIMESCALER_STATUS' }),
      value => value.state === 'running' && value.stats?.outputWidth === 1, 'native inline output was not presented');
    const reconfigured = await send({ type: 'ANIMESCALER_SETTINGS', settings: { ...settings, backend: 'tensorrt' } });
    assert.equal(reconfigured.state, 'running');
    assert.equal(reconfigured.settings.backend, 'tensorrt');
    const afterConfigure = await eventually(() => send({ type: 'ANIMESCALER_STATUS' }),
      value => value.state === 'running' && value.stats?.backend === 'tensorrt', 'native backend reconfiguration stopped inline frames');
    const overlay = await page.evaluate(() => {
      const canvas = document.querySelector('canvas[data-animescaler="canvas"]');
      return canvas && { connected: canvas.isConnected, display: canvas.style.display,
        width: canvas.width, height: canvas.height, encoded: canvas.toDataURL('image/png').startsWith('data:image/png;base64,') };
    });
    assert.deepEqual(overlay, { connected: true, display: 'block', width: 1, height: 1, encoded: true });
    const contextsBeforeStop = await worker.evaluate(() => chrome.runtime.getContexts({ contextTypes: ['OFFSCREEN_DOCUMENT'] }));
    assert.equal(contextsBeforeStop.length, 1);
    const stopped = await send({ type: 'ANIMESCALER_STOP' });
    assert.equal(stopped.state, 'stopped');
    const contextsAfterStop = await eventually(() => worker.evaluate(() => chrome.runtime.getContexts({ contextTypes: ['OFFSCREEN_DOCUMENT'] })),
      rows => rows.length === 0, 'product Stop left offscreen document running');
    const removedOverlay = await page.locator('canvas[data-animescaler="canvas"]').count();
    assert.equal(removedOverlay, 0);
    report.checks.push({ name: 'product direct video capture and native inline presentation', source,
      initialBackend: active.settings.backend, reconfiguredBackend: afterConfigure.settings.backend, stats: afterConfigure.stats,
      overlay, offscreenContextsDuringRun: contextsBeforeStop.length, offscreenContextsAfterStop: contextsAfterStop.length,
      overlayRemovedAfterStop: removedOverlay === 0 });
    report.limitations.push('Temporary extension copy substitutes deterministic NativeEngine output; no local ROCm/TensorRT/D3D12 service or GPU is used.');
  });
});

test('Chromium 151 opens native viewer window, captures the source tab and stops cleanly', { timeout: 60000 }, async t => {
  await withExtension(t, 'viewerWindow', async ({ context, worker, page, popup, tabId, fixtureUrl, report }) => {
    const source = await page.evaluate(async () => {
      const video = document.querySelector('video');
      const canvas = document.createElement('canvas');
      canvas.width = 320; canvas.height = 180;
      const paint = () => {
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#18a84a'; ctx.fillRect(0, 0, 320, 180);
        ctx.fillStyle = '#ffffff'; ctx.fillRect(24, 24, 80, 40);
      };
      paint(); setInterval(paint, 40);
      video.width = 320; video.height = 180; video.muted = true;
      video.srcObject = canvas.captureStream(24); await video.play();
      await new Promise(resolve => video.requestVideoFrameCallback(resolve));
      return { width: video.videoWidth, height: video.videoHeight, paused: video.paused };
    });
    assert.deepEqual(source, { width: 320, height: 180, paused: false });
    const start = await popup.evaluate(async ({ tabId, fixtureUrl }) => {
      const settings = { enabled: true, backend: 'rocm', model: 'zenkai-1-0', preset: 'performance',
        maxFps: 24, adaptive: false, showHud: false, exclusiveViewer: false };
      const tab = await chrome.tabs.get(tabId);
      const frames = [{ frameId: 0, parentFrameId: -1, url: fixtureUrl }];
      return chrome.runtime.sendMessage({ type: 'ANIMESCALER_START_VIEWER', tab, target: { frameId: 0 }, settings, frames });
    }, { tabId, fixtureUrl });
    assert.equal(start.state, 'running', JSON.stringify(start));
    assert.equal(start.captureMode, 'viewer');
    const viewer = await eventually(() => context.pages().find(candidate => candidate.url().includes('/viewer/viewer.html?session=')),
      Boolean, 'native viewer window did not open');
    await viewer.locator('#status').filter({ hasText: 'Viewer ROCM ativo' }).waitFor({ timeout: 15000 });
    await viewer.waitForFunction(() => document.querySelector('#metrics').textContent.includes('Viewer E2E'), null, { timeout: 15000 });
    const output = await viewer.locator('#output').evaluate(canvas => {
      const context = canvas.getContext('2d');
      const pixel = [...context.getImageData(300, 200, 1, 1).data];
      return { width: canvas.width, height: canvas.height, pixel };
    });
    assert.deepEqual(output, { width: 640, height: 360, pixel: [24, 168, 74, 255] });
    const sourceAfter = await page.locator('video').evaluate(video => ({ paused: video.paused, time: video.currentTime }));
    assert.equal(sourceAfter.paused, false);
    assert.ok(sourceAfter.time > 0);
    const stop = await popup.evaluate(tabId => chrome.runtime.sendMessage({ type: 'ANIMESCALER_STOP_VIEWER', sourceTabId: tabId }), tabId);
    assert.equal(stop.state, 'stopped');
    await eventually(() => worker.evaluate(async tabId => {
      const stored = await chrome.storage.session.get(`animeScalerViewerActive:${tabId}`);
      return stored[`animeScalerViewerActive:${tabId}`] ?? null;
    }, tabId), value => value === null, 'viewer session state was not removed');
    report.checks.push({ name: 'native viewer real tab capture and output', source, output, sourceAfter, stop,
      viewerWindowId: start.viewer?.viewerWindowId });
    report.limitations.push('Temporary copy substitutes synthetic media capture and NativeEngine drawing; it proves viewer window, crop, rendering and cleanup, not user-gesture tabCapture or GPU inference.');
  });
});

test('Chromium 151 OpenCL throughput guard hides slow enhancement and keeps source playing', { timeout: 60000 }, async t => {
  await withExtension(t, 'nativeRelaySlowFrames', async ({ page, send, report }) => {
    await page.evaluate(async () => {
      const video = document.querySelector('video');
      const canvas = document.createElement('canvas');
      canvas.width = 320;
      canvas.height = 180;
      const context = canvas.getContext('2d');
      const paint = () => { context.fillStyle = '#00ff00'; context.fillRect(0, 0, 1, 1); };
      paint();
      setInterval(paint, 40);
      video.srcObject = canvas.captureStream(24);
      await video.play();
      await new Promise(resolve => video.requestVideoFrameCallback(resolve));
    });
    const settings = { backend: 'opencl', model: 'anime4kcpp-acnet-f8b4', preset: 'performance',
      enabled: true, adaptive: false, maxFps: 24, showHud: false };
    const started = await send({ type: 'ANIMESCALER_START', settings });
    assert.equal(started.state, 'running');
    const failed = await eventually(() => send({ type: 'ANIMESCALER_STATUS' }),
      value => value.state === 'error', 'slow OpenCL enhancement did not trip the sustained throughput guard', 20000);
    assert.equal(failed.error.code, 'THROUGHPUT_BELOW_TARGET');
    const playback = await page.evaluate(() => ({ paused: document.querySelector('video').paused,
      overlayDisplay: document.querySelector('canvas[data-animescaler="canvas"]')?.style.display ?? null }));
    assert.deepEqual(playback, { paused: false, overlayDisplay: 'none' });
    report.checks.push({ name: 'OpenCL sustained throughput guard', errorCode: failed.error.code, playback });
    await send({ type: 'ANIMESCALER_STOP' });
  });
});

test('Chromium 151 production NativeInline relay: content ERROR and disconnect cleanup', { timeout: 60000 }, async t => {
  await withExtension(t, 'nativeRelay', async ({ context, worker, page, fixtureUrl, extensionId, tabId, report }) => {
    const portName = 'animescaler-native-inline-v1';
    assert.equal(worker.url(), `chrome-extension://${extensionId}/background.js`);
    const initialContexts = await worker.evaluate(() => chrome.runtime.getContexts({ contextTypes: ['OFFSCREEN_DOCUMENT'] }));
    assert.deepEqual(initialContexts, [], 'probe requires a clean offscreen baseline');
    report.checks.push({ name: 'clean initial offscreen state', contexts: initialContexts });
    await worker.evaluate(() => {
      const events = globalThis.__nativeRelayE2eEvents = [];
      const senderInfo = sender => ({ id: sender?.id, url: sender?.url,
        tab: sender?.tab === undefined ? { state: 'absent' } : { state: 'observed', value: { id: sender.tab.id } },
        documentId: sender?.documentId === undefined ? { state: 'absent' } : { state: 'observed', value: sender.documentId },
        frameId: sender?.frameId === undefined ? { state: 'absent' } : { state: 'observed', value: sender.frameId } });
      chrome.runtime.onConnect.addListener(port => {
        if (port.name !== 'animescaler-native-inline-v1') return;
        events.push({ kind: 'content-connect', name: port.name, sender: senderInfo(port.sender) });
        port.onDisconnect.addListener(() => events.push({ kind: 'content-disconnect', name: port.name,
          sender: senderInfo(port.sender) }));
      });
      chrome.runtime.onMessage.addListener((message, sender, respond) => {
        if (message?.type !== '__ANIMESCALER_NATIVE_RELAY_E2E_OBSERVATION') return;
        events.push({ kind: 'offscreen-observation', observation: message.observation,
          sender: senderInfo(sender) });
        respond({ observed: true });
      });
    });

    await page.waitForFunction(() => document.documentElement.dataset.nativeRelayProbeReady === 'true',
      undefined, { timeout: 10000 });
    await page.evaluate(() => window.dispatchEvent(new Event('native-relay-e2e-start')));
    try {
      await page.waitForFunction(() => {
        const value = document.documentElement.dataset.nativeRelayE2e;
        if (!value) return false;
        const result = JSON.parse(value);
        return result.error?.type === 'ERROR' && result.disconnected;
      }, undefined, { timeout: 15000 });
    } catch (error) {
      const [pageState, relayEvents, contexts] = await Promise.all([
        page.evaluate(() => ({ ready: document.documentElement.dataset.nativeRelayProbeReady,
          state: document.documentElement.dataset.nativeRelayE2e ?? null })),
        worker.evaluate(() => globalThis.__nativeRelayE2eEvents ?? []),
        worker.evaluate(() => chrome.runtime.getContexts({ contextTypes: ['OFFSCREEN_DOCUMENT'] })),
      ]);
      report.checks.push({ name: 'failed relay trace', pageState, relayEvents, contexts,
        workerConsole: [...report.workerConsole], pageErrors: [...report.pageErrors], webErrors: [...report.webErrors] });
      throw error;
    }
    const client = await page.evaluate(() => JSON.parse(document.documentElement.dataset.nativeRelayE2e));
    assert.match(client.sessionId, /^native-relay-e2e-[0-9a-f-]{36}$/);
    assert.equal(client.requestKeys.includes('streamId'), false);
    assert.equal(client.requestKeys.includes('token'), false);
    assert.equal(client.settingsKeys.includes('token'), false);
    assert.deepEqual(client.messages.map(message => message.type), ['ERROR'], 'invalid backend must fail before engine initialization');
    assert.equal(client.error.protocol, 1);
    assert.equal(client.error.type, 'ERROR');
    assert.equal(client.error.sessionId, client.sessionId);
    assert.equal(client.error.code, 'BACKEND_UNAVAILABLE');
    assert.equal(client.disconnected, true, 'content probe must close its production Port after ERROR');

    const relayEvents = await eventually(() => worker.evaluate(() => globalThis.__nativeRelayE2eEvents),
      events => events.some(event => event.kind === 'content-connect' &&
          event.sender.url === fixtureUrl) &&
        events.some(event => event.kind === 'offscreen-observation' &&
          event.observation?.kind === 'host-received-start' && event.observation.sessionId === client.sessionId),
      'production relay sender/host trace missing');
    const contentConnect = relayEvents.find(event => event.kind === 'content-connect' && event.sender.url === fixtureUrl);
    assert.equal(contentConnect.name, portName);
    assert.equal(contentConnect.sender.id, extensionId);
    assert.equal(contentConnect.sender.url, fixtureUrl);
    assert.equal(contentConnect.sender.tab.state, 'observed');
    assert.equal(contentConnect.sender.tab.value.id, tabId);
    assert.equal(contentConnect.sender.frameId.value, 0);
    const hostObservation = relayEvents.find(event => event.kind === 'offscreen-observation' &&
      event.observation?.kind === 'host-received-start' && event.observation.sessionId === client.sessionId);
    assert.equal(hostObservation.sender.id, extensionId, 'offscreen runtime message must reach the real service worker');
    assert.equal(hostObservation.sender.url, `chrome-extension://${extensionId}/native-inline.html`);
    assert.equal(hostObservation.sender.tab.state, 'absent');
    assert.equal(hostObservation.sender.documentId.state, 'absent');
    assert.equal(hostObservation.observation.hostUrl, `chrome-extension://${extensionId}/native-inline.html`);
    assert.equal(hostObservation.observation.portName, portName);
    assert.equal(hostObservation.observation.sender.id, extensionId, 'NativeInlineHost must observe SW as Port sender');
    assert.equal(hostObservation.observation.sender.url, worker.url());
    assert.equal(hostObservation.observation.sender.tabId, undefined);
    assert.equal(hostObservation.observation.sender.documentId, undefined);
    assert.equal(Object.hasOwn(hostObservation.observation.sender, 'tabId'), false);
    assert.equal(Object.hasOwn(hostObservation.observation.sender, 'documentId'), false);
    assert.deepEqual(hostObservation.observation.message, {
      protocol: 1, type: 'START', sessionId: client.sessionId,
      backend: 'e2e-invalid-backend', model: 'e2e-invalid-model', captureMode: 'direct-png',
      sourceWidth: 320, sourceHeight: 180,
      messageKeys: ['backend', 'captureMode', 'model', 'protocol', 'sessionId', 'settings', 'sourceHeight', 'sourceWidth', 'type'],
      settingsKeys: ['backend', 'model', 'preset'],
    });

    assert.equal(client.disconnected, true, 'service worker must close the content Port after terminal ERROR');
    const offscreenClose = await eventually(() => worker.evaluate(() => chrome.runtime.getContexts({ contextTypes: ['OFFSCREEN_DOCUMENT'] })),
      contexts => contexts.length === 0, 'production offscreen document survived content disconnect');
    report.checks.push({ name: 'production content to SW sender', sender: contentConnect.sender },
      { name: 'production SW to NativeInlineHost sender and START', observation: hostObservation.observation,
        offscreenSender: hostObservation.sender },
      { name: 'production relay ERROR returned to content', error: client.error },
      { name: 'content Port closed after terminal ERROR', disconnected: client.disconnected },
      { name: 'offscreen cleanup', contexts: offscreenClose });
    report.limitations.push('Temporary content probe sends an invalid backend START, so NativeEngine.initialize and native service are not invoked.',
      'Passive host observer sends a separate runtime message to expose real offscreen MessageSender metadata; production ERROR itself is checked on the content Port.',
      'No production manifest, background worker, or NativeInlineHost source is changed; only the disposable copy receives probe scripts.');
  });
});
