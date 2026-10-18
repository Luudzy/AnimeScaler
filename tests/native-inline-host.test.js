import test from 'node:test';
import assert from 'node:assert/strict';
import { NativeInlineHost, MAX_NATIVE_FRAME_BYTES } from '../src/host/native-inline-host.js';
import { NativeEngine } from '../src/engine/native-engine.js';
import { normalizeSettings } from '../src/core/settings.js';

function pngDataUrl(width = 2, height = 2, byteLength = 33) {
  const bytes = new Uint8Array(byteLength);
  bytes.set([137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82]);
  new DataView(bytes.buffer).setUint32(16, width);
  new DataView(bytes.buffer).setUint32(20, height);
  return `data:image/png;base64,${Buffer.from(bytes).toString('base64')}`;
}

function setup({ maxFrameBytes = MAX_NATIVE_FRAME_BYTES, processImage } = {}) {
  const sent = [];
  const calls = [];
  const engine = {
    async initialize(options) { calls.push(['initialize', options]); return { backend: options.settings.backend }; },
    async processImage(bitmap, options) {
      calls.push(['processImage', bitmap, options]);
      return processImage ? processImage(bitmap, options) : {
        sequence: options.sequence, mediaTime: options.mediaTime, width: 4, height: 4,
        mime: 'image/png', dataUrl: pngDataUrl(4, 4),
      };
    },
    async updateSettings(settings) { calls.push(['updateSettings', settings]); },
    async destroy() { calls.push(['destroy']); },
  };
  const host = new NativeInlineHost({
    send: message => sent.push(message),
    createEngine: () => engine,
    createCanvas: () => ({ width: 1, height: 1 }),
    decodeInput: async dataUrl => ({ bitmap: { close() {} }, ...inspectPng(dataUrl) }),
    maxFrameBytes,
    extensionId: 'test',
  });
  return { host, sent, calls, engine };
}

function mockPort(sender, name = 'animescaler-native-inline-v1') {
  const messages = new Set();
  const disconnects = new Set();
  return {
    name,
    sender,
    onMessage: {
      addListener(listener) { messages.add(listener); },
      removeListener(listener) { messages.delete(listener); },
    },
    onDisconnect: {
      addListener(listener) { disconnects.add(listener); },
      removeListener(listener) { disconnects.delete(listener); },
    },
    messages,
    disconnects,
    disconnect() { for (const listener of [...disconnects]) listener(); },
  };
}

function frameMessage(sequence, overrides = {}) {
  return {
    protocol: 1, type: 'FRAME_IN', sessionId: 's-1', sequence, mediaTime: sequence / 24,
    width: 2, height: 2, mime: 'image/png', dataUrl: pngDataUrl(), ...overrides,
  };
}

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

function inspectPng(dataUrl) {
  const bytes = Buffer.from(dataUrl.slice('data:image/png;base64,'.length), 'base64');
  return {
    width: new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(16),
    height: new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(20),
    byteLength: bytes.byteLength,
  };
}

const startMessage = overrides => ({
  protocol: 1, type: 'START', sessionId: 's-1', backend: 'rocm', model: 'zenkai-1-0',
  settings: { backend: 'rocm', model: 'zenkai-1-0', preset: 'performance' },
  captureMode: 'direct-png', sourceWidth: 1280, sourceHeight: 720, ...overrides,
});

test('host forwards only finite allowlisted native stats', async () => {
  const { host, sent } = setup({ processImage: async (_, options) => ({
    ...options, width: 4, height: 4, mime: 'image/png', dataUrl: pngDataUrl(4, 4),
    stats: { inferenceMs: 8, warmup: false, reused: true, token: 'secret', url: 'http://private',
      captureMs: -1, workerMs: NaN, temporal: { acceptedResults: 2, url: 'private' } },
  }) });
  await host.handleMessage(startMessage());
  await host.handleMessage(frameMessage(1));
  assert.deepEqual(sent.find(message => message.type === 'FRAME_OUT').stats,
    { inferenceMs: 8, warmup: false, reused: true, temporal: { acceptedResults: 2 } });
  await host.disconnect();
});

for (const [width, height] of [[8193, 1], [1, 8193], [8192, 8192], [7680, 4321]]) {
  test(`host rejects decoded allocation ${width}x${height} before decoder`, async () => {
    const { host, sent } = setup();
    let decoded = 0;
    host.decodeInput = async () => { decoded++; throw new Error('must not decode'); };
    await host.handleMessage(startMessage());
    await host.handleMessage(frameMessage(1, { width, height, dataUrl: pngDataUrl(width, height) }));
    assert.equal(decoded, 0);
    assert.equal(sent.at(-1).code, 'FRAME_TOO_LARGE');
    await host.disconnect();
  });
}

for (const [width, height] of [[8192, 1], [7680, 4320]]) {
  test(`host accepts bounded PNG allocation ${width}x${height}`, async () => {
    const { host, sent } = setup();
    await host.handleMessage(startMessage());
    await host.handleMessage(frameMessage(1, { width, height, dataUrl: pngDataUrl(width, height) }));
    assert.equal(sent.at(-1).accepted, true);
    await host.disconnect();
  });
}

test('host rejects oversized output before publishing and closes input', async () => {
  const { host, sent } = setup({ processImage: async (_, options) => ({
    sequence: options.sequence, mediaTime: options.mediaTime, width: 8193, height: 1,
    mime: 'image/png', dataUrl: pngDataUrl(8193, 1),
  }) });
  let closed = 0;
  host.decodeInput = async dataUrl => ({ bitmap: { close() { closed++; } }, ...inspectPng(dataUrl) });
  await host.handleMessage(startMessage());
  await host.handleMessage(frameMessage(1));
  assert.equal(sent.some(message => message.type === 'FRAME_OUT'), false);
  assert.equal(sent.at(-1).code, 'FRAME_TOO_LARGE');
  assert.equal(closed, 1);
  assert.equal(host.session, null);
});

test('native inline host initializes extension-owned engine and acknowledges PNG after output', async () => {
  const { host, sent, calls } = setup();
  await host.handleMessage(startMessage());
  assert.equal(calls[0][1].sessionId, 's-1');
  await host.handleMessage({ protocol: 1, type: 'FRAME_IN', sessionId: 's-1', sequence: 1, mediaTime: 1.25,
    width: 2, height: 2, mime: 'image/png', dataUrl: pngDataUrl() });

  assert.equal(calls[0][1].video.videoWidth, 1280);
  assert.equal(calls[0][1].video.videoHeight, 720);
  assert.deepEqual(sent.map(message => message.type), ['STARTED', 'FRAME_OUT', 'FRAME_ACK']);
  assert.equal(sent[1].sequence, 1);
  assert.equal(sent[1].mediaTime, 1.25);
  assert.equal(sent[1].mime, 'image/png');
  assert.equal(sent[2].accepted, true);
});

test('native host rejects stream IDs and credentials before engine initialization', async () => {
  const { host, sent, calls } = setup();
  await host.handleMessage(startMessage({ streamId: 'must-not-cross-context' }));
  assert.equal(sent.at(-1).code, 'SOURCE_NOT_ISOLATED');
  assert.equal(calls.length, 0);

  await host.handleMessage(startMessage({ sessionId: 's-2', settings: { model: 'zenkai-1-0', token: 'secret' } }));
  assert.equal(sent.at(-1).code, 'SOURCE_NOT_ISOLATED');
  assert.equal(calls.length, 0);
});

test('native host rejects protocol mismatch, invalid backend, and concurrent sessions', async () => {
  const { host, sent } = setup();
  await host.handleMessage(startMessage({ protocol: 2 }));
  assert.equal(sent.at(-1).code, 'BACKEND_UNAVAILABLE');
  await host.handleMessage(startMessage({ backend: 'd3d12', model: 'adore' }));
  assert.equal(sent.at(-1).code, 'BACKEND_UNAVAILABLE');
  await host.handleMessage(startMessage());
  await host.handleMessage(startMessage({ sessionId: 's-2' }));
  assert.equal(sent.at(-1).code, 'NATIVE_SESSION_BUSY');
});

test('native host validates frame session, sequence, dimensions, and PNG size', async () => {
  const { host, sent, calls } = setup({ maxFrameBytes: 40 });
  await host.handleMessage(startMessage());
  await host.handleMessage({ protocol: 1, type: 'FRAME_IN', sessionId: 'stale', sequence: 1, mediaTime: 0,
    width: 2, height: 2, mime: 'image/png', dataUrl: pngDataUrl() });
  assert.equal(sent.at(-1).code, 'STALE_SESSION');
  assert.equal(sent.at(-1).staleOrigin, 'native_host');
  assert.equal(sent.at(-1).messageType, 'FRAME_IN');
  assert.equal(sent.at(-1).sequence, 1);

  await host.handleMessage({ protocol: 1, type: 'FRAME_IN', sessionId: 's-1', sequence: 1, mediaTime: 0,
    width: 3, height: 2, mime: 'image/png', dataUrl: pngDataUrl() });
  assert.equal(sent.at(-1).code, 'SOURCE_NOT_ISOLATED');
  assert.equal(calls.filter(([name]) => name === 'processImage').length, 0);

  await host.handleMessage({ protocol: 1, type: 'FRAME_IN', sessionId: 's-1', sequence: 1, mediaTime: 0,
    width: 2, height: 2, mime: 'image/png', dataUrl: pngDataUrl(2, 2, 64) });
  assert.equal(sent.at(-1).code, 'FRAME_TOO_LARGE');
});

test('native host enforces the specified 16 MiB frame ceiling', () => {
  assert.equal(MAX_NATIVE_FRAME_BYTES, 16 * 1024 * 1024);
});

test('stop and port disconnect destroy the engine exactly once', async () => {
  const { host, sent, calls } = setup();
  await host.handleMessage(startMessage());
  await host.handleMessage({ protocol: 1, type: 'STOP', sessionId: 's-1', reason: 'user' });
  assert.equal(sent.at(-1).type, 'STOPPED');
  await host.handleMessage({ protocol: 1, type: 'STOP', sessionId: 's-1', reason: 'duplicate' });
  assert.equal(calls.filter(([name]) => name === 'destroy').length, 1);
});

const invalidAttachSenders = [
  ['wrong ID', { id: 'other', url: 'chrome-extension://test/background.js' }],
  ['missing ID', { url: 'chrome-extension://test/background.js' }],
  ['missing URL', { id: 'test' }],
  ['other extension ID in URL', { id: 'test', url: 'chrome-extension://other/background.js' }],
  ['non-extension URL', { id: 'test', url: 'https://example.test/background.js' }],
  ['popup page', { id: 'test', url: 'chrome-extension://test/popup/popup.html' }],
  ['old host page', { id: 'test', url: 'chrome-extension://test/native-inline.html' }],
  ['background URL with query', { id: 'test', url: 'chrome-extension://test/background.js?session=1' }],
  ['background URL with path suffix', { id: 'test', url: 'chrome-extension://test/background.js/extra' }],
  ['background URL with filename suffix', { id: 'test', url: 'chrome-extension://test/background.js-extra' }],
  ['tab sender present', { id: 'test', url: 'chrome-extension://test/background.js', tab: { id: 1 } }],
];
for (const [label, sender] of invalidAttachSenders) {
  test(`attachPort rejects ${label}`, () => {
    const { host } = setup();
    assert.throws(() => host.attachPort(mockPort(sender)), { code: 'BACKEND_UNAVAILABLE' });
  });
}

test('attachPort accepts only the exact SW sender URL without a tab sender', () => {
  const { host } = setup();
  const port = mockPort({ id: 'test', url: 'chrome-extension://test/background.js' });
  host.attachPort(port);
  assert.equal(port.messages.size, 1);
  assert.equal(port.disconnects.size, 1);
});

test('port disconnect clears session and destroys engine exactly once', async () => {
  const { host, calls } = setup();
  const port = mockPort({ id: 'test', url: 'chrome-extension://test/background.js' });
  host.attachPort(port);
  await host.handleMessage(startMessage(), port.sender);

  port.disconnect();
  await new Promise(resolve => setImmediate(resolve));
  await host.disconnect();

  assert.equal(host.session, null);
  assert.equal(calls.filter(([name]) => name === 'destroy').length, 1);
  assert.equal(port.messages.size, 0);
  assert.equal(port.disconnects.size, 0);
});

test('one frame in flight coalesces two newer frames to the latest with matching ACKs', async () => {
  const first = deferred();
  const { host, sent, calls } = setup({ processImage: (_bitmap, options) =>
    options.sequence === 1 ? first.promise : Promise.resolve({
      sequence: options.sequence, mediaTime: options.mediaTime, width: 4, height: 4,
      mime: 'image/png', dataUrl: pngDataUrl(4, 4),
    }) });
  await host.handleMessage(startMessage());

  const inFlight = host.handleMessage(frameMessage(1));
  await new Promise(resolve => setImmediate(resolve));
  const second = host.handleMessage(frameMessage(2));
  const latest = host.handleMessage(frameMessage(3));
  await Promise.all([second, latest]);
  first.resolve({ sequence: 1, mediaTime: 1 / 24, width: 4, height: 4,
    mime: 'image/png', dataUrl: pngDataUrl(4, 4) });
  await inFlight;

  assert.deepEqual(calls.filter(([name]) => name === 'processImage').map(([, , options]) => options.sequence), [1, 3]);
  assert.deepEqual(sent.filter(message => message.type === 'FRAME_OUT').map(message => message.sequence), [1, 3]);
  assert.deepEqual(sent.filter(message => message.type === 'FRAME_ACK').map(message => message.sequence), [2, 1, 3]);
  assert.deepEqual(sent.filter(message => message.type === 'FRAME_ACK').map(message => message.accepted), [false, true, true]);
});

test('pending replacement immediately rejects each displaced frame without retaining rejection history', async () => {
  const processing = deferred();
  const { host, sent, calls } = setup({ processImage: (_bitmap, options) => options.sequence === 1
    ? processing.promise : Promise.resolve({ sequence: options.sequence, mediaTime: options.mediaTime,
      width: 4, height: 4, mime: 'image/png', dataUrl: pngDataUrl(4, 4) }) });
  await host.handleMessage(startMessage());
  const inFlight = host.handleMessage(frameMessage(1));
  await new Promise(resolve => setImmediate(resolve));
  try {
    await host.handleMessage(frameMessage(2));
    for (let sequence = 3; sequence <= 100; sequence++) {
      await host.handleMessage(frameMessage(sequence));
      const ack = sent.at(-1);
      assert.equal(ack.type, 'FRAME_ACK');
      assert.equal(ack.sequence, sequence - 1);
      assert.equal(ack.accepted, false);
      assert.equal(ack.code, 'STALE_FRAME');
      assert.equal(host.session.pending.sequence, sequence);
      assert.equal(host.session.rejectedSequences, undefined);
      assert.equal(calls.filter(([name]) => name === 'processImage').length, 1);
    }
  } finally {
    processing.resolve({ sequence: 1, mediaTime: 1 / 24, width: 4, height: 4,
      mime: 'image/png', dataUrl: pngDataUrl(4, 4) });
    await inFlight;
  }
  assert.deepEqual(calls.filter(([name]) => name === 'processImage').map(([, , options]) => options.sequence), [1, 100]);
});

for (const code of ['BACKEND_UNAVAILABLE', 'FRAME_TIMEOUT']) {
  test(`active ${code} also rejects pending with a terminal ACK`, async () => {
    const processing = deferred();
    const { host, sent, calls } = setup({ processImage: () => processing.promise });
    await host.handleMessage(startMessage());
    const session = host.session;
    const inFlight = host.handleMessage(frameMessage(1));
    await new Promise(resolve => setImmediate(resolve));
    await host.handleMessage(frameMessage(2));
    processing.reject(Object.assign(new Error('private detail must not escape'), { code }));
    await inFlight;
    assert.deepEqual(sent.filter(message => message.type === 'FRAME_ACK').map(message =>
      [message.sequence, message.accepted, message.code]), [[1, false, code], [2, false, code]]);
    assert.equal(sent.some(message => message.type === 'FRAME_OUT'), false);
    assert.equal(JSON.stringify(sent).includes('private detail'), false);
    assert.equal(host.session, null);
    assert.equal(session.pending, null);
    assert.equal(calls.filter(([name]) => name === 'destroy').length, 1);
  });
}

function controlledTimers(t) {
  let now = 0;
  let nextId = 0;
  const timers = new Map();
  t.mock.method(globalThis, 'setTimeout', (callback, delay) => {
    const id = ++nextId;
    timers.set(id, { callback, due: now + delay });
    return id;
  });
  t.mock.method(globalThis, 'clearTimeout', id => timers.delete(id));
  return { timers, tick(milliseconds) {
    now += milliseconds;
    for (const [id, timer] of [...timers]) if (timer.due <= now) {
      timers.delete(id);
      timer.callback();
    }
  } };
}

// Decision ID 48: inline had only a 40 s failure deadline, no freshness contract.
// Use the existing temporal history budget of 300 ms for admission, not inference.
// Admission at age >= 300 ms expires; timestamps stay local to each queue.
for (const age of [299, 300]) test(`pending frame admission at ${age} ms preserves ACKs and skips expired decode`, async t => {
  let now = 0;
  t.mock.method(performance, 'now', () => now);
  const first = deferred();
  const { host, sent, calls } = setup({ processImage: async (_, options) => options.sequence === 1 ? first.promise : {
    sequence: options.sequence, mediaTime: options.mediaTime, width: 4, height: 4,
    mime: 'image/png', dataUrl: pngDataUrl(4, 4),
  } });
  const decoded = [], closed = [];
  host.decodeInput = async dataUrl => {
    decoded.push(now);
    return { ...inspectPng(dataUrl), bitmap: { close() { closed.push(now); } } };
  };
  await host.handleMessage(startMessage());
  const processing = host.handleMessage(frameMessage(1));
  await new Promise(resolve => setImmediate(resolve));
  await host.handleMessage(frameMessage(2));
  now = age;
  first.resolve({ sequence: 1, mediaTime: 1 / 24, width: 4, height: 4, mime: 'image/png', dataUrl: pngDataUrl(4, 4) });
  await processing;
  assert.equal(decoded.length, age === 299 ? 2 : 1);
  assert.equal(closed.length, decoded.length);
  assert.deepEqual(calls.filter(([name]) => name === 'processImage').map(([, , options]) => options.sequence), age === 299 ? [1, 2] : [1]);
  assert.deepEqual(sent.filter(message => message.type === 'FRAME_ACK').map(message => [message.sequence, message.accepted, message.code]),
    age === 299 ? [[1, true, undefined], [2, true, undefined]] : [[1, true, undefined], [2, false, 'STALE_FRAME']]);
  assert.equal(host.session.pending, null);
  assert.equal(sent.some(message => message.type === 'ERROR'), false);
  assert.equal(host.session.closed, false);
  await host.handleMessage(frameMessage(3));
  assert.equal(sent.at(-1).accepted, true);
  await host.disconnect();
});

test('newest pending frame keeps its own arrival age without an extra queue or timer', async t => {
  let now = 0;
  t.mock.method(performance, 'now', () => now);
  const clock = controlledTimers(t), first = deferred();
  const { host, sent, calls } = setup({ processImage: async (_, options) => options.sequence === 1 ? first.promise : {
    sequence: options.sequence, mediaTime: options.mediaTime, width: 4, height: 4,
    mime: 'image/png', dataUrl: pngDataUrl(4, 4),
  } });
  await host.handleMessage(startMessage());
  const processing = host.handleMessage(frameMessage(1));
  await new Promise(resolve => setImmediate(resolve));
  await host.handleMessage(frameMessage(2));
  now = 250;
  await host.handleMessage(frameMessage(3));
  now = 500;
  first.resolve({ sequence: 1, mediaTime: 1 / 24, width: 4, height: 4, mime: 'image/png', dataUrl: pngDataUrl(4, 4) });
  await processing;
  assert.deepEqual(calls.filter(([name]) => name === 'processImage').map(([, , options]) => options.sequence), [1, 3]);
  assert.deepEqual(sent.filter(message => message.type === 'FRAME_ACK').map(message => [message.sequence, message.accepted]),
    [[2, false], [1, true], [3, true]]);
  assert.equal(clock.timers.size, 0);
  assert.equal(host.session.pending, null);
  await host.disconnect();
});

test('60 Hz arrivals with slow inference keep one pending PNG and ACK each frame once', async t => {
  let now = 0;
  t.mock.method(performance, 'now', () => now);
  const clock = controlledTimers(t), first = deferred();
  const { host, sent, calls } = setup({ processImage: async (_, options) => options.sequence === 1 ? first.promise : {
    sequence: options.sequence, mediaTime: options.mediaTime, width: 4, height: 4,
    mime: 'image/png', dataUrl: pngDataUrl(4, 4),
  } });
  let decoded = 0, closed = 0;
  host.decodeInput = async dataUrl => {
    decoded++;
    return { ...inspectPng(dataUrl), bitmap: { close() { closed++; } } };
  };
  await host.handleMessage(startMessage());
  const processing = host.handleMessage(frameMessage(1));
  await new Promise(resolve => setImmediate(resolve));
  for (let sequence = 2; sequence <= 61; sequence++) {
    now = (sequence - 2) * 1000 / 60;
    await host.handleMessage(frameMessage(sequence));
    assert.equal(host.session.pending.sequence, sequence);
    assert.equal(decoded, 1);
    assert.equal(clock.timers.size, 1);
  }
  now += 301;
  first.resolve({ sequence: 1, mediaTime: 1 / 24, width: 4, height: 4, mime: 'image/png', dataUrl: pngDataUrl(4, 4) });
  await processing;
  const acknowledgements = sent.filter(message => message.type === 'FRAME_ACK');
  assert.equal(acknowledgements.length, 61);
  assert.equal(new Set(acknowledgements.map(message => message.sequence)).size, 61);
  assert.equal(acknowledgements.find(message => message.sequence === 61).code, 'STALE_FRAME');
  assert.deepEqual(calls.filter(([name]) => name === 'processImage').map(([, , options]) => options.sequence), [1]);
  assert.equal(decoded, 1);
  assert.equal(closed, 1);
  assert.equal(host.session.pending, null);
  assert.equal(host.session.pendingReceivedAt, undefined);
  assert.equal(clock.timers.size, 0);
  await host.disconnect();
});

test('frame expiring while PNG decoder awaits skips inference and closes bitmap once', async t => {
  let now = 0;
  t.mock.method(performance, 'now', () => now);
  const decoding = deferred();
  const { host, sent, calls } = setup();
  let closed = 0;
  host.decodeInput = () => decoding.promise;
  await host.handleMessage(startMessage());
  const processing = host.handleMessage(frameMessage(1));
  now = 300;
  decoding.resolve({ ...inspectPng(pngDataUrl()), bitmap: { close() { closed++; } } });
  await processing;
  assert.equal(calls.some(([name]) => name === 'processImage'), false);
  assert.equal(closed, 1);
  assert.deepEqual(sent.filter(message => message.type === 'FRAME_ACK').map(message => [message.sequence, message.code]), [[1, 'STALE_FRAME']]);
  assert.equal(host.session.closed, false);
  await host.disconnect();
});

test('internal 40-second deadline rejects active and pending and suppresses late completion', async t => {
  const clock = controlledTimers(t);
  const processing = deferred();
  const { host, sent, calls } = setup({ processImage: () => processing.promise });
  await host.handleMessage(startMessage());
  const inFlight = host.handleMessage(frameMessage(1));
  await new Promise(resolve => setImmediate(resolve));
  await host.handleMessage(frameMessage(2));
  try {
    assert.equal(clock.timers.size, 1, 'host must arm its own deadline');
    clock.tick(39999);
    assert.equal(host.session.closed, false);
    assert.equal(sent.some(message => message.type === 'FRAME_ACK'), false);
    clock.tick(1);
    await inFlight;
    assert.equal(host.session, null);
    assert.equal(clock.timers.size, 0);
    assert.deepEqual(sent.filter(message => message.type === 'FRAME_ACK').map(message =>
      [message.sequence, message.accepted, message.code]),
    [[1, false, 'FRAME_TIMEOUT'], [2, false, 'FRAME_TIMEOUT']]);
    assert.equal(calls.filter(([name]) => name === 'destroy').length, 1);
    const finished = sent.length;
    processing.resolve({ sequence: 1, mediaTime: 1 / 24, width: 4, height: 4,
      mime: 'image/png', dataUrl: pngDataUrl(4, 4) });
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(sent.length, finished);
    assert.equal(sent.some(message => message.type === 'FRAME_OUT'), false);
    await host.disconnect();
    assert.equal(calls.filter(([name]) => name === 'destroy').length, 1);
  } finally {
    processing.resolve({ sequence: 1, mediaTime: 1 / 24, width: 4, height: 4,
      mime: 'image/png', dataUrl: pngDataUrl(4, 4) });
    await inFlight;
  }
});

test('normal completion clears the internal deadline', async t => {
  const clock = controlledTimers(t);
  const { host, sent } = setup();
  await host.handleMessage(startMessage());
  await host.handleMessage(frameMessage(1));
  const finished = sent.length;
  assert.equal(clock.timers.size, 0);
  clock.tick(40000);
  assert.equal(sent.length, finished);
  assert.equal(host.session.closed, false);
});

for (const reason of ['STOP', 'disconnect']) {
  test(`${reason} clears pending and deadline without emitting later ACKs`, async t => {
    const clock = controlledTimers(t);
    const processing = deferred();
    const { host, sent, calls } = setup({ processImage: () => processing.promise });
    await host.handleMessage(startMessage());
    const session = host.session;
    const inFlight = host.handleMessage(frameMessage(1));
    await new Promise(resolve => setImmediate(resolve));
    await host.handleMessage(frameMessage(2));
    if (reason === 'STOP') await host.handleMessage({ protocol: 1, type: 'STOP', sessionId: 's-1' });
    else await host.disconnect();
    const finished = sent.length;
    clock.tick(40000);
    processing.reject(Object.assign(new Error('late failure'), { code: 'FRAME_TIMEOUT' }));
    await inFlight;
    assert.equal(clock.timers.size, 0);
    assert.equal(session.pending, null);
    assert.equal(sent.length, finished);
    assert.equal(sent.some(message => message.type === 'FRAME_ACK'), false);
    assert.equal(calls.filter(([name]) => name === 'destroy').length, 1);
  });
}

test('stop discards output from an older in-flight frame', async () => {
  const processing = deferred();
  const { host, sent } = setup({ processImage: () => processing.promise });
  await host.handleMessage(startMessage());

  const inFlight = host.handleMessage(frameMessage(1));
  await new Promise(resolve => setImmediate(resolve));
  await host.handleMessage({ protocol: 1, type: 'STOP', sessionId: 's-1', reason: 'user' });
  processing.resolve({ sequence: 1, mediaTime: 1 / 24, width: 4, height: 4,
    mime: 'image/png', dataUrl: pngDataUrl(4, 4) });
  await inFlight;

  assert.equal(sent.some(message => message.type === 'FRAME_OUT' && message.sequence === 1), false);
  assert.equal(sent.some(message => message.type === 'FRAME_ACK' && message.sequence === 1), false);
  assert.equal(sent.at(-1).type, 'STOPPED');
});

test('timed out frame emits rejection ACK and never publishes late output', async () => {
  const processing = deferred();
  const { host, sent, calls } = setup({ processImage: () => processing.promise });
  await host.handleMessage(startMessage());

  const inFlight = host.handleMessage(frameMessage(1));
  await new Promise(resolve => setImmediate(resolve));
  const timeout = Object.assign(new Error('frame timeout'), { code: 'FRAME_TIMEOUT' });
  processing.reject(timeout);
  await inFlight;

  assert.equal(sent.some(message => message.type === 'FRAME_OUT' && message.sequence === 1), false);
  assert.ok(sent.some(message => message.type === 'ERROR' && message.code === 'FRAME_TIMEOUT'));
  assert.ok(sent.some(message => message.type === 'FRAME_ACK' && message.sequence === 1 &&
    message.accepted === false && message.code === 'FRAME_TIMEOUT'));
  assert.equal(calls.filter(([name]) => name === 'destroy').length, 1);
});

for (const outcome of ['resolve', 'reject']) {
  test(`STOP cancels first then waits for CONFIGURE to ${outcome} and remains terminal`, async () => {
    const updating = deferred();
    const { host, sent, engine } = setup();
    const events = [];
    engine.updateSettings = async () => {
      events.push('update-start');
      try { await updating.promise; }
      finally { events.push('update-finished'); }
    };
    engine.destroy = async () => { events.push('destroy'); };
    await host.handleMessage(startMessage());
    const session = host.session;
    const previousSettings = session.settings;
    const configuring = host.handleMessage({ protocol: 1, type: 'CONFIGURE', sessionId: 's-1',
      revision: 1, adaptiveLevel: 0, sourceWidth: 320, sourceHeight: 180, settings: { preset: 'quality' } });
    await new Promise(resolve => setImmediate(resolve));
    let stopped = false;
    const stopping = host.handleMessage({ protocol: 1, type: 'STOP', sessionId: 's-1' })
      .then(() => { stopped = true; });
    try {
      await new Promise(resolve => setImmediate(resolve));
      assert.equal(stopped, false, 'STOP must wait for updateSettings');
      assert.deepEqual(events, ['update-start', 'destroy']);
      assert.equal(sent.some(message => message.type === 'STOPPED'), false);
    } finally {
      if (outcome === 'resolve') updating.resolve();
      else updating.reject(Object.assign(new Error('private configuration error'), { code: 'BACKEND_UNAVAILABLE' }));
      await Promise.all([configuring, stopping]);
    }
    assert.deepEqual(events, ['update-start', 'destroy', 'update-finished']);
    assert.equal(host.session, null);
    assert.equal(session.settings, previousSettings, 'closed session must not publish configured settings');
    assert.equal(sent.at(-1).type, 'STOPPED');
    assert.equal(sent.filter(message => message.type === 'STOPPED').length, 1);
    const terminalIndex = sent.findIndex(message => message.type === 'STOPPED');
    await new Promise(resolve => setImmediate(resolve));
    assert.deepEqual(sent.slice(terminalIndex + 1), []);
    assert.equal(sent.some(message => message.type === 'ERROR' || message.type === 'FRAME_OUT'), false);
    await host.disconnect();
    assert.equal(events.filter(event => event === 'destroy').length, 1);
  });
}

test('STOP aborts NativeEngine initialization fetch before waiting and never publishes STARTED', async t => {
  const originalChrome = globalThis.chrome;
  globalThis.chrome = { runtime: { getURL: path => `chrome-extension://test/${path}` } };
  t.after(() => { globalThis.chrome = originalChrome; });
  const fetching = deferred();
  let signal;
  t.mock.method(globalThis, 'fetch', (_url, options) => {
    signal = options.signal;
    signal?.addEventListener('abort', () => fetching.reject(new DOMException('Aborted', 'AbortError')), { once: true });
    return fetching.promise;
  });
  const native = new NativeEngine();
  native.diagnostics = { event() {}, flush() {} };
  const { host, sent } = setup();
  host.createEngine = () => native;
  const starting = host.handleMessage(startMessage());
  await new Promise(resolve => setImmediate(resolve));
  const stopping = host.handleMessage({ protocol: 1, type: 'STOP', sessionId: 's-1' });
  try {
    assert.ok(signal, 'private connection fetch must receive AbortSignal');
    assert.equal(signal.aborted, true, 'STOP must abort fetch synchronously');
  } finally {
    fetching.reject(new DOMException('Aborted', 'AbortError'));
    await Promise.all([starting, stopping]);
  }
  assert.equal(host.session, null);
  assert.equal(native.closed, true);
  assert.deepEqual(sent.map(message => message.type), ['STOPPED']);
});

for (const reason of ['STOP', 'disconnect']) {
  test(`${reason} cancels real CONFIGURE Worker request before waiting without late settings`, async () => {
    const native = new NativeEngine();
    native.settings = normalizeSettings({ backend: 'rocm', model: 'adore', adaptive: false });
    native.currentSize = { outputWidth: 320, outputHeight: 180 };
    native.diagnostics = { event() {}, flush() {} };
    const previousSettings = native.settings;
    const previousSize = native.currentSize;
    let terminated = 0;
    let requestId;
    native.worker = { postMessage(message) { requestId = message.id; }, terminate() { terminated++; } };
    const { host, sent, engine } = setup();
    engine.updateSettings = native.updateSettings.bind(native);
    engine.destroy = native.destroy.bind(native);
    await host.handleMessage(startMessage());
    native.presentationFlight = native.call('present', {});
    const request = native.requests.get(requestId);
    const flightRejected = assert.rejects(native.presentationFlight, /ROCm encerrado/);
    const configuring = host.handleMessage({ protocol: 1, type: 'CONFIGURE', sessionId: 's-1',
      revision: 1, adaptiveLevel: 0, sourceWidth: 320, sourceHeight: 180, settings: { preset: 'quality' } });
    await new Promise(resolve => setImmediate(resolve));
    const closing = reason === 'STOP'
      ? host.handleMessage({ protocol: 1, type: 'STOP', sessionId: 's-1' }) : host.disconnect();
    try {
      assert.equal(terminated, 1, 'cleanup must terminate Worker before awaiting CONFIGURE');
      assert.equal(native.requests.size, 0);
    } finally {
      clearTimeout(request.timer);
      request.resolve({ presented: true });
      await Promise.all([configuring, closing, flightRejected]);
    }
    assert.equal(native.settings, previousSettings);
    assert.equal(native.currentSize, previousSize);
    assert.equal(host.session, null);
    assert.deepEqual(sent.map(message => message.type), reason === 'STOP' ? ['STARTED', 'STOPPED'] : ['STARTED']);
    await host.disconnect();
    assert.equal(terminated, 1);
  });
}

test('CONFIGURE failure before STOP still reports its error and destroys once', async () => {
  const { host, sent, calls, engine } = setup();
  engine.updateSettings = async () => { throw Object.assign(new Error('private settings failure'), { code: 'BACKEND_UNAVAILABLE' }); };
  await host.handleMessage(startMessage());
  await host.handleMessage({ protocol: 1, type: 'CONFIGURE', sessionId: 's-1', revision: 1,
    adaptiveLevel: 0, sourceWidth: 320, sourceHeight: 180, settings: { preset: 'quality' } });
  assert.equal(sent.at(-1).type, 'ERROR');
  assert.equal(sent.at(-1).code, 'BACKEND_UNAVAILABLE');
  assert.equal(JSON.stringify(sent).includes('private settings failure'), false);
  assert.equal(host.session, null);
  await host.disconnect();
  assert.equal(calls.filter(([name]) => name === 'destroy').length, 1);
});

test('CONFIGURE applies revision, adaptive level and source dimensions before acknowledging', async () => {
  const messages = [];
  const calls = [];
  const host = new NativeInlineHost({ send: message => messages.push(message),
    createEngine: () => ({
      async initialize() {},
      async updateSettings(settings, video, level) { calls.push({ settings, video: { ...video }, level }); },
      async destroy() {},
    }), createCanvas: () => ({ width: 1, height: 1 }) });
  await host.handleMessage(startMessage());
  await host.handleMessage({ protocol: 1, type: 'CONFIGURE', sessionId: 's-1', revision: 1,
    adaptiveLevel: 3, sourceWidth: 640, sourceHeight: 360,
    settings: { backend: 'rocm', model: 'zenkai-1-0', preset: 'quality' } });
  assert.deepEqual(calls.at(-1).video, { videoWidth: 640, videoHeight: 360 });
  assert.equal(calls.at(-1).level, 3);
  assert.deepEqual(messages.at(-1), { protocol: 1, type: 'CONFIGURED', sessionId: 's-1', revision: 1,
    backend: 'rocm', model: 'zenkai-1-0' });
});

test('CONFIGURE replaces native executor when backend changes', async () => {
  const sent = [];
  const engines = [];
  const canvases = [];
  const host = new NativeInlineHost({ send: message => sent.push(message),
    createCanvas: () => { const canvas = { width: 1, height: 1 }; canvases.push(canvas); return canvas; },
    createEngine: () => {
      const engine = { initialized: null, destroyed: 0,
        async initialize(options) { this.initialized = options; },
        async updateSettings() {}, async processImage() {},
        async destroy() { this.destroyed++; } };
      engines.push(engine);
      return engine;
    } });
  await host.handleMessage(startMessage());
  await host.handleMessage({ protocol: 1, type: 'CONFIGURE', sessionId: 's-1', revision: 1,
    adaptiveLevel: 1, sourceWidth: 640, sourceHeight: 360,
    settings: { backend: 'tensorrt', model: 'zenkai-1-0', preset: 'performance' } });
  assert.equal(engines.length, 2);
  assert.equal(engines[0].destroyed, 1);
  assert.equal(engines[1].initialized.settings.backend, 'tensorrt');
  assert.equal(engines[1].initialized.adaptiveLevel, 1);
  assert.deepEqual(engines[1].initialized.video, { videoWidth: 640, videoHeight: 360 });
  assert.equal(host.session.engine, engines[1]);
  assert.equal(host.session.canvas, canvases[1]);
  assert.equal(sent.at(-1).type, 'CONFIGURED');
});
