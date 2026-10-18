import test from 'node:test';
import assert from 'node:assert/strict';
import { NativeHostClient } from '../src/content/native-host-client.js';
import { sessionTag } from '../src/core/session-diagnostics.js';

class EventSlot {
  listeners = new Set();
  addListener(listener) { this.listeners.add(listener); }
  removeListener(listener) { this.listeners.delete(listener); }
  emit(value) { for (const listener of this.listeners) listener(value); }
}

class FakePort {
  constructor() {
    this.name = 'animescaler-native-inline-v1';
    this.onMessage = new EventSlot();
    this.onDisconnect = new EventSlot();
    this.sent = [];
  }
  postMessage(message) { this.sent.push(message); }
  disconnect() { this.onDisconnect.emit(); }
  receive(message) { this.onMessage.emit(message); }
}

function harness(options = {}) {
  const port = new FakePort();
  const client = new NativeHostClient({ runtime: { connect: () => port }, ...options });
  return { client, port };
}

const start = { backend: 'rocm', model: 'zenkai-1-0', settings: { backend: 'rocm', model: 'zenkai-1-0' },
  captureMode: 'direct-png', sourceWidth: 320, sourceHeight: 180 };

test('client rejects frames during stop and stop failures share one handled promise', async () => {
  const { client, port } = harness();
  const starting = client.start({ ...start, sessionId: 'session-1' });
  port.receive({ protocol: 1, type: 'STARTED', sessionId: 'session-1' });
  await starting;
  const stopping = client.stop();
  const repeated = client.stop();
  const rejection = assert.rejects(stopping, { code: 'HOST_UNAVAILABLE' });
  const repeatedRejection = assert.rejects(repeated, { code: 'HOST_UNAVAILABLE' });
  const frameRejection = assert.rejects(client.processFrame({ sequence: 1, mediaTime: 0, width: 1, height: 1,
    mime: 'image/png', dataUrl: 'data:image/png;base64,AA==' }), { code: 'STALE_SESSION' });
  await new Promise(resolve => setImmediate(resolve));
  port.disconnect();
  await rejection;
  await repeatedRejection;
  await frameRejection;
  assert.equal(repeated, stopping);
  assert.equal(port.onMessage.listeners.size, 0);
});
const output = (sequence, mediaTime) => ({ protocol: 1, type: 'FRAME_OUT', sessionId: 'session-1',
  sequence, mediaTime, width: 320, height: 180, mime: 'image/png', dataUrl: 'data:image/png;base64,AA==' });

// ID 48: 300 ms is an admission budget, not a timeout for already-running inference.
for (const age of [299, 300]) test(`client pending admission boundary ${age} ms releases request once`, async t => {
  let now = 0;
  t.mock.method(performance, 'now', () => now);
  const { client, port } = harness();
  t.after(() => client.fail(Object.assign(new Error('test cleanup'), { code: 'STALE_SESSION' })));
  const starting = client.start({ ...start, sessionId: 'session-1' });
  port.receive({ protocol: 1, type: 'STARTED', sessionId: 'session-1' });
  await starting;
  const frame = sequence => ({ sequence, mediaTime: sequence, width: 1, height: 1,
    mime: 'image/png', dataUrl: 'data:image/png;base64,AA==' });
  const first = client.processFrame(frame(1));
  const pending = client.processFrame(frame(2));
  let rejected;
  const settled = pending.then(() => {}, error => { rejected = error; });
  now = age;
  port.receive(output(1, 1));
  port.receive({ protocol: 1, type: 'FRAME_ACK', sessionId: 'session-1', sequence: 1, accepted: true });
  await first;
  if (age === 299) {
    port.receive(output(2, 2));
    port.receive({ protocol: 1, type: 'FRAME_ACK', sessionId: 'session-1', sequence: 2, accepted: true });
  }
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(port.sent.filter(message => message.type === 'FRAME_IN').map(message => message.sequence), age === 299 ? [1, 2] : [1]);
  assert.equal(rejected?.code, age === 299 ? undefined : 'STALE_FRAME');
  await settled;
  assert.equal(client.pending, null);
  assert.equal(client.inFlight, null);
  assert.equal(client.closed, false);
  const next = client.processFrame(frame(3));
  port.receive(output(3, 3));
  port.receive({ protocol: 1, type: 'FRAME_ACK', sessionId: 'session-1', sequence: 3, accepted: true });
  await next;
  const stopping = client.stop();
  port.receive({ protocol: 1, type: 'STOPPED', sessionId: 'session-1' });
  await stopping;
});

test('client retains only safe output fields and detached stats', async () => {
  const { client, port } = harness();
  const starting = client.start({ ...start, sessionId: 'session-1' });
  port.receive({ protocol: 1, type: 'STARTED', sessionId: 'session-1' });
  await starting;
  const processing = client.processFrame({ sequence: 1, mediaTime: 1, width: 1, height: 1,
    mime: 'image/png', dataUrl: 'data:image/png;base64,AA==' });
  const message = { ...output(1, 1), token: 'secret', url: 'private', stats: {
    inferenceMs: 4, temporal: { acceptedResults: 2, token: 'secret' }, token: 'secret',
  } };
  port.receive(message);
  message.stats.temporal.acceptedResults = 99;
  port.receive({ protocol: 1, type: 'FRAME_ACK', sessionId: 'session-1', sequence: 1, accepted: true });
  const result = await processing;
  assert.equal('token' in result, false);
  assert.equal('url' in result, false);
  assert.deepEqual(result.stats, { inferenceMs: 4, temporal: { acceptedResults: 2 } });
  const stopping = client.stop();
  port.receive({ protocol: 1, type: 'STOPPED', sessionId: 'session-1' });
  await stopping;
});

test('client output carries local freshness deadline without adding protocol fields', async t => {
  let now = 100;
  t.mock.method(performance, 'now', () => now);
  const { client, port } = harness();
  const starting = client.start({ ...start, sessionId: 'session-1' });
  port.receive({ protocol: 1, type: 'STARTED', sessionId: 'session-1' });
  await starting;
  const frame = client.processFrame({ sequence: 1, mediaTime: 1, width: 1, height: 1,
    mime: 'image/png', dataUrl: 'data:image/png;base64,AA==' });
  now = 600;
  port.receive({ ...output(1, 1), expiresAt: 999999 });
  port.receive({ protocol: 1, type: 'FRAME_ACK', sessionId: 'session-1', sequence: 1, accepted: true });
  const result = await frame;
  assert.equal(result.expiresAt, 400, 'deadline must start on admission, not reset on output arrival');
  assert.equal('expiresAt' in port.sent.at(-1), false);
  assert.equal(Object.keys(result).includes('expiresAt'), false);
  assert.equal(client.closed, false);
  const stopping = client.stop();
  port.receive({ protocol: 1, type: 'STOPPED', sessionId: 'session-1' });
  await stopping;
});

test('client newest-wins replaces pending timestamp and receives nonterminal stale ACK', async t => {
  let now = 0;
  t.mock.method(performance, 'now', () => now);
  const { client, port } = harness();
  const starting = client.start({ ...start, sessionId: 'session-1' });
  port.receive({ protocol: 1, type: 'STARTED', sessionId: 'session-1' });
  await starting;
  const frame = sequence => ({ sequence, mediaTime: sequence, width: 1, height: 1,
    mime: 'image/png', dataUrl: 'data:image/png;base64,AA==' });
  const first = client.processFrame(frame(1));
  const displaced = client.processFrame(frame(2));
  const rejection = assert.rejects(displaced, { code: 'STALE_FRAME' });
  now = 250;
  const newest = client.processFrame(frame(3));
  const newestRejection = assert.rejects(newest, { code: 'STALE_FRAME' });
  await rejection;
  now = 500;
  port.receive(output(1, 1));
  port.receive({ protocol: 1, type: 'FRAME_ACK', sessionId: 'session-1', sequence: 1, accepted: true });
  await first;
  assert.deepEqual(port.sent.filter(message => message.type === 'FRAME_IN').map(message => message.sequence), [1, 3]);
  port.receive({ protocol: 1, type: 'FRAME_ACK', sessionId: 'session-1', sequence: 3, accepted: false, code: 'STALE_FRAME' });
  await newestRejection;
  assert.equal(client.inFlight, null);
  assert.equal(client.pending, null);
  assert.equal(client.closed, false);
  const stopping = client.stop();
  port.receive({ protocol: 1, type: 'STOPPED', sessionId: 'session-1' });
  await stopping;
});

test('NativeHostClient correlates start, output, ACK and stop with one session', async () => {
  const { client, port } = harness();
  const starting = client.start({ ...start, sessionId: 'session-1' });
  assert.equal(port.sent[0].type, 'START');
  port.receive({ protocol: 1, type: 'STARTED', sessionId: 'session-1', backend: 'rocm' });
  await starting;

  const frame = client.processFrame({ sequence: 1, mediaTime: 1.25, width: 320, height: 180,
    mime: 'image/png', dataUrl: 'data:image/png;base64,AA==' });
  assert.equal(port.sent.at(-1).type, 'FRAME_IN');
  port.receive(output(1, 1.25));
  port.receive({ protocol: 1, type: 'FRAME_ACK', sessionId: 'session-1', sequence: 1, accepted: true });
  assert.deepEqual(await frame, output(1, 1.25));

  const configuring = client.configure({ settings: { backend: 'rocm', model: 'zenkai-1-0', preset: 'quality' },
    adaptiveLevel: 2, sourceWidth: 640, sourceHeight: 360 });
  assert.deepEqual(port.sent.at(-1), { protocol: 1, type: 'CONFIGURE', sessionId: 'session-1', revision: 1,
    settings: { backend: 'rocm', model: 'zenkai-1-0', preset: 'quality' },
    adaptiveLevel: 2, sourceWidth: 640, sourceHeight: 360 });
  port.receive({ protocol: 1, type: 'CONFIGURED', sessionId: 'session-1', revision: 1, backend: 'rocm', model: 'zenkai-1-0' });
  await configuring;

  const stopping = client.stop('test');
  assert.equal(port.sent.at(-1).type, 'STOP');
  port.receive({ protocol: 1, type: 'STOPPED', sessionId: 'session-1' });
  await stopping;
});

test('NativeHostClient keeps one frame in flight and sends only newest pending frame', async () => {
  const { client, port } = harness();
  const starting = client.start({ ...start, sessionId: 'session-1' });
  port.receive({ protocol: 1, type: 'STARTED', sessionId: 'session-1', backend: 'rocm' });
  await starting;

  const first = client.processFrame({ sequence: 1, mediaTime: 1, width: 1, height: 1, mime: 'image/png', dataUrl: 'data:image/png;base64,AA==' });
  const displaced = client.processFrame({ sequence: 2, mediaTime: 2, width: 1, height: 1, mime: 'image/png', dataUrl: 'data:image/png;base64,AA==' });
  const latest = client.processFrame({ sequence: 3, mediaTime: 3, width: 1, height: 1, mime: 'image/png', dataUrl: 'data:image/png;base64,AA==' });
  await assert.rejects(displaced, { code: 'STALE_FRAME' });
  assert.deepEqual(port.sent.filter(message => message.type === 'FRAME_IN').map(message => message.sequence), [1]);

  port.receive(output(1, 1));
  port.receive({ protocol: 1, type: 'FRAME_ACK', sessionId: 'session-1', sequence: 1, accepted: true });
  await first;
  assert.deepEqual(port.sent.filter(message => message.type === 'FRAME_IN').map(message => message.sequence), [1, 3]);
  port.receive(output(3, 3));
  port.receive({ protocol: 1, type: 'FRAME_ACK', sessionId: 'session-1', sequence: 3, accepted: true });
  await latest;
  const stopping = client.stop('test');
  port.receive({ protocol: 1, type: 'STOPPED', sessionId: 'session-1' });
  await stopping;
});

test('NativeHostClient rejects stale output and fails requests on disconnect', async () => {
  const { client, port } = harness();
  const starting = client.start({ ...start, sessionId: 'session-1' });
  port.receive({ protocol: 1, type: 'STARTED', sessionId: 'session-1', backend: 'rocm' });
  await starting;
  const frame = client.processFrame({ sequence: 4, mediaTime: 4, width: 1, height: 1, mime: 'image/png', dataUrl: 'data:image/png;base64,AA==' });
  port.receive(output(3, 3));
  await assert.rejects(frame, { code: 'STALE_SESSION' });
  const second = harness();
  const secondStart = second.client.start({ ...start, sessionId: 'session-1' });
  second.port.receive({ protocol: 1, type: 'STARTED', sessionId: 'session-1', backend: 'rocm' });
  await secondStart;
  const next = second.client.processFrame({ sequence: 5, mediaTime: 5, width: 1, height: 1,
    mime: 'image/png', dataUrl: 'data:image/png;base64,AA==' });
  second.port.disconnect();
  await assert.rejects(next, { code: 'HOST_UNAVAILABLE' });
});

test('NativeHostClient retains safe stale origin and frame correlation from host ERROR', async () => {
  const { client, port } = harness();
  const starting = client.start({ ...start, sessionId: 'session-1' });
  port.receive({ protocol: 1, type: 'STARTED', sessionId: 'session-1' });
  await starting;
  const frame = client.processFrame({ sequence: 12, mediaTime: 0.5, width: 1, height: 1,
    mime: 'image/png', dataUrl: 'data:image/png;base64,AA==' });
  const rejection = assert.rejects(frame, error => {
    assert.equal(error.code, 'STALE_SESSION');
    assert.deepEqual(error.diagnostic, {
      staleOrigin: 'native_host', sessionTag: sessionTag('session-1'), messageType: 'FRAME_IN', sequence: 12,
      hasInFlight: true, hasPending: false,
    });
    return true;
  });
  port.receive({ protocol: 1, type: 'ERROR', sessionId: 'session-1', code: 'STALE_SESSION',
    staleOrigin: 'native_host', messageType: 'FRAME_IN', sequence: 12, token: 'secret' });
  await rejection;
  assert.equal(JSON.stringify(client).includes('secret'), false);
});

test('NativeHostClient rejects PNG data above the 16 MiB binary limit before posting', async () => {
  const { client, port } = harness();
  const starting = client.start({ ...start, sessionId: 'session-1' });
  port.receive({ protocol: 1, type: 'STARTED', sessionId: 'session-1', backend: 'rocm' });
  await starting;
  const oversized = `data:image/png;base64,${'A'.repeat(Math.ceil(16 * 1024 * 1024 / 3) * 4 + 4)}`;
  await assert.rejects(client.processFrame({ sequence: 1, mediaTime: 1, width: 1, height: 1,
    mime: 'image/png', dataUrl: oversized }), { code: 'FRAME_TOO_LARGE' });
  assert.equal(port.sent.some(message => message.type === 'FRAME_IN'), false);
});

test('client bounds input axes and area before posting', async () => {
  const { client, port } = harness();
  const starting = client.start({ ...start, sessionId: 'session-1' });
  port.receive({ protocol: 1, type: 'STARTED', sessionId: 'session-1' });
  await starting;
  for (const [width, height] of [[8193, 1], [7680, 4321]]) {
    await assert.rejects(client.processFrame({ sequence: 1, mediaTime: 0, width, height,
      mime: 'image/png', dataUrl: 'data:image/png;base64,AA==' }), { code: 'FRAME_TOO_LARGE' });
  }
  assert.equal(port.sent.some(message => message.type === 'FRAME_IN'), false);
  const stopping = client.stop();
  port.receive({ protocol: 1, type: 'STOPPED', sessionId: 'session-1' });
  await stopping;
});
