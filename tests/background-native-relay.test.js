import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import { normalizeSettings } from '../src/core/settings.js';
import { isNativeMessageShape } from '../src/core/native-protocol.js';
import { isStaleOrigin } from '../src/core/session-diagnostics.js';

class EventSlot {
  listeners = new Set();
  addListener(listener) { this.listeners.add(listener); }
  removeListener(listener) { this.listeners.delete(listener); }
  async dispatch(...args) {
    for (const listener of [...this.listeners]) await listener(...args);
  }
}

class Port {
  onMessage = new EventSlot();
  onDisconnect = new EventSlot();
  sent = [];
  disconnected = false;
  constructor(name, sender) { this.name = name; this.sender = sender; }
  postMessage(message) {
    if (this.disconnected) throw new Error('port disconnected');
    this.sent.push(message);
  }
  receive(message) { return this.onMessage.dispatch(message); }
  async disconnect() {
    if (this.disconnected) return;
    this.disconnected = true;
    await this.onDisconnect.dispatch();
  }
}

async function createRelay({ failCreate = false } = {}) {
  const runtimeOnMessage = new EventSlot();
  const runtimeOnConnect = new EventSlot();
  const clientPort = new Port('animescaler-native-inline-v1', {
    id: 'extension-id', tab: { id: 41 }, frameId: 0, url: 'https://video.test/watch',
  });
  const inputPorts = [];
  const created = { closeDocument: 0 };
  const chrome = {
    storage: { local: {}, session: { get: async () => ({}), remove: async () => {} } },
    tabs: { onRemoved: new EventSlot() },
    runtime: {
      id: 'extension-id',
      onMessage: runtimeOnMessage,
      onConnect: runtimeOnConnect,
      onInstalled: new EventSlot(),
      getURL: path => `chrome-extension://extension-id/${path}`,
      getContexts: async () => [],
      connect: () => {
        const port = new Port('animescaler-native-inline-v1', {
          id: 'extension-id', url: 'chrome-extension://extension-id/background.js',
        });
        inputPorts.push(port);
        return port;
      },
    },
    offscreen: {
      createDocument: async () => {
        if (failCreate) throw new Error('offscreen create failed');
        await runtimeOnMessage.dispatch({ protocol: 1, type: 'NATIVE_INLINE_HOST_READY' }, {
          id: 'extension-id', url: 'chrome-extension://extension-id/native-inline.html',
        }, reply => { created.readyReply = reply; });
      },
      closeDocument: async () => { created.closeDocument++; },
    },
  };
  const source = (await readFile(new URL('../background.js', import.meta.url), 'utf8'))
    .replace(/^import .*;\r?\n/gm, '') + '\nglobalThis.__relayState = () => nativeRelay;';
  const context = vm.createContext({
    chrome, URL, crypto: { randomUUID: () => 'connection-1' },
    setTimeout, clearTimeout,
    normalizeSettings, startManualViewer: async () => ({}),
    createDiagnosticStore: () => ({ append: async () => {}, read: async () => [], clear: async () => {} }),
    restoreExclusiveSource: async () => {}, isNativeMessageShape, isStaleOrigin,
  });
  vm.runInContext(source, context);
  await runtimeOnConnect.dispatch(clientPort);
  return { context, clientPort, inputPorts, runtimeOnConnect, created };
}

async function waitFor(predicate, message) {
  for (let attempt = 0; attempt < 30; attempt++) {
    if (predicate()) return;
    await new Promise(resolve => setImmediate(resolve));
  }
  assert.fail(message);
}

async function startSession(relay) {
  const { clientPort, inputPorts, runtimeOnConnect } = relay;
  const starting = clientPort.receive({ protocol: 1, type: 'START', sessionId: 'session-1',
    backend: 'rocm', model: 'zenkai-1-0', settings: { backend: 'rocm', model: 'zenkai-1-0' },
    captureMode: 'direct-png', sourceWidth: 320, sourceHeight: 180 });
  await waitFor(() => inputPorts.length === 1 && inputPorts[0].sent.length === 1, 'offscreen attach was not sent');
  const input = inputPorts[0];
  const hostPort = new Port(input.name, {
    id: 'extension-id', url: 'chrome-extension://extension-id/native-inline.html',
  });
  await runtimeOnConnect.dispatch(hostPort);
  await hostPort.receive({ protocol: 1, type: 'HOST_CONNECTED', connectionId: 'connection-1' });
  await starting;
  assert.deepEqual(input.sent.map(message => message.type), ['HOST_ATTACH', 'START']);
  await hostPort.receive({ protocol: 1, type: 'STARTED', sessionId: 'session-1' });
  return { input, hostPort };
}

test('service worker relays STOP, closes owned offscreen document, and ignores late messages', async () => {
  const relay = await createRelay();
  const { clientPort, created, context } = relay;
  const { input, hostPort } = await startSession(relay);
  const stopping = clientPort.receive({ protocol: 1, type: 'STOP', sessionId: 'session-1', reason: 'user' });
  await stopping;
  assert.equal(input.sent.at(-1).type, 'STOP');
  await hostPort.receive({ protocol: 1, type: 'STOPPED', sessionId: 'session-1' });
  await waitFor(() => clientPort.disconnected, 'content Port remained open after STOP');
  assert.equal(created.closeDocument, 1);
  assert.equal(clientPort.sent.at(-1).type, 'STOPPED');
  const messagesAfterStop = clientPort.sent.length;
  await hostPort.receive({ protocol: 1, type: 'FRAME_ACK', sessionId: 'session-1', sequence: 1, accepted: true });
  assert.equal(clientPort.sent.length, messagesAfterStop);
  assert.equal(context.__relayState(), null);
  assert.equal(clientPort.onMessage.listeners.size, 0);
});

test('relay marks stale host session rejection with its origin', async () => {
  const relay = await createRelay();
  const { clientPort, context } = relay;
  const { hostPort } = await startSession(relay);
  await hostPort.receive({ protocol: 1, type: 'FRAME_ACK', sessionId: 'wrong-session', sequence: 2,
    accepted: false, code: 'STALE_SESSION' });
  const error = clientPort.sent.at(-1);
  assert.equal(error.type, 'ERROR');
  assert.equal(error.code, 'STALE_SESSION');
  assert.equal(error.staleOrigin, 'background_relay');
  assert.equal(error.messageType, 'FRAME_ACK');
  assert.equal(error.sequence, 2);
  await waitFor(() => context.__relayState() === null, 'stale relay remained active');
});

test('content Port disconnect cleans relay and offscreen document like a closed source tab', async () => {
  const relay = await createRelay();
  const { clientPort, created, context } = relay;
  const { hostPort } = await startSession(relay);
  await clientPort.disconnect();
  await waitFor(() => context.__relayState() === null, 'tab-close disconnect left the relay active');
  assert.equal(created.closeDocument, 1);
  const messagesAtClose = clientPort.sent.length;
  await hostPort.receive({ protocol: 1, type: 'FRAME_OUT', sessionId: 'session-1', sequence: 1 });
  assert.equal(clientPort.sent.length, messagesAtClose);
});

test('offscreen Port disconnect fails session and closes all relay endpoints', async () => {
  const relay = await createRelay();
  const { clientPort, created, context } = relay;
  const { hostPort } = await startSession(relay);
  await hostPort.disconnect();
  await waitFor(() => context.__relayState() === null, 'host disconnect left relay active');
  assert.equal(clientPort.disconnected, true);
  assert.equal(created.closeDocument, 1);
  assert.ok(clientPort.sent.some(message => message.type === 'ERROR' && message.code === 'HOST_UNAVAILABLE'));
  const messagesAtClose = clientPort.sent.length;
  await hostPort.receive({ protocol: 1, type: 'FRAME_ACK', sessionId: 'session-1', sequence: 1 });
  assert.equal(clientPort.sent.length, messagesAtClose);
});

test('failed offscreen startup reports error and closes client Port without leaked context', async () => {
  const relay = await createRelay({ failCreate: true });
  const { clientPort, created, context } = relay;
  await clientPort.receive({ protocol: 1, type: 'START', sessionId: 'session-1',
    backend: 'rocm', model: 'zenkai-1-0', settings: { backend: 'rocm', model: 'zenkai-1-0' },
    captureMode: 'direct-png', sourceWidth: 320, sourceHeight: 180 });
  await waitFor(() => clientPort.disconnected, 'failed start left content Port open');
  assert.equal(clientPort.sent.at(-1).type, 'ERROR');
  assert.equal(clientPort.sent.at(-1).code, 'HOST_UNAVAILABLE');
  assert.equal(created.closeDocument, 0);
  assert.equal(context.__relayState(), null);
  const messagesAfterFailure = clientPort.sent.length;
  await clientPort.receive({ protocol: 1, type: 'FRAME_IN', sessionId: 'session-1', sequence: 1 });
  assert.equal(clientPort.sent.length, messagesAfterFailure);
});
