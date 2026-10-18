import { normalizeSettings } from './src/core/settings.js';
import { startManualViewer, waitForManualViewerStart } from './src/core/start-viewer.js';
import { activeViewerKey, viewerSessionKey } from './src/core/viewer-session.js';
import { createDiagnosticStore } from './src/core/diagnostic-log.js';
import { restoreExclusiveSource } from './src/core/exclusive-viewer.js';
import { isNativeMessageShape } from './src/core/native-protocol.js';
import { isStaleOrigin } from './src/core/session-diagnostics.js';

const diagnostics = createDiagnosticStore(chrome.storage.local);

// pagehide may be interrupted; the worker also owns restoration on tab close.
chrome.tabs.onRemoved.addListener(tabId => {
  void (async () => {
    const stored = await chrome.storage.session.get(null);
    for (const [key, session] of Object.entries(stored)) {
      if (!key.startsWith('animeScalerViewer:') ||
          (session.viewerTabId !== tabId && session.sourceTabId !== tabId)) continue;
      await restoreExclusiveSource(session);
      await chrome.storage.session.remove([key, `animeScalerViewerActive:${session.sourceTabId}`]);
      if (session.sourceTabId === tabId && session.viewerWindowId != null) {
        try { await chrome.windows.remove(session.viewerWindowId); } catch { /* Window already closed. */ }
      }
    }
  })().catch(() => {});
});

async function activeViewerForTab(tabId) {
  if (!Number.isInteger(tabId)) return null;
  const activeKey = activeViewerKey(tabId);
  const stored = await chrome.storage.session.get(activeKey);
  const active = stored[activeKey];
  if (!active || !Number.isInteger(active.viewerWindowId)) return null;
  try {
    const window = await chrome.windows.get(active.viewerWindowId);
    if (window?.id === active.viewerWindowId) return active;
  } catch { /* Remove stale session state below. */ }
  const sessionKey = viewerSessionKey(active.sessionId);
  const stale = await chrome.storage.session.get(sessionKey);
  if (stale[sessionKey]) await restoreExclusiveSource(stale[sessionKey]);
  await chrome.storage.session.remove([activeKey, sessionKey]);
  return null;
}

async function stopViewerForTab(tabId) {
  await waitForManualViewerStart(tabId);
  const active = await activeViewerForTab(tabId);
  if (!active) return { state: 'stopped' };
  const key = viewerSessionKey(active.sessionId);
  const stored = await chrome.storage.session.get(key);
  const session = stored[key];
  if (session) await restoreExclusiveSource(session);
  try { await chrome.windows.remove(active.viewerWindowId); } catch { /* Window already closed. */ }
  await chrome.storage.session.remove([key, activeViewerKey(tabId)]);
  return { state: 'stopped', captureMode: 'viewer' };
}

// Window creation can close the popup; the worker owns the complete startup.
chrome.runtime.onMessage.addListener((message, sender, respond) => {
  if (sender.id === chrome.runtime.id && message?.type?.startsWith('ANIMESCALER_DIAGNOSTIC')) {
    let operation;
    if (message.type === 'ANIMESCALER_DIAGNOSTIC' && message.event) operation = diagnostics.append(message.event);
    else if (sender.url === chrome.runtime.getURL('popup/popup.html')) {
      if (message.type === 'ANIMESCALER_DIAGNOSTIC_EXPORT') operation = diagnostics.read();
      if (message.type === 'ANIMESCALER_DIAGNOSTIC_CLEAR') operation = diagnostics.clear();
    }
    if (!operation) return;
    operation.then(data => respond({ok: true, data}), error => respond({ok: false, error: error.message}));
    return true;
  }
  if (sender.id === chrome.runtime.id && sender.url === chrome.runtime.getURL('popup/popup.html') &&
      message?.type === 'ANIMESCALER_VIEWER_STATUS') {
    activeViewerForTab(message.sourceTabId).then(active => respond(active
      ? { state: 'running', captureMode: 'viewer', viewer: active }
      : { state: 'stopped' }), () => respond({ state: 'stopped' }));
    return true;
  }
  if (sender.id === chrome.runtime.id && sender.url === chrome.runtime.getURL('popup/popup.html') &&
      message?.type === 'ANIMESCALER_STOP_VIEWER') {
    stopViewerForTab(message.sourceTabId).then(respond, error => respond({
      state: 'error', captureMode: 'viewer', error: { code: 'VIEWER_STOP_FAILED', message: error.message },
    }));
    return true;
  }
  if (message?.type !== 'ANIMESCALER_START_VIEWER') return;
  if (sender.id !== chrome.runtime.id || sender.url !== chrome.runtime.getURL('popup/popup.html')) return;
  startManualViewer(message.tab, message.target, normalizeSettings(message.settings), message.frames)
    .then(respond, error => {
      void diagnostics.append({type: 'error', session: 'viewer-start', data: {code: 'VIEWER_START_FAILED', message: error.message}}).catch(() => {});
      respond({state:'error', captureMode:'viewer', error:{code:'VIEWER_START_FAILED', message:error.message}});
    });
  return true;
});

chrome.runtime.onInstalled.addListener(async () => {
  const { animeScalerSettings } = await chrome.storage.local.get('animeScalerSettings');
  await chrome.storage.local.set({ animeScalerSettings: normalizeSettings(animeScalerSettings) });
});

const NATIVE_PORT_NAME = 'animescaler-native-inline-v1';
const NATIVE_HOST_URL = chrome.runtime.getURL('native-inline.html');
const NATIVE_HANDSHAKE_MS = 10000;
const NATIVE_INPUT_TYPES = new Set(['START', 'FRAME_IN', 'CONFIGURE', 'STOP']);
const NATIVE_OUTPUT_TYPES = new Set(['STARTED', 'CONFIGURED', 'FRAME_OUT', 'FRAME_ACK', 'STOPPED', 'ERROR']);
let nativeRelay = null;

function isNativeHostSender(sender) {
  return sender?.id === chrome.runtime.id && sender.url === NATIVE_HOST_URL && sender.tab === undefined;
}

function isNativeContentSender(sender) {
  if (sender?.id !== chrome.runtime.id || !Number.isInteger(sender.tab?.id) || sender.tab.id < 0 ||
      !Number.isInteger(sender.frameId) || sender.frameId < 0 || typeof sender.url !== 'string') return false;
  // Extension pages are not content clients. Inherited about:blank/srcdoc frames retain the web origin.
  try {
    const protocol = new URL(sender.url).protocol;
    return ['http:', 'https:'].includes(protocol) || (protocol === 'about:' &&
      ['http:', 'https:'].includes(new URL(sender.origin).protocol));
  } catch { return false; }
}

function nativeEnvelope(message) {
  return message?.protocol === 1 && typeof message.sessionId === 'string' &&
    message.sessionId.length > 0 && message.sessionId.length <= 256;
}

function nativeError(state, code, message = null, staleOrigin = 'background_relay') {
  const stale = code === 'STALE_SESSION' && isStaleOrigin(staleOrigin) ? {
    staleOrigin,
    ...(typeof message?.type === 'string' ? { messageType: message.type } : {}),
    ...(Number.isSafeInteger(message?.sequence) ? { sequence: message.sequence } : {}),
    ...(Number.isSafeInteger(message?.revision) ? { revision: message.revision } : {}),
  } : {};
  try {
    state.client.postMessage({ protocol: 1, type: 'ERROR', sessionId: state.sessionId ?? '', code, ...stale });
    if (message?.type === 'FRAME_IN') state.client.postMessage({ protocol: 1, type: 'FRAME_ACK',
      sessionId: state.sessionId ?? '', sequence: message.sequence, accepted: false, code, ...stale });
  } catch { /* Client already disconnected. */ }
}

function closeNativeRelay(state) {
  if (state.closing) return state.closing;
  state.closed = true;
  clearTimeout(state.timer);
  state.rejectReady?.(new Error('HOST_UNAVAILABLE'));
  state.rejectConnected?.(new Error('HOST_UNAVAILABLE'));
  state.client.onMessage.removeListener(state.onClientMessage);
  state.client.onDisconnect.removeListener(state.onClientDisconnect);
  state.input?.disconnect();
  state.output?.disconnect();
  state.closing = (async () => {
    let cleaned = false;
    try {
      // createDocument cannot be aborted; a late-created owned document must still be closed.
      if (state.creating) await state.creating.catch(() => {});
      if (state.ownsDocument) await chrome.offscreen.closeDocument();
      cleaned = true;
    } finally {
      if (state.stopBeforeReady) {
        try { state.client.postMessage(cleaned
          ? { protocol: 1, type: 'STOPPED', sessionId: state.sessionId }
          : { protocol: 1, type: 'ERROR', sessionId: state.sessionId, code: 'HOST_UNAVAILABLE' }); }
        catch { /* The client may have disconnected while Chrome closed the document. */ }
      }
      state.client.disconnect();
      if (nativeRelay === state) nativeRelay = null;
    }
  })();
  return state.closing;
}

function failNativeRelay(state, code, staleOrigin = 'background_relay', message = null) {
  if (state.closed) return;
  nativeError(state, code, message, staleOrigin);
  void closeNativeRelay(state).catch(() => { /* Ports are closed even if Chrome rejects document cleanup. */ });
}

async function openNativeRelay(state) {
  if (!chrome.offscreen?.createDocument || !chrome.runtime.getContexts) throw new Error('HOST_UNAVAILABLE');
  const ready = new Promise((resolve, reject) => { state.resolveReady = resolve; state.rejectReady = reject; });
  // Observe rejection immediately, including when creation fails before the READY wait.
  ready.catch(() => {});
  state.timer = setTimeout(() => failNativeRelay(state, 'HOST_UNAVAILABLE'), NATIVE_HANDSHAKE_MS);
  const documents = await chrome.runtime.getContexts({ contextTypes: ['OFFSCREEN_DOCUMENT'] });
  if (state.closed) return;
  if (documents.length) throw new Error('HOST_UNAVAILABLE');
  state.creating = chrome.offscreen.createDocument({ url: 'native-inline.html', reasons: ['WORKERS'],
    justification: 'Run native inference Workers on isolated PNG frames in extension context.' })
    .then(() => { state.ownsDocument = true; });
  await state.creating;
  if (state.closed) return;
  await ready;
  if (state.closed) return;
  const connected = new Promise((resolve, reject) => { state.resolveConnected = resolve; state.rejectConnected = reject; });
  connected.catch(() => {});
  state.input = chrome.runtime.connect({ name: NATIVE_PORT_NAME });
  state.input.onDisconnect.addListener(() => {
    void chrome.runtime.lastError;
    if (!state.closed) failNativeRelay(state, 'HOST_UNAVAILABLE');
  });
  state.input.postMessage({ protocol: 1, type: 'HOST_ATTACH', connectionId: state.connectionId });
  await connected;
  if (state.closed) return;
  clearTimeout(state.timer);
  state.connected = true;
}

chrome.runtime.onMessage.addListener((message, sender, respond) => {
  if (message?.type !== 'NATIVE_INLINE_HOST_READY') return;
  const state = nativeRelay;
  if (message.protocol !== 1 || !isNativeHostSender(sender) || !state || state.closed || !state.creating) {
    respond({ ok: false });
    return;
  }
  state.resolveReady();
  respond({ ok: true });
});

chrome.runtime.onConnect.addListener(port => {
  if (port.name !== NATIVE_PORT_NAME) return;
  if (isNativeHostSender(port.sender)) {
    const state = nativeRelay;
    if (!state || state.closed || !state.input || state.output) { port.disconnect(); return; }
    state.output = port;
    port.onDisconnect.addListener(() => {
      void chrome.runtime.lastError;
      if (!state.closed) failNativeRelay(state, 'HOST_UNAVAILABLE');
    });
    port.onMessage.addListener(message => {
      if (state.closed) return;
      if (!state.connected) {
        if (message?.protocol !== 1 || message.type !== 'HOST_CONNECTED' || message.connectionId !== state.connectionId) {
          failNativeRelay(state, 'HOST_UNAVAILABLE');
          return;
        }
        state.resolveConnected();
        return;
      }
      if (!nativeEnvelope(message) || message.sessionId !== state.sessionId || !NATIVE_OUTPUT_TYPES.has(message.type) ||
          ((message.type === 'ERROR' || message.type === 'FRAME_ACK') && message.code === 'STALE_SESSION' &&
            message.staleOrigin !== undefined && !isStaleOrigin(message.staleOrigin))) {
        failNativeRelay(state, 'STALE_SESSION', 'background_relay', message);
        return;
      }
      // Forward the original message. The service worker never reads, stores or modifies PNG data.
      try { state.client.postMessage(message); }
      catch { void closeNativeRelay(state).catch(() => {}); return; }
      if (message.type === 'FRAME_ACK' && state.pendingFrames > 0) state.pendingFrames--;
      if (message.type === 'ERROR') {
        state.failed = true;
        clearTimeout(state.timer);
        state.timer = setTimeout(() => failNativeRelay(state, 'HOST_UNAVAILABLE'), NATIVE_HANDSHAKE_MS);
      }
      if (message.type === 'STOPPED' || (state.failed && state.pendingFrames === 0)) {
        void closeNativeRelay(state).catch(() => {});
      }
    });
    return;
  }
  if (!isNativeContentSender(port.sender)) { port.disconnect(); return; }
  if (nativeRelay) {
    // Bind the rejection to the attempted session without keeping its payload.
    const rejectClient = message => {
      clearTimeout(timer);
      port.onMessage.removeListener(rejectClient);
      nativeError({ client: port, sessionId: nativeEnvelope(message) ? message.sessionId : '' }, 'NATIVE_SESSION_BUSY', message);
      port.disconnect();
    };
    const timer = setTimeout(() => { port.onMessage.removeListener(rejectClient); port.disconnect(); }, NATIVE_HANDSHAKE_MS);
    port.onMessage.addListener(rejectClient);
    port.onDisconnect.addListener(() => { clearTimeout(timer); port.onMessage.removeListener(rejectClient); });
    return;
  }
  const state = nativeRelay = { client: port, sessionId: null, connectionId: crypto.randomUUID(),
    closed: false, connected: false, ownsDocument: false, pendingFrames: 0 };
  state.onClientDisconnect = () => { void closeNativeRelay(state).catch(() => {}); };
  state.onClientMessage = async message => {
    if (state.closed) return;
    if (!nativeEnvelope(message) || !NATIVE_INPUT_TYPES.has(message.type)) { failNativeRelay(state, 'BACKEND_UNAVAILABLE'); return; }
    if (state.sessionId === null) {
      if (message.type !== 'START') { failNativeRelay(state, 'STALE_SESSION'); return; }
      state.sessionId = message.sessionId;
    } else if (message.sessionId !== state.sessionId) { failNativeRelay(state, 'STALE_SESSION'); return; }
    if (!isNativeMessageShape(message)) { failNativeRelay(state, 'SOURCE_NOT_ISOLATED'); return; }
    if (state.failed || state.stopping) { nativeError(state, 'NATIVE_SESSION_BUSY', message); return; }
    if (message.type === 'STOP' && !state.connected) {
      state.stopping = true;
      state.stopBeforeReady = true;
      try { await closeNativeRelay(state); }
      catch { /* Cleanup failure cannot resume the session. */ }
      return;
    }
    if (message.type !== 'START' && !state.connected) {
      nativeError(state, 'HOST_UNAVAILABLE', message);
      void closeNativeRelay(state).catch(() => {});
      return;
    }
    try {
      if (message.type === 'START') {
        if (state.startRequested) { nativeError(state, 'NATIVE_SESSION_BUSY'); return; }
        state.startRequested = true;
        await state.opening;
      }
      if (state.closed) return;
      if (message.type === 'FRAME_IN') state.pendingFrames++;
      if (message.type === 'STOP') {
        state.stopping = true;
        clearTimeout(state.timer);
        state.timer = setTimeout(() => failNativeRelay(state, 'HOST_UNAVAILABLE'), NATIVE_HANDSHAKE_MS);
      }
      state.input.postMessage(message);
    } catch { failNativeRelay(state, 'HOST_UNAVAILABLE'); }
  };
  port.onMessage.addListener(state.onClientMessage);
  port.onDisconnect.addListener(state.onClientDisconnect);
  state.opening = openNativeRelay(state);
  state.opening.catch(() => failNativeRelay(state, 'HOST_UNAVAILABLE'));
});
