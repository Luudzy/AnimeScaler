import { NativeInlineHost } from './native-inline-host.js';

const PORT_NAME = 'animescaler-native-inline-v1';
let active = null;

function endHost(state) {
  // Closing the Ports makes cleanup failure observable to the service worker without exposing engine details.
  void closeHost(state).catch(() => {});
}

async function closeHost(state) {
  if (state.closing) return state.closing;
  state.closed = true;
  state.input.onMessage.removeListener(state.attach);
  state.closing = state.host.disconnect().finally(() => {
    if (active === state) active = null;
    try { state.input.disconnect(); } catch { /* Already disconnected. */ }
    try { state.output?.disconnect(); } catch { /* Already disconnected. */ }
  });
  return state.closing;
}

chrome.runtime.onConnect.addListener(input => {
  if (input.name !== PORT_NAME) return;
  const host = new NativeInlineHost({ send: message => {
    const state = active;
    if (!state || state.host !== host || state.closed || !state.output) return;
    try { state.output.postMessage(message); }
    catch { endHost(state); }
  } });
  try { host.validateSender(input.sender); }
  catch { input.disconnect(); return; }
  if (active) { input.disconnect(); return; }
  const state = active = { host, input, output: null, closed: false };
  state.attach = message => {
    if (state.closed || state.output) return;
    if (message?.protocol !== 1 || message.type !== 'HOST_ATTACH' ||
        typeof message.connectionId !== 'string' || !message.connectionId || message.connectionId.length > 256) {
      endHost(state);
      return;
    }
    input.onMessage.removeListener(state.attach);
    try {
      // The reverse Port authenticates this document through browser-owned sender metadata.
      state.output = chrome.runtime.connect({ name: PORT_NAME });
      state.output.onDisconnect.addListener(() => {
        void chrome.runtime.lastError;
        endHost(state);
      });
      // Attach after this event dispatch: HOST_ATTACH must never reach the session message handler.
      queueMicrotask(() => {
        if (state.closed) return;
        try {
          host.attachPort(input);
          state.output.postMessage({ protocol: 1, type: 'HOST_CONNECTED', connectionId: message.connectionId });
        } catch { endHost(state); }
      });
    } catch { endHost(state); }
  };
  input.onMessage.addListener(state.attach);
  input.onDisconnect.addListener(() => { endHost(state); });
});

// READY is sent only after the receiving onConnect listener is registered.
chrome.runtime.sendMessage({ protocol: 1, type: 'NATIVE_INLINE_HOST_READY' })
  .then(reply => { if (reply?.ok !== true && active) endHost(active); })
  .catch(() => { if (active) endHost(active); });
