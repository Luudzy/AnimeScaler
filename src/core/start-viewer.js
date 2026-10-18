import { chooseCaptureHostFrame } from './tab-capture.js';
import { activeViewerKey, buildViewerSession, viewerSessionKey } from './viewer-session.js';
import { expandExclusiveSource, restoreExclusiveSource, exclusiveFrameChain } from './exclusive-viewer.js';

function waitForViewerTab(chrome, tabId, expectedUrl, timeoutMs = 10000) {
  return new Promise((resolve, reject) => {
    let finished = false;
    const finish = error => {
      if (finished) return;
      finished = true;
      clearTimeout(timer);
      chrome.tabs.onUpdated.removeListener(onUpdated);
      chrome.tabs.onRemoved.removeListener(onRemoved);
      chrome.runtime.onMessage?.removeListener(onReady);
      if (error) reject(error); else resolve();
    };
    const check = tab => {
      if (tab.url === expectedUrl && tab.status === 'complete') finish();
    };
    // Chrome can omit tabs.url for our own extension page without "tabs";
    // webNavigation also returns null for that document. A message from the
    // exact top-level consumer verifies that its navigation has committed.
    const onReady = (message, sender, respond) => {
      if (message?.type !== 'ANIMESCALER_VIEWER_READY' || sender.id !== chrome.runtime.id
          || sender.tab?.id !== tabId || sender.frameId !== 0 || sender.url !== expectedUrl) return;
      finish(); respond?.({ready:true});
      return false;
    };
    const onUpdated = (id, change, tab) => { if (id === tabId) check(tab); };
    const onRemoved = id => { if (id === tabId) finish(new Error('O viewer foi fechado antes de iniciar a captura.')); };
    const timer = setTimeout(() => finish(new Error('O viewer não terminou de carregar a tempo.')), timeoutMs);
    // Subscribe before reading the tab so a navigation between these steps is not lost.
    chrome.tabs.onUpdated.addListener(onUpdated);
    chrome.tabs.onRemoved.addListener(onRemoved);
    chrome.runtime.onMessage?.addListener(onReady);
    chrome.tabs.get(tabId).then(check, finish);
  });
}

const startsBySourceTab = new Map();

export function waitForManualViewerStart(sourceTabId) {
  return startsBySourceTab.get(sourceTabId) ?? Promise.resolve();
}

export function startManualViewer(tab, target, settings, frames, chrome = globalThis.chrome) {
  const previous = waitForManualViewerStart(tab.id);
  const start = previous.catch(() => {}).then(() => startManualViewerOnce(tab, target, settings, frames, chrome));
  startsBySourceTab.set(tab.id, start);
  return start.finally(() => {
    if (startsBySourceTab.get(tab.id) === start) startsBySourceTab.delete(tab.id);
  });
}

async function startManualViewerOnce(tab, target, settings, frames, chrome) {
  const active = await chrome.storage.session.get(activeViewerKey(tab.id));
  const existing = active[activeViewerKey(tab.id)];
  if (existing?.viewerWindowId != null) {
    try {
      await chrome.windows.update(existing.viewerWindowId, { focused: true });
      return { state: 'running', captureMode: 'viewer', viewer: existing };
    } catch {
      const stale = await chrome.storage.session.get(viewerSessionKey(existing.sessionId));
      if (stale[viewerSessionKey(existing.sessionId)]) await restoreExclusiveSource(stale[viewerSessionKey(existing.sessionId)], chrome);
      await chrome.storage.session.remove([activeViewerKey(tab.id), viewerSessionKey(existing.sessionId)]);
    }
  }
  const host = chooseCaptureHostFrame(frames, target.frameId);
  if (!host) {
    return { state: 'error', captureMode: 'viewer', error: { code: 'VIEWER_TARGET_UNRESOLVED', message: 'Não foi possível resolver o iframe do player para o viewer.' } };
  }

  const sessionId = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  const viewerUrl = chrome.runtime.getURL(`viewer/viewer.html?session=${encodeURIComponent(sessionId)}`);
  let viewerWindow;
  let exclusiveFrames = [];
  try {
    if (settings.exclusiveViewer) exclusiveFrames = await expandExclusiveSource(tab.id, target.frameId, frames, sessionId, chrome);
    viewerWindow = await chrome.windows.create({
      url: viewerUrl,
      type: 'popup',
      focused: false,
      width: 1280,
      height: 800,
    });
    let viewerTabId = viewerWindow?.tabs?.[0]?.id;
    if (viewerTabId == null && viewerWindow?.id != null) {
      const tabs = await chrome.tabs.query({ windowId: viewerWindow.id });
      viewerTabId = tabs?.[0]?.id;
    }
    if (viewerTabId == null) throw new Error('A janela do viewer não retornou um tabId.');

    await waitForViewerTab(chrome, viewerTabId, viewerUrl);
    const streamId = await chrome.tabCapture.getMediaStreamId({
      targetTabId: tab.id,
      consumerTabId: viewerTabId,
    });
    const captureFrame = {
      frameId: host.frameId,
      parentFrameId: host.parentFrameId,
      url: host.url,
      targetFrameId: target.frameId,
    };
    const session = buildViewerSession({
      streamId,
      sourceTabId: tab.id,
      sourceWindowId: tab.windowId,
      viewerTabId,
      viewerWindowId: viewerWindow.id,
      captureFrame,
      settings,
    });
    const active = {
      sessionId,
      viewerWindowId: viewerWindow.id,
      viewerTabId,
      targetFrameId: target.frameId,
    };
    Object.assign(session, {sessionId, exclusiveFrames, captureChain:exclusiveFrameChain(frames,target.frameId),
      windowedBounds:{state:'normal',left:viewerWindow.left,top:viewerWindow.top,width:viewerWindow.width,height:viewerWindow.height}});
    await chrome.storage.session.set({
      [viewerSessionKey(sessionId)]: session,
      [activeViewerKey(tab.id)]: active,
    });
    await chrome.windows.update(viewerWindow.id, { focused: true, ...(settings.exclusiveViewer ? {state:'fullscreen'} : {}) });
    return { state: 'running', captureMode: 'viewer', viewer: active };
  } catch (error) {
    await restoreExclusiveSource({sourceTabId:tab.id, exclusiveFrames, sessionId}, chrome);
    try {
      await chrome.storage.session.remove([viewerSessionKey(sessionId), activeViewerKey(tab.id)]);
    } catch { /* preserve the original startup error */ }
    if (viewerWindow?.id != null) {
      try { await chrome.windows.remove(viewerWindow.id); } catch { /* no-op */ }
    }
    return {
      state: 'error',
      captureMode: 'viewer',
      error: { code: 'VIEWER_START_FAILED', message: error?.message || String(error) },
    };
  }
}
