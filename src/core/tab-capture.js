export function shouldFallbackToTabCapture(status) {
  return status?.state === 'error' && status?.error?.code === 'CROSS_ORIGIN_VIDEO';
}

export function shouldFallbackToManualViewer(status) {
  return status?.state === 'error' && status?.error?.code === 'TAB_CAPTURE_NEEDS_VIEWER';
}

export function buildTabCaptureConstraints(streamId) {
  if (!streamId) throw new Error('Tab capture stream id is required');
  return {
    audio: false,
    video: {
      mandatory: {
        chromeMediaSource: 'tab',
        chromeMediaSourceId: streamId,
      },
    },
  };
}

export function chooseCaptureHostFrame(frames = [], targetFrameId = 0) {
  const byId = new Map(frames.map((frame) => [frame.frameId, frame]));
  let current = byId.get(targetFrameId);
  if (!current) return null;
  if (current.frameId === 0) return current;

  let guard = frames.length + 1;
  while (current && current.parentFrameId !== 0 && current.parentFrameId !== -1 && guard-- > 0) {
    const parent = byId.get(current.parentFrameId);
    if (!parent) break;
    current = parent;
  }
  return current;
}

export function normalizeFrameUrl(value) {
  if (!value) return '';
  try {
    const url = new URL(value);
    url.hash = '';
    if (url.pathname.length > 1) url.pathname = url.pathname.replace(/\/+$/, '');
    return url.href.replace(/\/$/, '');
  } catch {
    return String(value).replace(/#.*$/, '').replace(/\/+$/, '');
  }
}

export function buildTabCaptureStartMessage({ streamId, settings, frames = [], targetFrameId }) {
  const host = chooseCaptureHostFrame(frames, targetFrameId);
  if (!host) throw new Error('Unable to resolve capture host frame');
  return {
    type: 'ANIMESCALER_START_TAB_CAPTURE',
    streamId,
    settings,
    captureFrame: {
      frameId: host.frameId,
      parentFrameId: host.parentFrameId,
      url: host.url,
      targetFrameId,
    },
  };
}
