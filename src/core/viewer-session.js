export function viewerSessionKey(sessionId) {
  return `animeScalerViewer:${sessionId}`;
}

export function activeViewerKey(sourceTabId) {
  return `animeScalerViewerActive:${sourceTabId}`;
}

export function buildViewerSession({
  streamId,
  sourceTabId,
  sourceWindowId,
  viewerTabId,
  viewerWindowId,
  captureFrame,
  settings,
}) {
  return {
    streamId,
    sourceTabId,
    sourceWindowId,
    viewerTabId,
    viewerWindowId,
    captureFrame,
    targetFrameId: captureFrame?.targetFrameId ?? captureFrame?.frameId ?? 0,
    settings,
  };
}

export function shouldForwardSettingsToSource(status, frameId) {
  return status?.state === 'running'
    && status?.captureMode !== 'viewer'
    && frameId != null;
}
