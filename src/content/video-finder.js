export function scoreVideoCandidate({ width, height, videoWidth, videoHeight, visible = true, opacity = 1, playing = false }) {
  if (!visible || !Number.isFinite(width) || !Number.isFinite(height) || width < 64 || height < 64) return -1;
  const safeOpacity = Number.isFinite(opacity) ? Math.max(0, Math.min(1, opacity)) : 1;
  const decoded = Number.isFinite(videoWidth) && Number.isFinite(videoHeight) && videoWidth > 0 && videoHeight > 0;
  const readinessFactor = decoded ? 1 : 0.9;
  return width * height * Math.max(0.05, safeOpacity) * readinessFactor * (playing ? 1.25 : 1);
}

// Each cross-origin frame runs its own content script. Only traverse DOM roots
// accessible in this frame; never try to read another frame's document.
export function* queryOpenRoots(root, selector) {
  yield* root.querySelectorAll(selector);
  for (const element of root.querySelectorAll('*')) {
    if (element.shadowRoot) yield* queryOpenRoots(element.shadowRoot, selector);
  }
}

export function isElementVisible(element) {
  for (let node = element; node; node = node.parentElement || node.getRootNode?.().host) {
    const style = globalThis.getComputedStyle?.(node);
    if (style?.display === 'none' || style?.visibility === 'hidden'
      || style?.visibility === 'collapse' || Number.parseFloat(style?.opacity) === 0) return false;
  }
  return true;
}

function elementCandidate(video) {
  const rect = video.getBoundingClientRect();
  const style = getComputedStyle(video);
  const visible = isElementVisible(video) && style.display !== 'none'
    && style.visibility !== 'hidden'
    && rect.bottom > 0
    && rect.right > 0
    && rect.top < window.innerHeight
    && rect.left < window.innerWidth;

  return {
    width: rect.width,
    height: rect.height,
    videoWidth: video.videoWidth,
    videoHeight: video.videoHeight,
    visible,
    opacity: Number.parseFloat(style.opacity || '1'),
    playing: video.paused === false && !video.ended && video.readyState >= 2,
  };
}

export function inspectBestVideo(root = document) {
  let best = null;
  let bestScore = -1;
  for (const video of queryOpenRoots(root, 'video')) {
    const candidate = elementCandidate(video);
    const score = scoreVideoCandidate(candidate);
    if (score > bestScore) {
      best = video;
      bestScore = score;
    }
  }
  return { video: best, score: bestScore };
}

export function findBestVideo(root = document) {
  return inspectBestVideo(root).video;
}
