import { normalizeFrameUrl } from '../core/tab-capture.js';
import { containedVideoRect } from '../core/capture-geometry.js';
import { findBestVideo, queryOpenRoots, isElementVisible } from './video-finder.js';

function isVisibleRect(rect, viewport) {
  const width = Number(viewport?.width) || globalThis.innerWidth || 0;
  const height = Number(viewport?.height) || globalThis.innerHeight || 0;
  return rect.width >= 64
    && rect.height >= 64
    && rect.bottom > 0
    && rect.right > 0
    && rect.top < height
    && rect.left < width;
}

export function scoreFrameElement(element, { url = '', viewport = null } = {}) {
  if (!element?.isConnected || typeof element.getBoundingClientRect !== 'function') return -1;
  if (!isElementVisible(element)) return -1;
  const rect = element.getBoundingClientRect();
  if (!isVisibleRect(rect, viewport)) return -1;

  const area = rect.width * rect.height;
  const targetUrl = normalizeFrameUrl(url);
  const candidateUrl = normalizeFrameUrl(element.src || element.getAttribute?.('src') || '');
  if (!targetUrl || targetUrl === 'about:blank') return area;
  if (candidateUrl === targetUrl) return area + 1_000_000_000;

  try {
    const target = new URL(targetUrl);
    const candidate = new URL(candidateUrl);
    if (target.origin === candidate.origin && target.pathname === candidate.pathname) return area + 500_000_000;
    if (target.origin === candidate.origin) return area + 100_000_000;
  } catch { /* best-effort URL matching */ }
  return area;
}

export function findCaptureTargetElement(root = document, frameInfo = {}) {
  if (frameInfo.frameId === 0) return findBestVideo(root);
  const viewport = frameInfo.viewport ?? {
    width: globalThis.innerWidth ?? 0,
    height: globalThis.innerHeight ?? 0,
  };
  let best = null;
  let bestScore = -1;
  for (const element of queryOpenRoots(root, 'iframe, frame')) {
    const score = scoreFrameElement(element, { url: frameInfo.url, viewport });
    if (score > bestScore) {
      best = element;
      bestScore = score;
    }
  }
  return best;
}


export function describeCaptureTarget(root = document, frameInfo = {}, targetElement = null) {
  const viewport = frameInfo.viewport ?? {
    width: globalThis.innerWidth ?? 0,
    height: globalThis.innerHeight ?? 0,
  };
  const element = targetElement ?? findCaptureTargetElement(root, { ...frameInfo, viewport });
  if (!element?.isConnected || typeof element.getBoundingClientRect !== 'function' || !isElementVisible(element)) return null;
  let rect = element.getBoundingClientRect();
  if (!isVisibleRect(rect, viewport)) return null;
  const style = globalThis.getComputedStyle?.(element);
  // Work in the content box: iframe borders and video padding are not pixels
  // from the source video. Account for CSS scaling of the element as well.
  if (style && element.offsetWidth > 0 && element.offsetHeight > 0) {
    const sx = rect.width / element.offsetWidth, sy = rect.height / element.offsetHeight;
    const px = name => parseFloat(style[name]) || 0;
    const left = px('borderLeftWidth') + px('paddingLeft');
    const top = px('borderTopWidth') + px('paddingTop');
    rect = {left:rect.left+left*sx,top:rect.top+top*sy,
      width:rect.width-(left+px('borderRightWidth')+px('paddingRight'))*sx,
      height:rect.height-(top+px('borderBottomWidth')+px('paddingBottom'))*sy};
  }
  if (element.videoWidth > 0 && style?.objectFit === 'contain') {
    rect = containedVideoRect(rect,element.videoWidth,element.videoHeight,style.objectPosition);
  }
  return {
    rect: {
      left: rect.left,
      top: rect.top,
      width: rect.width,
      height: rect.height,
    },
    viewport: {
      width: Number(viewport.width) || 0,
      height: Number(viewport.height) || 0,
    },
  };
}
