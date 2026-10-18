function finite(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

export function containedVideoRect(rect, width, height, position = '50% 50%') {
  if (!(width > 0 && height > 0)) return {...rect};
  const scale = Math.min(rect.width / width, rect.height / height);
  const w = width * scale, h = height * scale;
  const parts = position.trim().split(/\s+/);
  const offset = (value, space) => {
    if (value === 'left' || value === 'top') return 0;
    if (value === 'right' || value === 'bottom') return space;
    if (value?.endsWith('%')) return space * parseFloat(value) / 100;
    if (value?.endsWith('px')) return parseFloat(value);
    return space / 2;
  };
  return {left:rect.left + offset(parts[0],rect.width-w),top:rect.top + offset(parts[1],rect.height-h),width:w,height:h};
}

// Each rectangle describes the child frame's content box, except the last,
// which describes the pixels occupied by the video inside its own viewport.
export function composeCaptureGeometry(chain) {
  if (!chain?.length || chain.some(g => !g?.rect || !(g.viewport?.width > 0 && g.viewport?.height > 0))) {
    throw new Error('CAPTURE_TARGET_NOT_FOUND: geometria do vídeo indisponível.');
  }
  let rect = {...chain.at(-1).rect};
  for (let i = chain.length - 1; i >= 0; i--) {
    const viewport = chain[i].viewport;
    if (!(rect.width > 0 && rect.height > 0) || rect.left < -1 || rect.top < -1 ||
        rect.left + rect.width > viewport.width + 1 || rect.top + rect.height > viewport.height + 1) {
      throw new Error('CAPTURE_TARGET_CLIPPED: vídeo parcialmente fora da captura. Reabra com Tela cheia exclusiva ativada.');
    }
    if (i > 0) {
      const parent = chain[i-1].rect;
      const sx = parent.width / viewport.width, sy = parent.height / viewport.height;
      rect = {left:parent.left + rect.left*sx,top:parent.top + rect.top*sy,width:rect.width*sx,height:rect.height*sy};
    }
  }
  return {rect,viewport:chain[0].viewport};
}

export function mapCssRectToCapturePixels({ rect, viewport, video }) {
  const viewportWidth = finite(viewport?.width);
  const viewportHeight = finite(viewport?.height);
  const videoWidth = finite(video?.width);
  const videoHeight = finite(video?.height);
  if (viewportWidth <= 0 || viewportHeight <= 0 || videoWidth <= 0 || videoHeight <= 0) return null;

  const left = finite(rect?.left);
  const top = finite(rect?.top);
  const right = left + finite(rect?.width);
  const bottom = top + finite(rect?.height);
  const clippedLeft = clamp(left, 0, viewportWidth);
  const clippedTop = clamp(top, 0, viewportHeight);
  const clippedRight = clamp(right, 0, viewportWidth);
  const clippedBottom = clamp(bottom, 0, viewportHeight);
  if (clippedRight <= clippedLeft || clippedBottom <= clippedTop) return null;

  const scaleX = videoWidth / viewportWidth;
  const scaleY = videoHeight / viewportHeight;
  const x = clamp(Math.round(clippedLeft * scaleX), 0, Math.max(0, videoWidth - 1));
  const y = clamp(Math.round(clippedTop * scaleY), 0, Math.max(0, videoHeight - 1));
  const endX = clamp(Math.round(clippedRight * scaleX), x + 1, videoWidth);
  const endY = clamp(Math.round(clippedBottom * scaleY), y + 1, videoHeight);
  return { x, y, width: endX - x, height: endY - y };
}
