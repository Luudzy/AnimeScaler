import test from 'node:test';
import assert from 'node:assert/strict';
import { mapCssRectToCapturePixels } from '../src/core/capture-geometry.js';

test('maps a player CSS rect into tab-capture video pixels', () => {
  assert.deepEqual(mapCssRectToCapturePixels({
    rect: { left: 100, top: 50, width: 800, height: 450 },
    viewport: { width: 1000, height: 600 },
    video: { width: 2000, height: 1200 },
  }), { x: 200, y: 100, width: 1600, height: 900 });
});

test('clips partially offscreen player geometry to the captured viewport', () => {
  assert.deepEqual(mapCssRectToCapturePixels({
    rect: { left: -50, top: 100, width: 500, height: 400 },
    viewport: { width: 1000, height: 800 },
    video: { width: 1500, height: 1200 },
  }), { x: 0, y: 150, width: 675, height: 600 });
});

test('rejects empty or fully offscreen capture geometry', () => {
  assert.equal(mapCssRectToCapturePixels({
    rect: { left: 1200, top: 0, width: 300, height: 300 },
    viewport: { width: 1000, height: 800 },
    video: { width: 1920, height: 1080 },
  }), null);
});
