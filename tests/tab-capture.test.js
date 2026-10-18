import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildTabCaptureConstraints,
  chooseCaptureHostFrame,
  shouldFallbackToTabCapture,
  normalizeFrameUrl,
} from '../src/core/tab-capture.js';

test('falls back to tab capture only for cross-origin frame access errors', () => {
  assert.equal(shouldFallbackToTabCapture({ state: 'error', error: { code: 'CROSS_ORIGIN_VIDEO' } }), true);
  assert.equal(shouldFallbackToTabCapture({ state: 'error', error: { code: 'MODEL_INIT_FAILED' } }), false);
  assert.equal(shouldFallbackToTabCapture({ state: 'running' }), false);
});

test('builds a video-only tab capture constraint so original page audio keeps playing', () => {
  assert.deepEqual(buildTabCaptureConstraints('stream-123'), {
    audio: false,
    video: {
      mandatory: {
        chromeMediaSource: 'tab',
        chromeMediaSourceId: 'stream-123',
      },
    },
  });
});

test('chooses the outermost child frame when the video is nested across origins', () => {
  const frames = [
    { frameId: 0, parentFrameId: -1, url: 'https://animefire.app/watch' },
    { frameId: 3, parentFrameId: 0, url: 'https://player.example/embed/abc' },
    { frameId: 7, parentFrameId: 3, url: 'https://cdn.example/player' },
  ];
  assert.equal(chooseCaptureHostFrame(frames, 7)?.frameId, 3);
});

test('top-frame video keeps frame zero as the capture host', () => {
  const frames = [{ frameId: 0, parentFrameId: -1, url: 'https://example.test/' }];
  assert.equal(chooseCaptureHostFrame(frames, 0)?.frameId, 0);
});

test('normalizes frame URLs for iframe matching', () => {
  assert.equal(normalizeFrameUrl('https://player.example/embed/1#hash'), 'https://player.example/embed/1');
  assert.equal(normalizeFrameUrl('https://player.example/embed/1/'), 'https://player.example/embed/1');
});

test('builds the top-frame tab capture message from the selected nested video frame', async () => {
  const { buildTabCaptureStartMessage } = await import('../src/core/tab-capture.js');
  const frames = [
    { frameId: 0, parentFrameId: -1, url: 'https://animefire.app/video/ep' },
    { frameId: 2, parentFrameId: 0, url: 'https://player.example/embed/ep' },
    { frameId: 9, parentFrameId: 2, url: 'https://cdn.example/video' },
  ];
  const settings = { enabled: true, preset: 'performance' };
  const message = buildTabCaptureStartMessage({ streamId: 'stream-x', settings, frames, targetFrameId: 9 });

  assert.equal(message.type, 'ANIMESCALER_START_TAB_CAPTURE');
  assert.equal(message.streamId, 'stream-x');
  assert.equal(message.captureFrame.frameId, 2);
  assert.equal(message.captureFrame.targetFrameId, 9);
  assert.equal(message.captureFrame.url, 'https://player.example/embed/ep');
  assert.equal(message.settings, settings);
});

test('falls back to the external manual viewer when inline tab capture cannot restrict the track', async () => {
  const { shouldFallbackToManualViewer } = await import('../src/core/tab-capture.js');
  assert.equal(shouldFallbackToManualViewer({ state: 'error', error: { code: 'TAB_CAPTURE_NEEDS_VIEWER' } }), true);
  assert.equal(shouldFallbackToManualViewer({ state: 'error', error: { code: 'MODEL_INIT_FAILED' } }), false);
  assert.equal(shouldFallbackToManualViewer({ state: 'running' }), false);
});
