import test from 'node:test';
import assert from 'node:assert/strict';
import { activeViewerKey, buildViewerSession, viewerSessionKey } from '../src/core/viewer-session.js';

test('viewer session keys are scoped to the generated session and source tab', () => {
  assert.equal(viewerSessionKey('abc'), 'animeScalerViewer:abc');
  assert.equal(activeViewerKey(42), 'animeScalerViewerActive:42');
});

test('buildViewerSession preserves the source frame needed for geometry and controls', () => {
  const settings = { enabled: true, preset: 'performance' };
  const captureFrame = { frameId: 3, targetFrameId: 9, url: 'https://player.test/embed' };
  assert.deepEqual(buildViewerSession({
    streamId: 'stream-x', sourceTabId: 12, sourceWindowId: 4, viewerTabId: 88,
    viewerWindowId: 7, captureFrame, settings,
  }), {
    streamId: 'stream-x', sourceTabId: 12, sourceWindowId: 4, viewerTabId: 88,
    viewerWindowId: 7, captureFrame, targetFrameId: 9, settings,
  });
});

test('viewer mode keeps setting changes in storage instead of restarting the source frame', async () => {
  const { shouldForwardSettingsToSource } = await import('../src/core/viewer-session.js');
  assert.equal(shouldForwardSettingsToSource({ state: 'running', captureMode: 'viewer' }, 9), false);
  assert.equal(shouldForwardSettingsToSource({ state: 'running', captureMode: 'direct' }, 9), true);
  assert.equal(shouldForwardSettingsToSource({ state: 'stopped', captureMode: 'direct' }, 9), false);
});
