import test from 'node:test';
import assert from 'node:assert/strict';
import { AnimeScalerController } from '../src/content/controller.js';

test('in-flight STALE_SESSION after explicit stop does not turn stopped playback into an error', async t => {
  const previous = Object.fromEntries(['window', 'innerWidth', 'innerHeight', 'getComputedStyle',
    'MutationObserver', 'document'].map(key => [key, globalThis[key]]));
  globalThis.window = { innerWidth: 1280, innerHeight: 720, removeEventListener() {} };
  globalThis.innerWidth = 1280;
  globalThis.innerHeight = 720;
  globalThis.getComputedStyle = element => element.style || {};
  globalThis.MutationObserver = class { disconnect() {} };
  globalThis.document = { querySelectorAll() { return []; }, documentElement: {} };
  t.after(() => { for (const [key, value] of Object.entries(previous)) globalThis[key] = value; });

  let rejectFrame;
  let loggedErrors = 0;
  const target = { isConnected: true, videoWidth: 640, videoHeight: 360, readyState: 4,
    paused: false, ended: false, currentTime: 1, style: {},
    getBoundingClientRect: () => ({ left: 0, top: 0, right: 640, bottom: 360, width: 640, height: 360 }),
    requestVideoFrameCallback: () => 1, cancelVideoFrameCallback() {} };
  const controller = new AnimeScalerController();
  controller.state = 'running';
  controller.video = controller.displayTarget = target;
  controller.settings = { ...controller.settings, maxFps: 24, adaptive: false };
  controller.diagnostics = { sourceFrame() {}, error() { loggedErrors++; } };
  controller.engine = {
    needsReconfigure: () => false,
    processFrame: () => new Promise((_, reject) => { rejectFrame = reject; }),
    destroy() {
      rejectFrame(Object.assign(new Error('STALE_SESSION'), { code: 'STALE_SESSION' }));
    },
  };

  const processing = controller.onVideoFrame(0, { mediaTime: 1 });
  await Promise.resolve();
  await controller.stop();
  await processing;

  assert.equal(controller.state, 'stopped');
  assert.equal(controller.error, null);
  assert.equal(loggedErrors, 0);
});
