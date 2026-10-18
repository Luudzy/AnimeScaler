import test from 'node:test';
import assert from 'node:assert/strict';
import { AnimeScalerController } from '../src/content/controller.js';

class MutationObserverStub {
  observe() {}
  disconnect() {}
}

test('controller exposes tab capture mode and fails clearly when the capture target is missing', async () => {
  const oldMutationObserver = globalThis.MutationObserver;
  const oldWindow = globalThis.window;
  const oldDocument = globalThis.document;
  const oldInnerWidth = globalThis.innerWidth;
  const oldInnerHeight = globalThis.innerHeight;

  globalThis.MutationObserver = MutationObserverStub;
  globalThis.window = { addEventListener() {}, removeEventListener() {}, top: null, self: null };
  globalThis.window.top = globalThis.window;
  globalThis.window.self = globalThis.window;
  globalThis.innerWidth = 1920;
  globalThis.innerHeight = 1080;
  globalThis.document = {
    querySelectorAll() { return []; },
    documentElement: {},
  };

  try {
    const controller = new AnimeScalerController();
    const status = await controller.startTabCapture({
      streamId: 'stream-id',
      captureFrame: { frameId: 4, url: 'https://player.example/embed' },
      settings: { enabled: true },
    });

    assert.equal(status.state, 'error');
    assert.equal(status.captureMode, 'tab');
    assert.equal(status.error.code, 'CAPTURE_TARGET_NOT_FOUND');
  } finally {
    globalThis.MutationObserver = oldMutationObserver;
    globalThis.window = oldWindow;
    globalThis.document = oldDocument;
    globalThis.innerWidth = oldInnerWidth;
    globalThis.innerHeight = oldInnerHeight;
  }
});
