import test from 'node:test';
import assert from 'node:assert/strict';
import { AnimeScalerController } from '../src/content/controller.js';

class MutationObserverStub {
  observe() {}
  disconnect() {}
}

test('start cleans resources left by an error before retrying', async () => {
  const oldMutationObserver = globalThis.MutationObserver;
  const oldWindow = globalThis.window;
  const oldDocument = globalThis.document;

  globalThis.MutationObserver = MutationObserverStub;
  globalThis.window = {
    addEventListener() {},
    removeEventListener() {},
  };
  globalThis.document = {
    querySelectorAll() { return []; },
    documentElement: {},
  };

  try {
    const controller = new AnimeScalerController();
    let overlayDestroyed = 0;
    let hudDestroyed = 0;
    let engineDestroyed = 0;
    controller.state = 'error';
    controller.overlay = { destroy() { overlayDestroyed += 1; }, hide() {} };
    controller.hud = { destroy() { hudDestroyed += 1; }, update() {} };
    controller.engine = { async destroy() { engineDestroyed += 1; } };

    const status = await controller.start();

    assert.equal(overlayDestroyed, 1);
    assert.equal(hudDestroyed, 1);
    assert.equal(engineDestroyed, 1);
    assert.equal(status.state, 'error');
    assert.equal(status.error.code, 'NO_VIDEO');
  } finally {
    globalThis.MutationObserver = oldMutationObserver;
    globalThis.window = oldWindow;
    globalThis.document = oldDocument;
  }
});
