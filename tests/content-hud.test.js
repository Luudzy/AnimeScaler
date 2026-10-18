import test from 'node:test';
import assert from 'node:assert/strict';
import { AnimeScalerHud } from '../src/content/hud.js';

test('HUD keeps processed FPS value and explains reused frames are counted', t => {
  const previousDocument = globalThis.document;
  const root = { dataset: {}, style: {}, textContent: '', remove() {} };
  globalThis.document = {
    createElement: () => root,
    documentElement: { appendChild() {} },
  };
  t.after(() => {
    if (previousDocument === undefined) delete globalThis.document;
    else globalThis.document = previousDocument;
  });

  const hud = new AnimeScalerHud();
  hud.update({
    state: 'running',
    processedFps: 18.6,
    stats: {
      backend: 'rocm', modelLabel: 'Lite V3', modelPrecision: 'fp16',
      sourceWidth: 640, sourceHeight: 360, outputWidth: 1280, outputHeight: 720,
      totalMs: 42.3,
    },
  });

  const label = root.textContent.normalize('NFD').replace(/\p{Diacritic}/gu, '').toLowerCase();
  assert.match(label, /18\.6 fps/);
  assert.match(label, /fps.{0,100}(?:inclui|conta|considera|contabiliza).{0,60}quadros? (?:reutilizados?|reaproveitados?)|quadros? (?:reutilizados?|reaproveitados?).{0,60}(?:sao )?(?:incluidos?|contados?|considerados?|contabilizados?).{0,60}fps/);
});
