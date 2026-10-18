import test from 'node:test';
import assert from 'node:assert/strict';
import { AdaptiveQualityController } from '../src/core/adaptive-quality.js';

test('adaptive controller steps down after sustained overload', () => {
  const controller = new AdaptiveQualityController({ maxLevel: 3, overloadSamples: 4, recoverySamples: 50, cooldownSamples: 0, alpha: 1 });
  let result;
  for (let i = 0; i < 4; i += 1) result = controller.record(45, 41.7);
  assert.equal(result.changed, true);
  assert.equal(result.level, 1);
});

test('adaptive controller does not react to a single slow frame', () => {
  const controller = new AdaptiveQualityController({ maxLevel: 3, overloadSamples: 4, cooldownSamples: 0, alpha: 1 });
  const slow = controller.record(60, 41.7);
  const normal = controller.record(20, 41.7);
  assert.equal(slow.changed, false);
  assert.equal(normal.level, 0);
});

test('adaptive controller steps up only after sustained headroom', () => {
  const controller = new AdaptiveQualityController({ initialLevel: 2, maxLevel: 3, overloadSamples: 4, recoverySamples: 5, cooldownSamples: 0, alpha: 1 });
  let result;
  for (let i = 0; i < 5; i += 1) result = controller.record(15, 41.7);
  assert.equal(result.changed, true);
  assert.equal(result.level, 1);
});

test('multi-second frames reduce quality immediately even during cooldown', () => {
  const controller = new AdaptiveQualityController({ maxLevel: 3 });
  assert.equal(controller.record(5857, 41.7).level, 1);
  assert.equal(controller.record(2000, 41.7).level, 2);
  assert.equal(controller.record(1000, 41.7).level, 3);
  assert.equal(controller.record(1000, 41.7).changed, false);
});

test('default adaptation reaches the smallest resolution from native', () => {
  const controller = new AdaptiveQualityController();
  for (let i = 0; i < 7; i++) controller.record(1000, 41.7);
  assert.equal(controller.level, 7);
});
