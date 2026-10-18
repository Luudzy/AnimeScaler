import test from 'node:test';
import assert from 'node:assert/strict';
import { scoreVideoCandidate } from '../src/content/video-finder.js';

test('scoreVideoCandidate prefers a large decoded visible video', () => {
  const small = scoreVideoCandidate({ width: 320, height: 180, videoWidth: 1280, videoHeight: 720, visible: true, opacity: 1 });
  const large = scoreVideoCandidate({ width: 1280, height: 720, videoWidth: 1920, videoHeight: 1080, visible: true, opacity: 1 });
  assert.ok(large > small);
});

test('scoreVideoCandidate rejects hidden and tiny videos but keeps visible videos before metadata is decoded', () => {
  assert.equal(scoreVideoCandidate({ width: 1280, height: 720, videoWidth: 1920, videoHeight: 1080, visible: false, opacity: 1 }), -1);
  assert.ok(scoreVideoCandidate({ width: 1280, height: 720, videoWidth: 0, videoHeight: 0, visible: true, opacity: 1 }) > 0);
  assert.equal(scoreVideoCandidate({ width: 20, height: 20, videoWidth: 1920, videoHeight: 1080, visible: true, opacity: 1 }), -1);
});

test('scoreVideoCandidate penalizes nearly transparent videos', () => {
  const opaque = scoreVideoCandidate({ width: 800, height: 450, videoWidth: 1280, videoHeight: 720, visible: true, opacity: 1 });
  const transparent = scoreVideoCandidate({ width: 800, height: 450, videoWidth: 1280, videoHeight: 720, visible: true, opacity: 0.05 });
  assert.ok(opaque > transparent);
});
