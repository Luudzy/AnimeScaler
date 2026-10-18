import test from 'node:test';
import assert from 'node:assert/strict';
import { FramePacer } from '../src/core/frame-pacer.js';

test('pacer does not accumulate lateness when 60 Hz video is capped at 24 FPS', () => {
  const pacer = new FramePacer();
  let accepted = 0;
  for (let i=0;i<600;i++) if (pacer.take(i*1000/60,24)) accepted++;
  assert.equal(accepted,240);
});
test('pacer drops stale deadlines after a pause without a catch-up burst', () => {
  const pacer = new FramePacer();
  assert.equal(pacer.take(0,30),true);
  assert.equal(pacer.take(1000,30),true);
  assert.equal(pacer.take(1001,30),false);
  assert.equal(pacer.take(1033.3,30),true);
});
test('pacer resets cadence when the frame limit changes', () => {
  const pacer = new FramePacer();
  pacer.take(0,12);
  assert.equal(pacer.take(20,60),true);
  assert.equal(pacer.take(21,60),false);
  pacer.reset();
  assert.equal(pacer.take(21,60),true);
});
