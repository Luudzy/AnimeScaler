import test from 'node:test';
import assert from 'node:assert/strict';
import { chooseFrameForStart, chooseRunningFrame } from '../src/core/frame-routing.js';

test('chooseFrameForStart selects the frame with the highest visible video score', () => {
  const probes = [
    { frameId: 0, ok: true, hasVideo: false, videoScore: -1, state: 'stopped' },
    { frameId: 3, ok: true, hasVideo: true, videoScore: 180000, state: 'stopped' },
    { frameId: 7, ok: true, hasVideo: true, videoScore: 920000, state: 'stopped' },
  ];
  assert.equal(chooseFrameForStart(probes)?.frameId, 7);
});

test('chooseRunningFrame prefers an already running scaler even if another frame has a larger raw video', () => {
  const probes = [
    { frameId: 2, ok: true, hasVideo: true, videoScore: 900000, state: 'stopped' },
    { frameId: 5, ok: true, hasVideo: true, videoScore: 200000, state: 'running' },
  ];
  assert.equal(chooseRunningFrame(probes)?.frameId, 5);
});

test('chooseFrameForStart ignores unreachable frames', () => {
  const probes = [
    { frameId: 4, ok: false, hasVideo: true, videoScore: 999999, state: 'stopped' },
    { frameId: 6, ok: true, hasVideo: true, videoScore: 1000, state: 'stopped' },
  ];
  assert.equal(chooseFrameForStart(probes)?.frameId, 6);
});

test('chooseFrameForStatus keeps a tab-capture error visible even when the top frame has no native video', async () => {
  const { chooseFrameForStatus } = await import('../src/core/frame-routing.js');
  const probes = [
    { frameId: 0, ok: true, hasVideo: false, videoScore: -1, state: 'error', captureMode: 'tab', error: { code: 'TAB_CAPTURE_START_FAILED' } },
    { frameId: 4, ok: true, hasVideo: true, videoScore: 900000, state: 'stopped' },
  ];
  assert.equal(chooseFrameForStatus(probes)?.frameId, 0);
});
