import test from 'node:test';
import assert from 'node:assert/strict';
import { SustainedFpsGate } from '../src/core/sustained-fps-gate.js';

test('warmup frames do not count toward the sustained window', () => {
  const gate = new SustainedFpsGate({ targetFps: 24, windowMs: 5000 });
  assert.equal(gate.record(0, { warmup: true }).status, 'warming');
  assert.equal(gate.record(5000).status, 'warming');
});

test('gate remains warming until valid frames span the full window', () => {
  const gate = new SustainedFpsGate({ targetFps: 24, windowMs: 5000 });
  gate.record(1000);
  assert.deepEqual(gate.record(5999), { status: 'warming', fps: null });
});

test('gate detects sustained throughput below target', () => {
  const gate = new SustainedFpsGate({ targetFps: 24, windowMs: 5000 });
  for (let time = 0; time <= 5000; time += 200) gate.record(time);
  const result = gate.record(5200);
  assert.equal(result.status, 'below-target');
  assert.ok(result.fps < 24);
});

test('gate accepts throughput at or above target', () => {
  for (const intervalMs of [5000 / 120, 40]) {
    const gate = new SustainedFpsGate({ targetFps: 24, windowMs: 5000 });
    let result;
    for (let index = 0; index <= Math.round(5000 / intervalMs); index++) result = gate.record(index * intervalMs);
    assert.equal(result.status, 'within-target');
  }
});

test('expired samples leave only the current five-second window', () => {
  const gate = new SustainedFpsGate({ targetFps: 24, windowMs: 5000 });
  for (let time = 0; time <= 5000; time += 1000 / 24) gate.record(time);
  let result;
  for (let time = 6000; time <= 10000; time += 1000) result = gate.record(time);
  assert.equal(result.status, 'below-target');
  assert.ok(result.fps <= 1);
});
