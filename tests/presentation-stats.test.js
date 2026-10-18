import test from 'node:test';
import assert from 'node:assert/strict';
import { PresentationStats } from '../src/core/presentation-stats.js';

test('presentation statistics aggregate delivered AI separately from inference', () => {
  const stats = new PresentationStats({ startedAt: 0 });
  stats.record({ aiCoverage: 0, historyAgeMs: 50 }, 40);
  stats.record({ aiCoverage: .5, historyAgeMs: 100 }, 80);
  stats.record({ aiCoverage: 1, historyAgeMs: 250 }, 120);
  const result = stats.flush(1000);
  assert.equal(result.presentedFrames, 3);
  assert.equal(result.presentationFps, 3);
  assert.equal(result.aiCoverageMean, .5);
  assert.equal(result.historyAgeP95Ms, 250);
  assert.equal(result.lowConfidenceRatio, 1 / 3);
  assert.equal(stats.flush(2000), null);
});

test('invalid samples are rejected and idle time is excluded from the next window', () => {
  const stats = new PresentationStats({ startedAt: 0 });
  stats.record({ aiCoverage: NaN, historyAgeMs: 10 }, 20);
  stats.record({ aiCoverage: 2, historyAgeMs: 10 }, 30);
  assert.equal(stats.flush(1000), null);
  stats.record({ aiCoverage: 1, historyAgeMs: 12 }, 60000);
  assert.equal(stats.flush(60100).presentationFps, 10);
});

test('presentation windows remain bounded during a stalled diagnostic consumer', () => {
  const stats = new PresentationStats({ startedAt: 0, maxSamples: 4 });
  for (let n = 0; n < 20; n++) stats.record({ aiCoverage: n % 2, historyAgeMs: n }, n * 40);
  const result = stats.flush(1000);
  assert.equal(result.presentedFrames, 20);
  assert.equal(result.aiCoverageMean, .5);
  assert.equal(result.historyAgeP95Ms, 19);
  assert.ok(stats.ages.length <= 4);
});

test('coverage components average only samples that actually contain them',()=>{
  const stats=new PresentationStats({startedAt:0});
  stats.record({aiCoverage:.2,historyAgeMs:40},40);
  stats.record({aiCoverage:.4,historyAgeMs:80,motionCoverage:.8,confidenceCoverage:.5,temporalWeight:.8},80);
  const result=stats.flush(100);
  assert.equal(result.motionCoverageMean,.8);
  assert.equal(result.confidenceCoverageMean,.5);
  assert.equal(result.temporalWeightMean,.8);
  assert.ok(Math.abs(result.aiCoverageMean-.3)<1e-12);
  stats.record({aiCoverage:1,historyAgeMs:10},120);
  assert.equal('motionCoverageMean' in stats.flush(150),false);
});
