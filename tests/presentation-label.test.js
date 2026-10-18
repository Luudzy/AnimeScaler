import test from 'node:test';
import assert from 'node:assert/strict';
import { formatPresentationMetrics } from '../src/core/presentation-label.js';

test('hybrid label separates delivered contribution from processing speed',()=>{
  assert.equal(formatPresentationMetrics({aiCoverage:.124,historyAgeMs:62.4}), 'Contribuição IA 12% · atraso 62 ms');
  assert.equal(formatPresentationMetrics({aiCoverage:0,historyAgeMs:0}), 'Contribuição IA 0% · atraso 0 ms');
});

test('unavailable or invalid coverage never implies full AI contribution',()=>{
  for(const value of [null,{}, {aiCoverage:NaN,historyAgeMs:20},{aiCoverage:1.1,historyAgeMs:20},{aiCoverage:.5,historyAgeMs:-1}]) {
    assert.equal(formatPresentationMetrics(value),'');
  }
});
