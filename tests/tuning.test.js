import test from 'node:test';
import assert from 'node:assert/strict';
import { compareSamples, chooseCandidate } from '../diagnostics/tuning-math.js';
test('tuning rejects invalid, nonfinite and inaccurate outputs even if faster', () => {
  assert.equal(compareSamples([1], [NaN]).passed, false);
  assert.equal(compareSamples([1], [.8]).passed, false);
  assert.equal(compareSamples([1], []).passed, false);
  const valid = {p95Ms: 30, correctness: compareSamples([1, .2], [.999, .201])};
  assert.equal(chooseCandidate([{p95Ms: 10, correctness: {passed: false}}, valid]), valid);
  assert.equal(chooseCandidate([]), null);
});
