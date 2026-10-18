import test from 'node:test';
import assert from 'node:assert/strict';
import {temporalStrength} from '../src/engine/temporal-budget.js';
test('old AI fades before expiry; slow inference never removes the hard ceiling',()=>{
  assert.equal(temporalStrength(10,20),1);
  assert.ok(temporalStrength(260,20)>0 && temporalStrength(260,20)<1);
  assert.equal(temporalStrength(300,20),0);
  assert.equal(temporalStrength(301,800),0);
  assert.equal(temporalStrength(NaN),0);
  assert.equal(temporalStrength(-1),0);
});

test('regular 12/24/30 FPS AI arrivals do not pulse the blend weight',()=>{
  for(const latency of [20,40,70]) {
    const values=[0,16,33,50,66,83].map(offset=>temporalStrength(latency+offset,latency));
    assert.ok(Math.max(...values)-Math.min(...values)<.01);
    assert.equal(values[0],1);
  }
});
