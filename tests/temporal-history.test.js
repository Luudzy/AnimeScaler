import test from 'node:test';
import assert from 'node:assert/strict';
import {isSceneCut,ConfidenceHistory} from '../src/engine/temporal-history.js';

test('large scene changes reset history; brightness noise does not',()=>{
  const image=v=>new Uint8ClampedArray(80*45*4).fill(v);
  assert.equal(isSceneCut(image(20),image(230)),true);
  assert.equal(isSceneCut(image(100),image(110)),false);
  assert.equal(isSceneCut(null,image(20)),false);
});

test('rejected areas restore gradually without preserving invalid pixels',()=>{
  const history=new ConfidenceHistory();
  const reuse=a=>({tiles:[],mask:new Uint8ClampedArray([255,255,255,a])});
  assert.equal(history.filter(reuse(255),0).mask[3],255);
  assert.equal(history.filter(reuse(0),33).mask[3],0);
  const restored=history.filter(reuse(255),66).mask[3];
  assert.equal(restored,0);
  assert.equal(history.filter(reuse(0),99).mask[3],0);
  history.clear(100);
  assert.ok(history.filter(reuse(255),133).mask[3]<100);
  history.reset();assert.equal(history.filter(reuse(255),166).mask[3],255);
});

test('alternating valid and invalid matches cannot pulse AI back on every other frame',()=>{
  const history=new ConfidenceHistory();
  const reuse=a=>({tiles:[],mask:new Uint8ClampedArray([255,255,255,a])});
  history.filter(reuse(255),0);
  for(let frame=1;frame<=20;frame++) {
    const alpha=history.filter(reuse(frame%2?0:255),frame*33).mask[3];
    assert.equal(alpha,0,`AI flashed back on frame ${frame}`);
  }
  let recovered;
  for(let frame=21;frame<=40;frame++)recovered=history.filter(reuse(255),frame*33).mask[3];
  assert.equal(recovered,255,'stable detail must eventually recover');
});
