import test from 'node:test';
import assert from 'node:assert/strict';
import { analyzeMotion } from '../diagnostics/motion-analysis.js';

test('offline analysis separates pixel validation from erosion and excludes unsupported error',()=>{
  const reference=new Uint8ClampedArray(3*3*4).fill(100), current=reference.slice();
  current[4*4]=130;
  const mask=new Uint8ClampedArray(reference.length);
  const result=analyzeMotion(reference,current,3,3,{tiles:[{x:0,y:0,width:3,height:3,dx:0,dy:0}],mask});
  assert.equal(result.beforeErosionCoverage,8/9);
  assert.equal(result.coverage,0);
  assert.equal(result.erosionLoss,8/9);
  assert.equal(result.supportedMaxChannelErrorMean,null);
  assert.equal(result.supportedPixels,0);
});

test('offline residual uses the selected fractional warp and opacity weighting',()=>{
  const reference=Uint8ClampedArray.from([10,20,30,255,30,40,50,255]);
  const current=Uint8ClampedArray.from([20,30,40,255,0,0,0,255]);
  const mask=new Uint8ClampedArray(8);mask[3]=128;
  const result=analyzeMotion(reference,current,2,1,{tiles:[{x:0,y:0,width:1,height:1,dx:.5,dy:0}],mask});
  assert.equal(result.supportedPixels,1);
  assert.equal(result.supportedMaxChannelErrorMean,0);
  assert.equal(result.supportedMaxChannelErrorMax,0);
  assert.equal(result.beforeErosionCoverage,.5);
  assert.equal(result.coverage,128/510);
});
