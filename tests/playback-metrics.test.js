import test from 'node:test';
import assert from 'node:assert/strict';
import {analyzePresentation, summarize} from '../diagnostics/playback-metrics.js';

const row=(motion,confidence,weight,age=40)=>({motionCoverage:motion,confidenceCoverage:confidence,
  temporalWeight:weight,aiCoverage:confidence*weight,historyAgeMs:age});
const near=(actual,expected)=>assert.ok(Math.abs(actual-expected)<1e-9,`${actual} != ${expected}`);

test('coverage losses are attributed per frame, rather than multiplying component means',()=>{
  const result=analyzePresentation([row(1,.25,1),row(1,1,.25)]);
  assert.equal(result.attributedSamples,2);
  near(result.lossMean.motion,0);near(result.lossMean.confidence,.375);
  near(result.lossMean.age,.375);near(result.lossMean.ai,.25);
  near(Object.values(result.lossMean).reduce((a,b)=>a+b,0),1);
  assert.notEqual(result.components.confidenceCoverage.mean*result.components.temporalWeight.mean,result.lossMean.ai);
});

test('unusable history is not falsely attributed to motion; paused history preserves real age',()=>{
  const result=analyzePresentation([row(0,0,0,350),row(1,1,1,2000)]);
  near(result.lossMean.unusableHistory,.5);near(result.lossMean.motion,0);
  near(result.lossMean.ai,.5);
  assert.equal(result.ageBuckets.at(-1).samples,2);
});

test('missing and inconsistent components remain explicit and use their own sample counts',()=>{
  const result=analyzePresentation([
    {aiCoverage:.5,historyAgeMs:20},row(1,.5,1),row(.2,.5,1),{...row(1,1,1),temporalWeight:NaN},
  ]);
  assert.equal(result.samples,4);assert.equal(result.attributedSamples,1);
  assert.equal(result.missingComponents,2);assert.equal(result.inconsistentComponents,1);
  assert.equal(result.components.motionCoverage.count,3);
  assert.equal(result.components.temporalWeight.count,2);
  assert.equal(analyzePresentation([]).lossMean,null);
});

test('source-age buckets are mutually exclusive at their boundaries',()=>{
  const result=analyzePresentation([0,39.9,40,79.9,80,149.9,150,299.9,300].map(age=>row(1,1,1,age)));
  assert.deepEqual(result.ageBuckets.map(bucket=>bucket.samples),[2,2,2,2,1]);
});

test('stage summaries distinguish missing values from measured zero',()=>{
  assert.deepEqual(summarize([]),{count:0,mean:null,p50:null,p95:null,max:null});
  assert.deepEqual(summarize([0,2,4]),{count:3,mean:2,p50:2,p95:4,max:4});
});
