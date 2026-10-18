import test from 'node:test';
import assert from 'node:assert/strict';
import { replayTemporal, cadenceEvents } from '../diagnostics/temporal-replay.js';

const frame=value=>new Uint8ClampedArray(80*45*4).fill(value);
test('policy replay preserves old capture age and does not draw on completion',()=>{
  const result=replayTemporal({frames:[frame(100),frame(102)],events:[
    {type:'present',atMs:0,frameIndex:0},{type:'capture',atMs:0,frameIndex:0,id:1},
    {type:'complete',atMs:40,id:1},{type:'present',atMs:300,frameIndex:1},
  ]});
  assert.equal(result.draws,2);assert.equal(result.uploads,1);
  assert.equal(result.rows.at(-1).metrics.historyAgeMs,300);
  assert.equal(result.rows.at(-1).metrics.aiCoverage,0);
});
test('policy replay rejects a pre-cut result and preserves paused validated history',()=>{
  const cut=replayTemporal({frames:[frame(20),frame(230)],events:[
    {type:'present',atMs:0,frameIndex:0},{type:'capture',atMs:0,frameIndex:0,id:1},
    {type:'present',atMs:40,frameIndex:1},{type:'complete',atMs:50,id:1},
  ]});
  assert.equal(cut.rows.at(-1).accepted,false);assert.equal(cut.rows.at(-1).reason,'scene');
  assert.equal(cut.uploads,0);
  const pause=replayTemporal({frames:[frame(100)],events:[
    {type:'present',atMs:0,frameIndex:0},{type:'capture',atMs:0,frameIndex:0,id:1},
    {type:'complete',atMs:40,id:1},{type:'present',atMs:2000,frameIndex:0},
  ]});
  assert.equal(pause.rows.at(-1).metrics.historyAgeMs,2000);
  assert.equal(pause.rows.at(-1).metrics.aiCoverage,1);
});
test('delayed moving-source result expires and resume cannot reuse old paused history',()=>{
  const expired=replayTemporal({frames:[frame(100),frame(102)],events:[
    {type:'present',atMs:0,frameIndex:0},{type:'capture',atMs:0,frameIndex:0,id:1},
    {type:'present',atMs:100,frameIndex:1},{type:'complete',atMs:350,id:1},
  ]});
  assert.equal(expired.rows.at(-1).accepted,false);assert.equal(expired.rows.at(-1).reason,'expired');
  assert.equal(expired.uploads,0);
  const resumed=replayTemporal({frames:[frame(100),frame(102)],events:[
    {type:'present',atMs:0,frameIndex:0},{type:'capture',atMs:0,frameIndex:0,id:1},
    {type:'complete',atMs:40,id:1},{type:'present',atMs:2000,frameIndex:0},
    {type:'present',atMs:2040,frameIndex:1},
  ]});
  assert.equal(resumed.rows.at(-2).metrics.aiCoverage,1);
  assert.equal(resumed.rows.at(-1).metrics.aiCoverage,0);
});
test('replay restores canvas and clock even if a consumer throws',()=>{
  const canvas=Object.getOwnPropertyDescriptor(globalThis,'OffscreenCanvas'),clock=Object.getOwnPropertyDescriptor(performance,'now');
  assert.throws(()=>replayTemporal({frames:[frame(100)],events:[{type:'present',atMs:0,frameIndex:0}],onRow(){throw new Error('consumer');}}),/consumer/);
  assert.deepEqual(Object.getOwnPropertyDescriptor(globalThis,'OffscreenCanvas'),canvas);
  assert.deepEqual(Object.getOwnPropertyDescriptor(performance,'now'),clock);
});
test('cadence schedule limits one in-flight capture and orders completions before tied presentation',()=>{
  const events=cadenceEvents({frameCount:24,sourceFps:24,maxFps:24,latencyMs:45});
  assert.equal(events.filter(e=>e.type==='capture').length,12);
  let inFlight=0;
  for(const event of events) {if(event.type==='capture')inFlight++;if(event.type==='complete')inFlight--;assert.ok(inFlight>=0&&inFlight<=1);}
  const tied=cadenceEvents({frameCount:3,sourceFps:24,maxFps:24,latencyMs:1000/24});
  assert.deepEqual(tied.filter(e=>e.atMs===1000/24).map(e=>e.type),['complete','present','capture']);
});
