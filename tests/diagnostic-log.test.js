import test from 'node:test';
import assert from 'node:assert/strict';
import { DiagnosticLog, createDiagnosticStore, DIAGNOSTIC_KEY } from '../src/core/diagnostic-log.js';

test('throughput guard diagnostics retain measured and configured cadence fields', () => {
  const events = [];
  const diagnostics = new DiagnosticLog({ send: event => events.push(event) });
  diagnostics.event('throughput_guard', {
    backend: 'rocm', measuredFps: 6.15, targetFps: 24, minimumFps: 19.2, windowMs: 5000,
  });
  assert.deepEqual(events[0].data, {
    backend: 'rocm', measuredFps: 6.15, targetFps: 24, minimumFps: 19.2, windowMs: 5000,
  });
});

test('presentation windows report mean AI coverage independently of inference windows', () => {
  let now = 0; const events = [];
  const log = new DiagnosticLog({ send: event => events.push(event), now: () => now, intervalMs: 1000 });
  for (let n = 0; n < 30; n++) {
    now = (n + 1) * 1000 / 30;
    log.present({ aiCoverage: n % 2, historyAgeMs: n * 4 });
  }
  log.flush();
  const presentations = events.filter(event => event.type === 'presentation');
  assert.equal(presentations.length, 1);
  assert.equal(presentations[0].data.presentedFrames, 30);
  assert.equal(presentations[0].data.aiCoverageMean, .5);
  assert.equal(presentations[0].data.lowConfidenceRatio, .5);
  assert.ok(presentations[0].data.historyAgeP95Ms >= 108);
  assert.equal(events.some(event => event.type === 'performance'), false);
});

test('presentation model changes split contribution windows and retain safe context',()=>{
  let now=0; const events=[];
  const log=new DiagnosticLog({send:event=>events.push(event),now:()=>now});
  now=100;log.present({aiCoverage:.1,historyAgeMs:20},{model:'adore-lite-v3',token:'secret'});
  now=200;log.present({aiCoverage:.9,historyAgeMs:40},{model:'zenkai-animevideo-30k',token:'secret'});
  now=300;log.flush();
  assert.deepEqual(events.map(event=>event.data.model),['adore-lite-v3','zenkai-animevideo-30k']);
  assert.deepEqual(events.map(event=>event.data.aiCoverageMean),[.1,.9]);
  assert.equal(JSON.stringify(events).includes('secret'),false);
});

test('reused frame and byte totals aggregate without fragmenting log windows',()=>{
  const events=[];let now=0;
  const log=new DiagnosticLog({send:e=>events.push(e),now:()=>now});
  now=10;log.frame({totalMs:10,responseBytes:1000,reused:false});
  now=20;log.frame({totalMs:1,responseBytes:24,reused:true});
  log.flush();
  assert.equal(events.length,1);assert.equal(events[0].data.frames,2);
  assert.equal(events[0].data.reusedFrames,1);assert.equal(events[0].data.responseBytesTotal,1024);
});

test('temporal counters survive filtering without splitting frame windows', () => {
  const events=[];
  const log=new DiagnosticLog({send:event=>events.push(event),now:()=>100});
  log.frame({totalMs:20,temporal:{acceptedResults:1}});
  log.frame({totalMs:20,temporal:{acceptedResults:2}});
  log.event('temporal_history',{sceneCuts:1,rejectedResults:3,acceptedResults:2,code:'scene',pixels:'private'});
  log.flush();
  assert.equal(events.length,2);
  const counters=events.find(e=>e.type==='temporal_history').data;
  assert.deepEqual(counters,{sceneCuts:1,rejectedResults:3,acceptedResults:2,code:'scene'});
  assert.equal(events.find(e=>e.data.frames).data.frames,2);
});

test('frames are aggregated, with actual throughput and percentile latency', () => {
  let now = 0;
  const events = [];
  const log = new DiagnosticLog({ send: event => events.push(event), now: () => now });
  for (let i = 1; i <= 150; i++) {
    now = i * 1000 / 30;
    log.frame({ totalMs: 20, inferenceMs: 7, inputWidth: 640, modelPrecision: 'fp16' });
  }
  assert.equal(events.length, 1);
  assert.equal(events[0].data.frames, 150);
  assert.equal(events[0].data.fps, 30);
  assert.equal(events[0].data.p95Ms, 20);
  assert.equal(events[0].data.sessionRunMeanMs, 7);
});

test('performance separates completed outputs from known fresh inference and omits unknown stage samples', () => {
  let now = 0;
  const events = [];
  const log = new DiagnosticLog({ send: event => events.push(event), now: () => now });
  log.frame({ totalMs: 10, inferenceMs: 8, captureMs: 1, reused: false, repeated: false });
  log.frame({ totalMs: 20, reused: true });
  log.frame({ totalMs: 15, inferenceMs: 0, repeated: true });
  now = 1000;
  log.flush();
  const data = events.find(event => event.type === 'performance').data;
  assert.equal(data.processedFrames, 3);
  assert.equal(data.processedFps, 3);
  assert.equal(data.reusedFrames, 1);
  assert.equal(data.reuseKnownFrames, 2);
  assert.equal(data.repeatedFrames, 1);
  assert.equal(data.repeatKnownFrames, 2);
  assert.equal(data.inferenceFrames, 1);
  assert.equal(data.inferenceFps, 1);
  assert.equal(data.sessionRunMeanMs, 8);
  assert.equal(data.captureMeanMs, 1);
  assert.equal('transportMeanMs' in data, false);
  assert.equal('renderMeanMs' in data, false);
});

test('source rate counts distinct media times independently of processed and presented frames', () => {
  let now=0;
  const events = [];
  const log = new DiagnosticLog({ send: event => events.push(event), now: () => now });
  for(let frame=0;frame<30;frame++) {
    now=frame*1000/30;
    log.sourceFrame(frame/30,now,{model:'adore-lite-v3'});
  }
  now=1000;log.sourceFrame(29/30,now,{model:'adore-lite-v3'});
  log.flush();
  const data = events.find(event => event.type === 'source_rate').data;
  assert.equal(data.sourceCallbacks, 31);
  assert.equal(data.sourceFrames, 30);
  assert.equal(data.sourceFps, 30);
  assert.equal(data.model, 'adore-lite-v3');
});

test('source-rate context resets after a periodic window flush',()=>{
  let now=0;const events=[];
  const log=new DiagnosticLog({send:event=>events.push(event),now:()=>now,intervalMs:100});
  log.sourceFrame(0,0,{model:'model-a'});
  now=100;log.sourceFrame(1/30,now,{model:'model-a'});
  now=110;log.sourceFrame(0,now,{model:'model-b'});
  now=210;log.sourceFrame(1/30,now,{model:'model-b'});
  log.flush();
  assert.deepEqual(events.filter(event=>event.type==='source_rate').map(event=>event.data.model),['model-a','model-b']);
});

test('configuration changes split windows and failures never break playback', () => {
  let now = 0;
  const events = [];
  const log = new DiagnosticLog({ now: () => now, send: event => events.push(event) });
  now = 100; log.frame({totalMs: 20, inputWidth: 640});
  now = 200; log.frame({totalMs: 40, inputWidth: 1280});
  log.flush();
  assert.deepEqual(events.map(e => e.data.inputWidth), [640, 1280]);
  const broken = new DiagnosticLog({ send: () => { throw new Error('context invalidated'); } });
  assert.doesNotThrow(() => broken.event('error', { message: 'test' }));
});

test('error events retain allow-listed session correlation without arbitrary error fields', () => {
  const events = [];
  const log = new DiagnosticLog({ send: event => events.push(event), now: () => 1 });
  const error = Object.assign(new Error('STALE_SESSION'), { code: 'STALE_SESSION', diagnostic: {
    staleOrigin: 'native_client', sessionTag: '0123456789abcdef', messageType: 'FRAME_OUT',
    sequence: 42, token: 'secret', dataUrl: 'secret',
  } });
  log.error(error);
  assert.deepEqual(events.at(-1).data, {
    code: 'STALE_SESSION', message: 'STALE_SESSION', staleOrigin: 'native_client',
    sessionTag: '0123456789abcdef', messageType: 'FRAME_OUT', sequence: 42,
  });
  assert.equal(JSON.stringify(events).includes('secret'), false);
});

test('native stage timings aggregate without splitting one event per frame', () => {
  const events = [];
  let now = 0;
  const log = new DiagnosticLog({send: event => events.push(event), now: () => now});
  now = 20; log.frame({backend:'rocm', captureMethod:'VideoFrame.copyTo',totalMs:20,captureMs:1,transportMs:18,renderMs:1});
  now = 60; log.frame({backend:'rocm', captureMethod:'VideoFrame.copyTo',totalMs:40,captureMs:3,transportMs:34,renderMs:3});
  log.flush();
  assert.equal(events.length, 1);
  assert.equal(events[0].data.captureMeanMs, 2);
  assert.equal(events[0].data.transportMeanMs, 26);
  assert.equal(events[0].data.renderMeanMs, 2);
  assert.equal(events[0].data.backend, 'rocm');
});

test('worker boundary measurements aggregate without fabricated samples or per-frame windows',()=>{
  const events=[];let now=0;
  const log=new DiagnosticLog({send:event=>events.push(event),now:()=>now});
  now=20;log.frame({totalMs:20,sourceFrameMs:2,workerMs:12,workerRoundTripMs:15,workerOverheadMs:3});
  now=40;log.frame({totalMs:20,sourceFrameMs:4,workerMs:10,workerRoundTripMs:15,workerOverheadMs:5});
  now=60;log.frame({totalMs:20});
  log.flush();
  assert.equal(events.length,1);
  assert.equal(events[0].data.sourceFrameMeanMs,3);
  assert.equal(events[0].data.workerMeanMs,11);
  assert.equal(events[0].data.workerRoundTripMeanMs,15);
  assert.equal(events[0].data.workerOverheadMeanMs,4);
  now=80;log.frame({totalMs:20});log.flush();
  assert.equal('workerMeanMs' in events[1].data,false);
});

test('presentation diagnostics retain coverage components and discard private data',()=>{
  const events=[];let now=0;
  const log=new DiagnosticLog({send:event=>events.push(event),now:()=>now});
  now=40;log.present({aiCoverage:.2,historyAgeMs:40,motionCoverage:.8,confidenceCoverage:.4,temporalWeight:.5}, {model:'adore-lite-v3',token:'private'});
  log.flush();
  assert.equal(events[0].data.motionCoverageMean,.8);
  assert.equal(events[0].data.confidenceCoverageMean,.4);
  assert.equal(events[0].data.temporalWeightMean,.5);
  assert.equal(JSON.stringify(events).includes('private'),false);
});

test('starting after a long idle does not include idle time in first FPS window', () => {
  let now = 0;
  const events = [];
  const log = new DiagnosticLog({now: () => now, send: e => events.push(e)});
  now = 60000; log.event('start');
  now += 100; log.frame({totalMs: 20}); log.flush();
  assert.equal(events.at(-1).data.fps, 10);
  assert.equal('sessionRunMeanMs' in events.at(-1).data, false);
});

test('storage serializes concurrent appends, bounds history and removes sensitive fields', async () => {
  const data = {};
  const storage = {get: async key => ({[key]: data[key]}), set: async update => Object.assign(data, update)};
  const store = createDiagnosticStore(storage, 3);
  await Promise.all(Array.from({length: 6}, (_, i) => store.append({type: 'error', session: 's', data: {
    code: `E${i}`, message: 'failed https://site.test/video?token=secret', url: 'secret', streamId: 'secret',
  }})));
  const history = await store.read();
  assert.equal(history.events.length, 3);
  assert.equal(history.dropped, 3);
  assert.equal(history.events[2].data.code, 'E5');
  assert.equal(JSON.stringify(history).includes('secret'), false);
  await store.clear();
  assert.equal(data[DIAGNOSTIC_KEY].events.length, 0);
});
