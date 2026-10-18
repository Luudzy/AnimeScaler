import test from 'node:test';
import assert from 'node:assert/strict';
import {NativeEngine} from '../src/engine/native-engine.js';
import {normalizeSettings} from '../src/core/settings.js';
import {sessionTag} from '../src/core/session-diagnostics.js';

function engineWithClock(lastPresentationAt){
  const engine=new NativeEngine();
  engine.settings={fluidVideo:true,adaptive:false,model:'adore-lite'};
  engine.currentSize={outputWidth:80,outputHeight:45};
  engine.lastPresentationAt=lastPresentationAt;
  engine.makeFrame=async()=>({width:80,height:45,close(){}});
  engine.diagnostics={frame(){},event(){},flush(){}};
  const calls=[];
  engine.call=async(type)=>{
    calls.push(type);
    return type==='frame'?{captureMs:0,temporal:{acceptedResults:1}}:
      {presented:true,temporal:{acceptedResults:1,aiCoverage:1}};
  };
  return {engine,calls};
}

test('fluid completion wakes presentation when the capture clock has stopped',async()=>{
  const {engine,calls}=engineWithClock(performance.now()-1000);
  const result=await engine.processFrame({});
  assert.deepEqual(calls,['frame','present']);
  assert.equal(result.temporal.aiCoverage,1);
});

test('active presentation clock is not duplicated by inference completion',async()=>{
  const {engine,calls}=engineWithClock(performance.now());
  await engine.processFrame({});
  assert.deepEqual(calls,['frame']);
  clearTimeout(engine.presentationWake);
});

test('capture stopping just before completion gets one delayed presentation wake',async()=>{
  const {engine,calls}=engineWithClock(performance.now());
  await engine.processFrame({});
  await new Promise(resolve=>setTimeout(resolve,110));
  assert.deepEqual(calls,['frame','present']);
});

test('still comparison never requests a fluid presentation wake',async()=>{
  const {engine,calls}=engineWithClock(performance.now()-1000);
  await engine.processFrame({},null,{still:true});
  assert.deepEqual(calls,['frame']);
});

test('opening comparison cancels the pending idle wake from live playback',async()=>{
  const {engine,calls}=engineWithClock(performance.now());
  await engine.processFrame({});
  await engine.processFrame({},null,{still:true});
  await new Promise(resolve=>setTimeout(resolve,110));
  assert.deepEqual(calls,['frame','frame']);
});

test('destroy cancels presentation wake and releases its source reference',async()=>{
  const {engine,calls}=engineWithClock(performance.now());
  await engine.processFrame({});
  await engine.destroy();
  await new Promise(resolve=>setTimeout(resolve,110));
  assert.deepEqual(calls,['frame']);
});

test('actual presentation reports contribution independently of inference completion',async()=>{
  const {engine,calls}=engineWithClock(performance.now()-1000);
  const delivered=[];
  engine.diagnostics.present=metrics=>delivered.push({...metrics});
  assert.equal(await engine.presentFrame({}),true);
  assert.deepEqual(calls,['present']);
  assert.equal(delivered.length,1);
  assert.equal(delivered[0].aiCoverage,1);
  assert.equal(await engine.presentFrame({}),false,'pacer rejects duplicated presentations');
  assert.equal(delivered.length,1,'rejected presentation cannot inflate coverage or cadence');
  await engine.destroy();
});

test('reconfiguration cancels idle presentation with an obsolete crop',async()=>{
  const {engine,calls}=engineWithClock(performance.now());
  engine.settings=normalizeSettings({backend:'rocm',model:'adore-lite-v3',fluidVideo:true});
  await engine.processFrame({}, {x:0,y:0,width:80,height:45});
  await engine.reconfigure({videoWidth:160,videoHeight:90},0,{x:80,y:0,width:80,height:90});
  await new Promise(resolve=>setTimeout(resolve,110));
  assert.deepEqual(calls,['frame']);
  await engine.destroy();
});

test('a presentation awaiting capture cannot mix old crop with reconfigured dimensions',async()=>{
  const {engine,calls}=engineWithClock(performance.now()-1000);
  engine.settings=normalizeSettings({backend:'rocm',model:'adore-lite-v3',fluidVideo:true});
  let finish,closed=0;
  engine.makeFrame=()=>new Promise(resolve=>{finish=resolve;});
  const pending=engine.presentFrame({}, {x:0,y:0,width:80,height:45});
  await engine.reconfigure({videoWidth:160,videoHeight:90});
  finish({close(){closed++;}});
  assert.equal(await pending,false);
  assert.deepEqual(calls,[]);
  assert.equal(closed,1);
  await engine.destroy();
});

test('native boundary timings measure frame construction and worker round trip separately',async t=>{
  const {engine}=engineWithClock(0);engine.settings.fluidVideo=false;
  let now=0;t.mock.method(performance,'now',()=>now);
  engine.makeFrame=async()=>{now+=2;return {width:80,height:45,close(){}};};
  engine.call=async()=>{now+=15;return {captureMs:1,workerMs:12};};
  const row=await engine.processFrame({});
  assert.equal(row.sourceFrameMs,2);
  assert.equal(row.workerRoundTripMs,15);
  assert.equal(row.workerOverheadMs,3);
  assert.equal(row.captureMs,3);
  await engine.destroy();
});

for(const mode of ['reconfigure','updateSettings'])test(`${mode} waits for a dispatched presentation before changing geometry`,async()=>{
  const {engine}=engineWithClock(performance.now()-1000);
  engine.settings=normalizeSettings({backend:'rocm',model:'adore-lite-v3',fluidVideo:true});
  let deliver,dispatched;
  const sent=new Promise(resolve=>{dispatched=resolve;});
  engine.call=(type,data)=>new Promise(resolve=>{
    deliver=()=>{assert.equal(engine.currentSize.outputWidth,data.size.outputWidth);resolve({temporal:{}});};
    dispatched();
  });
  const presentation=engine.presentFrame({});await sent;
  let configured=false;
  const video={videoWidth:320,videoHeight:180};
  const configuration=(mode==='reconfigure'?engine.reconfigure(video):engine.updateSettings(engine.settings,video)).then(()=>{configured=true;});
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(configured,false,'an outstanding worker draw must finish before geometry changes');
  deliver();await presentation;await configuration;
  assert.equal(configured,true);
  await engine.destroy();
});

test('inference and presentation reserve source order before asynchronous capture',async()=>{
  const {engine}=engineWithClock(performance.now()-1000);
  let finish;
  engine.makeFrame=()=>new Promise(resolve=>{finish=resolve;});
  const presentation=engine.presentFrame({});const finishPresentation=finish;
  const messages=[];engine.call=async(type,data)=>{messages.push({type,sequence:data.sourceSequence});return {captureMs:0};};
  engine.settings.fluidVideo=false;
  engine.makeFrame=async()=>({width:80,height:45,close(){}});
  await engine.processFrame({});finishPresentation({close(){}});await presentation;
  assert.ok(Number.isInteger(messages[0].sequence));
  assert.ok(messages[0].sequence>messages[1].sequence,'later inference retains newer order despite earlier delivery');
  await engine.destroy();
});

test('a failed presentation barrier cannot authorize changing geometry',async()=>{
  const {engine}=engineWithClock(performance.now()-1000);
  engine.settings=normalizeSettings({backend:'rocm',model:'adore-lite-v3',fluidVideo:true});
  const oldSize=engine.currentSize;
  let fail,sent;
  const dispatched=new Promise(resolve=>{sent=resolve;});
  engine.call=()=>new Promise((resolve,reject)=>{fail=reject;sent();});
  const presentation=engine.presentFrame({});
  const rejectedPresentation=assert.rejects(presentation,/timeout/);
  await dispatched;
  const configuration=engine.reconfigure({videoWidth:320,videoHeight:180});
  const rejectedConfiguration=assert.rejects(configuration,/timeout/);
  await new Promise(resolve=>setImmediate(resolve));
  fail(new Error('timeout'));
  await rejectedPresentation;await rejectedConfiguration;
  assert.equal(engine.currentSize,oldSize,'timeout does not prove queued worker drawing has stopped');
  assert.equal(engine.reconfiguring,false);
  await engine.destroy();
});

for(const mode of ['reconfigure','updateSettings'])test(`${mode} queued behind other work retains a failed presentation barrier`,async()=>{
  const {engine}=engineWithClock(performance.now()-1000);
  engine.settings=normalizeSettings({backend:'rocm',model:'adore-lite-v3',fluidVideo:true});
  const oldSize=engine.currentSize;
  let unblock,fail,sent;
  const queueBlock=engine.exclusive(()=>new Promise(resolve=>{unblock=resolve;}));
  await new Promise(resolve=>setImmediate(resolve));
  const dispatched=new Promise(resolve=>{sent=resolve;});
  engine.call=()=>new Promise((resolve,reject)=>{fail=reject;sent();});
  const presentation=engine.presentFrame({});
  const rejectedPresentation=assert.rejects(presentation,/timeout/);
  await dispatched;
  const video={videoWidth:320,videoHeight:180};
  const configure=()=>mode==='reconfigure'?engine.reconfigure(video):engine.updateSettings(engine.settings,video);
  const configuration=configure(),rejectedConfiguration=assert.rejects(configuration,/timeout/);
  fail(new Error('timeout'));await rejectedPresentation;
  unblock();await queueBlock;await rejectedConfiguration;
  await assert.rejects(configure(),/timeout/,'a later retry cannot bypass missing worker acknowledgement');
  await assert.rejects(engine.presentFrame({}),/timeout/);
  await assert.rejects(engine.processFrame({}),/timeout/);
  assert.equal(engine.currentSize,oldSize);
  await engine.destroy();
});

test('destroy rejects a real Worker request immediately, ignores late replies and is idempotent', async t => {
  const originalWorker = globalThis.Worker;
  t.after(() => { globalThis.Worker = originalWorker; });
  let worker;
  let terminated = 0;
  const messages = [];
  globalThis.Worker = class {
    constructor() { worker = this; }
    postMessage(message) {
      messages.push(message);
      if (message.type === 'init') queueMicrotask(() => this.onmessage({ data: { id: message.id,
        result: { backend: 'rocm', gpu: 'mock', precision: 'fp16' } } }));
    }
    terminate() { terminated++; }
  };
  const engine = new NativeEngine();
  const diagnosticEvents = [];
  engine.diagnostics = { event(type, data) { diagnosticEvents.push({ type, data }); }, flush() {} };
  await engine.initialize({ canvas: { transferControlToOffscreen() { return {}; } },
    video: { videoWidth: 320, videoHeight: 180 },
    settings: normalizeSettings({ backend: 'rocm', model: 'adore', adaptive: false }), connection: {}, sessionId: 'session-1' });
  assert.equal(diagnosticEvents.find(event => event.type === 'native_ready').data.sessionTag, sessionTag('session-1'));
  const request = engine.call('present', {});
  const rejected = assert.rejects(request, /ROCm encerrado/);
  const lateReply = worker.onmessage;
  const id = messages.at(-1).id;
  const closing = engine.destroy();
  assert.equal(engine.closed, true);
  assert.equal(engine.requests.size, 0);
  assert.equal(terminated, 1);
  assert.equal(engine.destroy(), closing, 'destroy returns the same cleanup Promise');
  await rejected;
  await closing;
  lateReply({ data: { id, result: { presented: true } } });
  const count = messages.length;
  await assert.rejects(engine.call('frame', {}), /ROCm encerrado/);
  assert.equal(messages.length, count, 'closed engine cannot dispatch more work');
  assert.equal(terminated, 1);
});

test('NativeEngine preserves stable error codes returned by its worker', async t => {
  const originalWorker = globalThis.Worker;
  t.after(() => { globalThis.Worker = originalWorker; });
  let worker;
  const messages = [];
  globalThis.Worker = class {
    constructor() { worker = this; }
    postMessage(message) {
      messages.push(message);
      if (message.type === 'init') queueMicrotask(() => this.onmessage({ data: { id: message.id,
        result: { backend: 'rocm', gpu: 'mock', precision: 'fp16' } } }));
    }
    terminate() {}
  };
  const engine = new NativeEngine();
  const diagnosticEvents = [];
  engine.diagnostics = { event(type, data) { diagnosticEvents.push({ type, data }); }, flush() {} };
  await engine.initialize({ canvas: { transferControlToOffscreen() { return {}; } },
    video: { videoWidth: 320, videoHeight: 180 },
    settings: normalizeSettings({ backend: 'rocm', model: 'adore', adaptive: false }), connection: {}, sessionId: 'session-1' });
  assert.equal(diagnosticEvents.find(event => event.type === 'native_ready').data.sessionTag, sessionTag('session-1'));

  const request = engine.call('frame', {});
  const id = messages.at(-1).id;
  worker.onmessage({ data: { id, error: '[BACKEND_UNAVAILABLE] Falha no serviço ROCm.', errorCode: 'BACKEND_UNAVAILABLE' } });
  await assert.rejects(request, error => error.code === 'BACKEND_UNAVAILABLE');
  await engine.destroy();
});

test('NativeEngine worker crashes report the active TensorRT or OpenCL backend', async t => {
  const originalWorker = globalThis.Worker;
  t.after(() => { globalThis.Worker = originalWorker; });
  for (const { backend, model, expected } of [
    { backend: 'tensorrt', model: 'zenkai-1-0', expected: 'Serviço TensorRT desconectado. Abra Iniciar-TensorRT.cmd.' },
    { backend: 'rocm', model: 'anime4kcpp-acnet-f8b4', expected: 'Worker OpenCL desconectado. Verifique o serviço nativo configurado.' },
  ]) {
    let worker;
    globalThis.Worker = class { constructor() { worker = this; } postMessage() {} terminate() {} };
    const engine = new NativeEngine();
    engine.diagnostics = { event() {}, flush() {} };
    const initializing = engine.initialize({ canvas: { transferControlToOffscreen() { return {}; } },
      video: { videoWidth: 320, videoHeight: 180 }, settings: normalizeSettings({ backend, model, adaptive: false }), connection: {} });
    await new Promise(resolve => setImmediate(resolve));
    worker.onerror();
    await assert.rejects(initializing, error => error.code === 'BACKEND_UNAVAILABLE' && error.message === expected);
    await engine.destroy();
  }
});

for (const mode of ['reconfigure', 'updateSettings']) test(`${mode} cannot mutate after destroy during an awaited barrier`, async () => {
  const { engine } = engineWithClock(0);
  engine.settings = normalizeSettings({ backend: 'rocm', model: 'adore', adaptive: false });
  const settings = engine.settings;
  const size = engine.currentSize;
  let release;
  engine.presentationFlight = new Promise(resolve => { release = resolve; });
  const video = { videoWidth: 320, videoHeight: 180 };
  const configuration = mode === 'reconfigure' ? engine.reconfigure(video)
    : engine.updateSettings({ ...settings, preset: 'quality' }, video);
  const rejected = assert.rejects(configuration, /ROCm encerrado/);
  await new Promise(resolve => setImmediate(resolve));
  const closing = engine.destroy();
  release();
  await rejected;
  await closing;
  assert.equal(engine.settings, settings);
  assert.equal(engine.currentSize, size);
});

test('initialize cannot create Worker after destroy while awaiting private connection JSON', async t => {
  const originalChrome = globalThis.chrome;
  const originalWorker = globalThis.Worker;
  globalThis.chrome = { runtime: { getURL: path => path } };
  let created = 0;
  globalThis.Worker = class { constructor() { created++; } };
  t.after(() => { globalThis.chrome = originalChrome; globalThis.Worker = originalWorker; });
  let deliver;
  const json = new Promise(resolve => { deliver = resolve; });
  t.mock.method(globalThis, 'fetch', async () => ({ ok: true, json: () => json }));
  const engine = new NativeEngine();
  engine.diagnostics = { event() {}, flush() {} };
  const initializing = engine.initialize({ canvas: {}, video: { videoWidth: 320, videoHeight: 180 },
    settings: normalizeSettings({ backend: 'rocm', adaptive: false }) });
  const rejected = assert.rejects(initializing, /ROCm encerrado/);
  await new Promise(resolve => setImmediate(resolve));
  await engine.destroy();
  deliver({});
  await rejected;
  assert.equal(created, 0);
  assert.equal(engine.nativeInfo, undefined);
});
