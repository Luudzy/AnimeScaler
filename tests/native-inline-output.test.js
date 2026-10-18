import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFile} from 'node:fs/promises';
import {crc32} from 'node:zlib';
import {NativeEngine} from '../src/engine/native-engine.js';
import {VideoOverlay} from '../src/content/overlay.js';
import * as protocol from '../src/engine/native-protocol.js';
import {motionReuse} from '../src/engine/temporal-motion.js';
import {TEMPORAL_MAX_SOURCE_AGE_MS,temporalStrength} from '../src/engine/temporal-budget.js';
import {ConfidenceHistory,isSceneCut} from '../src/engine/temporal-history.js';
import {validateNativeService} from '../src/engine/native-service-contract.js';

const png=Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII=','base64');
const limit=16*1024*1024;

function overlayFixture(t) {
  const drawn=[];
  const overlay=Object.create(VideoOverlay.prototype);
  overlay.canvas={isConnected:true,width:1,height:1,getContext:()=>({drawImage(...args){drawn.push(args);}})};
  overlay.video={isConnected:true};
  let decoded=0,closed=0;
  t.mock.method(globalThis,'createImageBitmap',async()=>{decoded++;return {width:1,height:1,close(){closed++;}};});
  const frame={sequence:1,mediaTime:0,width:1,height:1,mime:'image/png',dataUrl:`data:image/png;base64,${png.toString('base64')}`};
  return {overlay,frame,drawn,decoded:()=>decoded,closed:()=>closed};
}

test('presenter rejects PNG header mismatch before bitmap decode',async t=>{
  const hadDecoder=Object.hasOwn(globalThis,'createImageBitmap');
  if(!hadDecoder){globalThis.createImageBitmap=()=>{};t.after(()=>delete globalThis.createImageBitmap);}
  const state=overlayFixture(t);
  const bytes=Buffer.from(png);bytes.writeUInt32BE(8193,16);
  state.frame.dataUrl=`data:image/png;base64,${bytes.toString('base64')}`;
  await assert.rejects(state.overlay.presentFrame(state.frame),{code:'INVALID_PNG_OUTPUT'});
  assert.equal(state.decoded(),0);
});

test('presenter rejects oversized compressed PNG before bitmap decode',async t=>{
  const hadDecoder=Object.hasOwn(globalThis,'createImageBitmap');
  if(!hadDecoder){globalThis.createImageBitmap=()=>{};t.after(()=>delete globalThis.createImageBitmap);}
  const state=overlayFixture(t);
  state.frame.dataUrl=`data:image/png;base64,${'A'.repeat(Math.ceil(limit/3)*4+4)}`;
  await assert.rejects(state.overlay.presentFrame(state.frame),{code:'FRAME_TOO_LARGE'});
  assert.equal(state.decoded(),0);
});

test('presenter bounds concurrent decode and closes stale bitmap',async t=>{
  const hadDecoder=Object.hasOwn(globalThis,'createImageBitmap');
  if(!hadDecoder){globalThis.createImageBitmap=()=>{};t.after(()=>delete globalThis.createImageBitmap);}
  const state=overlayFixture(t);
  let release,current=true,closed=0;
  globalThis.createImageBitmap=()=>new Promise(resolve=>{release=resolve;});
  const first=state.overlay.presentFrame(state.frame,()=>current);
  const second=state.overlay.presentFrame({...state.frame,sequence:2});
  const rejected=assert.rejects(second,{code:'NATIVE_SESSION_BUSY'});
  current=false;
  release({width:1,height:1,close(){closed++;}});
  assert.equal(await first,false);
  await rejected;
  assert.equal(closed,1);
  assert.equal(state.drawn.length,0);
});

// ID 48: deadline is created by client and consumed by overlay in the same content clock.
// No comparison of host performance clocks or unrelated tab/video media timelines.
for(const age of [299,300,301])test(`presenter age ${age} ms guards decode`,async t=>{
  const hadDecoder=Object.hasOwn(globalThis,'createImageBitmap');
  if(!hadDecoder){globalThis.createImageBitmap=()=>{};t.after(()=>delete globalThis.createImageBitmap);}
  const state=overlayFixture(t);
  t.mock.method(performance,'now',()=>age);
  state.frame.expiresAt=300;
  const accepted=await state.overlay.presentFrame(state.frame);
  assert.equal(accepted,age===299);
  assert.equal(state.decoded(),age===299?1:0);
  assert.equal(state.drawn.length,age===299?1:0);
});

test('presenter closes bitmap without draw when source ages during asynchronous decode',async t=>{
  const hadDecoder=Object.hasOwn(globalThis,'createImageBitmap');
  if(!hadDecoder){globalThis.createImageBitmap=()=>{};t.after(()=>delete globalThis.createImageBitmap);}
  const state=overlayFixture(t);
  let now=0;
  t.mock.method(performance,'now',()=>now);
  state.frame.expiresAt=300;
  let release,closed=0;
  globalThis.createImageBitmap=()=>new Promise(resolve=>{release=resolve;});
  const presenting=state.overlay.presentFrame(state.frame);
  now=300;
  release({width:1,height:1,close(){closed++;}});
  assert.equal(await presenting,false);
  assert.equal(closed,1);
  assert.equal(state.drawn.length,0);
});

function setup(t, reply={sequence:7,mediaTime:2.5,width:1,height:1,mime:'image/png',pngBytes:png.buffer.slice(png.byteOffset,png.byteOffset+png.byteLength)}) {
  const engine=new NativeEngine();
  engine.settings={model:'adore',fluidVideo:false};
  engine.currentSize={inputWidth:1,inputHeight:1,outputWidth:1,outputHeight:1};
  let closed=0;
  const bitmap={width:1,height:1,close(){closed++;}};
  engine.worker={postMessage(message,transfer){
    assert.equal(message.type,'frame');assert.equal(message.pngOutput,true);
    assert.equal(message.sequence,7);assert.equal(message.mediaTime,2.5);
    assert.equal(transfer[0],bitmap);
    queueMicrotask(()=>{const request=engine.requests.get(message.id);engine.requests.delete(message.id);clearTimeout(request.timer);request.resolve(reply);});
  },terminate(){}};
  t.after(()=>engine.destroy());
  return {engine,bitmap,closed:()=>closed};
}
test('isolated bitmap produces PNG Data URL with original sequence and media time',async t=>{
  const {engine,bitmap,closed}=setup(t);
  const result=await engine.processImage(bitmap,{sequence:7,mediaTime:2.5});
  assert.deepEqual(result,{sequence:7,mediaTime:2.5,width:1,height:1,mime:'image/png',dataUrl:`data:image/png;base64,${png.toString('base64')}`});
  assert.equal(closed(),1);assert.equal(engine.requests.size,0);
});
for(const [name,override] of [
  ['absent bytes',{pngBytes:undefined}],['empty bytes',{pngBytes:new ArrayBuffer(0)}],
  ['invalid signature',{pngBytes:new Uint8Array(40).buffer}],['truncated PNG',{pngBytes:png.buffer.slice(png.byteOffset,png.byteOffset+24)}],
  ['wrong sequence',{sequence:8}],['wrong media time',{mediaTime:3}],
  ['wrong dimensions',{width:2}],['wrong MIME',{mime:'image/jpeg'}],
]) test(`rejects ${name} and closes input`,async t=>{
  const reply={sequence:7,mediaTime:2.5,width:1,height:1,mime:'image/png',pngBytes:Uint8Array.from(png).buffer,...override};
  const {engine,bitmap,closed}=setup(t,reply);
  await assert.rejects(engine.processImage(bitmap,{sequence:7,mediaTime:2.5}),/INVALID_PNG_OUTPUT/);
  assert.equal(closed(),1);assert.equal(engine.requests.size,0);
});
test('rejects PNG header dimensions differing from configured output',async t=>{
  const bytes=Uint8Array.from(png);new DataView(bytes.buffer).setUint32(16,2);
  const {engine,bitmap}=setup(t,{sequence:7,mediaTime:2.5,width:1,height:1,mime:'image/png',pngBytes:bytes.buffer});
  await assert.rejects(engine.processImage(bitmap,{sequence:7,mediaTime:2.5}),/INVALID_PNG_OUTPUT/);
});
test('PNG encoding excludes simultaneous fluid presentation on shared canvas',async t=>{
  const {engine,bitmap}=setup(t);engine.settings.fluidVideo=true;
  let release,sent;
  const dispatched=new Promise(resolve=>{sent=resolve;});
  engine.worker.postMessage=message=>{release=()=>{const request=engine.requests.get(message.id);engine.requests.delete(message.id);clearTimeout(request.timer);request.resolve({sequence:7,mediaTime:2.5,width:1,height:1,mime:'image/png',pngBytes:Uint8Array.from(png).buffer});};sent();};
  engine.makeFrame=async()=>{throw new Error('presentation must not capture');};
  const processing=engine.processImage(bitmap,{sequence:7,mediaTime:2.5});await dispatched;
  try {assert.equal(await engine.presentFrame({}),false);} finally {release();await processing;}
});
for(const length of [limit,limit+1])test(`PNG byte limit ${length}`,async t=>{
  const bytes=new Uint8Array(length),end=png.length-12;
  bytes.set(png.subarray(0,end));
  const view=new DataView(bytes.buffer);view.setUint32(end,length-png.length-12);view.setUint32(end+4,0x72614e64);
  view.setUint32(length-16,crc32(bytes.subarray(end+4,length-16)));
  bytes.set(png.subarray(end),length-12);
  const {engine,bitmap}=setup(t,{sequence:7,mediaTime:2.5,width:1,height:1,mime:'image/png',pngBytes:bytes.buffer});
  if(length===limit)assert.equal((await engine.processImage(bitmap,{sequence:7,mediaTime:2.5})).mime,'image/png');
  else await assert.rejects(engine.processImage(bitmap,{sequence:7,mediaTime:2.5}),/FRAME_TOO_LARGE/);
});
for(const mode of ['dispatch','timeout','destroy','closed','invalid crop','invalid sequence'])test(`input cleanup on ${mode}`,async t=>{
  const {engine,bitmap,closed}=setup(t);
  const options={sequence:7,mediaTime:2.5};
  if(mode==='dispatch')engine.worker.postMessage=()=>{throw new Error('dispatch rejected');};
  if(mode==='timeout')t.mock.timers.enable({apis:['setTimeout']});
  if(mode==='timeout'||mode==='destroy')engine.worker.postMessage=()=>{};
  if(mode==='closed')await engine.destroy();
  if(mode==='invalid crop')options.crop={x:0,y:0,width:2,height:1};
  if(mode==='invalid sequence')options.sequence=-1;
  const pending=engine.processImage(bitmap,options);const rejection=assert.rejects(pending);
  await new Promise(resolve=>setImmediate(resolve));
  if(mode==='timeout')t.mock.timers.tick(35000);
  if(mode==='destroy')await engine.destroy();
  await rejection;assert.equal(closed(),1);assert.equal(engine.requests.size,0);
});

async function workerHarness(blob=new Blob([png],{type:'image/png'})) {
  const events=[];
  const context={drawImage(){events.push('draw');},putImageData(){events.push('pixels');},clearRect(){},getImageData(x,y,w,h){return {data:new Uint8ClampedArray(w*h*4)};}};
  class Canvas {constructor(w,h){this.width=w;this.height=h;}getContext(){return context;}async convertToBlob(options){events.push('encode');assert.deepEqual({...options},{type:'image/png'});return blob;}}
  class Socket {
    static OPEN=1;
    constructor(){this.readyState=1;queueMicrotask(()=>this.onopen());}
    send(data){queueMicrotask(()=>{
      if(typeof data==='string')this.onmessage({data:JSON.stringify({backend:'rocm',protocol:1,model:'adore',precision:'fp16',
        models:['adore'],revisions:{adore:'b'.repeat(64)},metadata:{adore:{id:'adore',status:'external-baseline'}}})});
      else {const response=new ArrayBuffer(28);protocol.writeHeader(response,new DataView(data).getUint32(4,true),1,1,1,1);this.onmessage({data:response});}
    });}
    close(){}
  }
  const replies=[];
  const scope={...protocol,motionReuse,TEMPORAL_MAX_SOURCE_AGE_MS,temporalStrength,ConfidenceHistory,isSceneCut,validateNativeService,createNativeGpuRenderer:async()=>null,
    OffscreenCanvas:Canvas,WebSocket:Socket,ImageData:class {constructor(data,width,height){Object.assign(this,{data,width,height});}},
    ArrayBuffer,Uint8Array,Uint8ClampedArray,DataView,Blob,performance,setTimeout,clearTimeout,
    self:{postMessage(message,transfer){replies.push({message,transfer});}}};
  const realm=vm.createContext(scope);
  const presenter=await readFile(new URL('../src/engine/temporal-presenter.js',import.meta.url),'utf8');
  vm.runInContext(presenter.replace(/^import .*;\r?\n/gm,'').replace('export class','class'),realm);
  const source=await readFile(new URL('../src/engine/native-worker.js',import.meta.url),'utf8');
  vm.runInContext(source.replace(/^import .*;\r?\n/gm,''),realm);
  const canvas=new Canvas(1,1);
  await scope.self.onmessage({data:{id:1,type:'init',canvas,connection:{url:'http://127.0.0.1:8766',token:'a'.repeat(64)},model:'adore',backend:'rocm'}});
  assert.equal(replies[0].message.error,undefined);
  let closed=0;const bitmap={width:40,height:40,close(){closed++;}};
  return {events,async frame(extra={}){await scope.self.onmessage({data:{id:2,type:'frame',bitmap,crop:{x:0,y:0,width:40,height:40},size:{inputWidth:40,inputHeight:40,outputWidth:1,outputHeight:1},sequence:7,mediaTime:2.5,sourceSequence:7,pngOutput:true,...extra}});return replies.at(-1);},closed:()=>closed};
}
test('fluid PNG encodes after drawing newly accepted AI composition',async()=>{
  const worker=await workerHarness();const {message}=await worker.frame({fluidVideo:true});
  assert.equal(message.error,undefined);assert.equal(message.result.temporal.acceptedResults,1);
  assert.deepEqual(worker.events.slice(-2),['draw','encode']);
});
test('worker encodes composed host canvas as lossless PNG and transfers bytes',async()=>{
  const worker=await workerHarness();const {message,transfer}=await worker.frame();
  assert.equal(message.error,undefined);
  assert.equal(message.result.mime,'image/png');assert.equal(message.result.sequence,7);assert.equal(message.result.mediaTime,2.5);
  assert.equal(message.result.width,1);assert.equal(message.result.height,1);
  assert.deepEqual(Buffer.from(message.result.pngBytes),png);assert.equal(transfer[0],message.result.pngBytes);
  assert.equal(worker.closed(),1);
});
test('worker rejects oversized PNG before reading blob bytes',async()=>{
  const worker=await workerHarness({type:'image/png',size:limit+1,arrayBuffer(){throw new Error('must not read');}});
  const {message}=await worker.frame();assert.match(message.error,/FRAME_TOO_LARGE/);assert.equal(worker.closed(),1);
});
test('worker closes bitmap on encoding failure',async()=>{
  const worker=await workerHarness({type:'image/png',size:1,arrayBuffer(){throw new Error('encoding failed');}});
  const {message}=await worker.frame();assert.match(message.error,/encoding failed/);assert.equal(worker.closed(),1);
});
