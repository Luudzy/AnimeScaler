import test from 'node:test';
import assert from 'node:assert/strict';
import { NativeEngine } from '../src/engine/native-engine.js';
import { normalizeSettings } from '../src/core/settings.js';

const TOKEN='a'.repeat(64);
const INFO={protocol:1,backend:'tensorrt',model:'adore',precision:'fp16',gpu:'test',architecture:'test',
  models:['adore'],revisions:{adore:'b'.repeat(64)},metadata:{adore:{id:'adore',status:'external-baseline'}}};

async function initializeWithProfile(t, available, video={videoWidth:1920,videoHeight:1080}, preset='1080p', busyResponses=0, adaptive=false) {
  const calls=[];
  let busyCount=0;
  class FakeWorker {
    postMessage(message) {
      calls.push(message);
      if(message.type==='init')queueMicrotask(()=>this.onmessage?.({data:{id:message.id,result:INFO}}));
      if(message.type==='profile-check')queueMicrotask(()=>{
        if(busyCount<busyResponses){
          busyCount++;
          this.onmessage?.({data:{id:message.id,error:'busy',errorCode:'NATIVE_SESSION_BUSY'}});
        } else this.onmessage?.({data:{id:message.id,result:{
          profileCheck:true,supported:true,available,model:message.model,shapes:message.shapes,
          missingShapes:available?[]:message.shapes,
        }}});
      });
    }
    terminate() {}
  }
  const oldWorker=globalThis.Worker;
  globalThis.Worker=FakeWorker;
  t.after(()=>{globalThis.Worker=oldWorker;});
  const engine=new NativeEngine();
  try {
    await engine.initialize({canvas:{transferControlToOffscreen:()=>({})},video,
      settings:normalizeSettings({model:'adore',backend:'tensorrt',preset,adaptive}),
      connection:{url:'http://127.0.0.1:8766',token:TOKEN}});
    return {calls,engine};
  } catch(error) {
    await engine.destroy();
    throw error;
  }
}

test('TensorRT initialization checks the exact profile before returning to playback',async t=>{
  const {calls,engine}=await initializeWithProfile(t,true);
  t.after(()=>engine.destroy());
  assert.deepEqual(calls.map(call=>call.type),['init','profile-check']);
  assert.deepEqual({model:calls[1].model,shapes:calls[1].shapes},
    {model:'adore',shapes:[[960,540]]});
});

test('TensorRT initialization blocks before playback when the exact shape profile is missing',async t=>{
  await assert.rejects(initializeWithProfile(t,false),/Perfis TensorRT ausentes.*960x540/i);
});

test('TensorRT preflight requests the variable-aspect input shape selected for the source',async t=>{
  const {calls,engine}=await initializeWithProfile(t,true,{videoWidth:1080,videoHeight:1920},'720p');
  t.after(()=>engine.destroy());
  assert.deepEqual({model:calls[1].model,shapes:calls[1].shapes},
    {model:'adore',shapes:[[640,1136]]});
});

test('TensorRT profile preflight retries while another local render owns the service',async t=>{
  const {calls,engine}=await initializeWithProfile(t,true,undefined,undefined,1);
  t.after(()=>engine.destroy());
  assert.equal(calls.filter(call=>call.type==='profile-check').length,2);
});

test('adaptive TensorRT preflight requires every variable-aspect level before playback',async t=>{
  const {calls,engine}=await initializeWithProfile(t,true,{videoWidth:1080,videoHeight:1920},'720p',0,true);
  t.after(()=>engine.destroy());
  assert.deepEqual(calls[1].shapes,[[640,1136],[480,852],[320,568],[240,424],[160,284]]);
});

test('adaptive TensorRT warning lists every missing shape preparation command',async t=>{
  await assert.rejects(
    initializeWithProfile(t,false,{videoWidth:1080,videoHeight:1920},'720p',0,true),
    /640x1136.*480x852.*320x568.*240x424.*160x284/,
  );
});
