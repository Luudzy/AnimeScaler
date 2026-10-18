import test from 'node:test';
import assert from 'node:assert/strict';
import {renderStillComparison} from '../viewer/still-comparison.js';
import {NativeEngine} from '../src/engine/native-engine.js';

test('comparison pauses capture before copying and requests a still AI result at the same size', async () => {
  const events = [];
  const capture = {pause(){events.push('pause');}};
  const frozen = {close(){events.push('close');}};
  const crop = {x:5,y:10,width:640,height:360};
  const context = {drawImage(...args){events.push('copy');assert.deepEqual(args,[frozen,5,10,640,360,0,0,1920,1080]);}};
  const original = {getContext(){return context;}};
  const engine = {currentSize:{outputWidth:1920,outputHeight:1080},async processFrame(video,rect,options){
    events.push('infer');assert.equal(video,frozen);assert.equal(rect,crop);assert.equal(options.still,true);
    return {outputWidth:1920,outputHeight:1080};
  }};
  const result = await renderStillComparison({capture,original,engine,crop,snapshot:async video=>{assert.equal(video,capture);events.push('snapshot');return frozen;}});
  assert.deepEqual(events,['pause','snapshot','copy','infer','close']);
  assert.equal(original.width,result.outputWidth);
  assert.equal(original.height,result.outputHeight);
});

test('native still comparison bypasses temporal blending without changing live settings',async()=>{
  const engine=new NativeEngine();
  engine.settings={fluidVideo:true,adaptive:false,model:'adore-lite'};
  engine.currentSize={outputWidth:1280,outputHeight:720};
  engine.makeFrame=async()=>({width:640,height:360,close(){}});
  engine.diagnostics={frame(){}};
  const modes=[];
  engine.call=async(type,data)=>{modes.push(data.fluidVideo);return {captureMs:0};};
  await engine.processFrame({},null,{still:true});
  await engine.processFrame({});
  assert.deepEqual(modes,[false,true]);
  assert.equal(engine.settings.fluidVideo,true);
});
