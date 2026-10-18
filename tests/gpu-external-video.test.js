import test from 'node:test';
import assert from 'node:assert/strict';
import {GpuPrePost} from '../src/engine/gpu-prepost.js';
import {NativeGpuRenderer} from '../src/engine/native-gpu-renderer.js';

function fixture(t){
  const old=globalThis.VideoFrame;
  globalThis.VideoFrame=class{constructor(){this.displayWidth=1920;this.displayHeight=1080;}};
  t.after(()=>{globalThis.VideoFrame=old;});
  const gpu=Object.create(GpuPrePost.prototype),calls={copies:0,imports:0,writes:[],pipelines:[]};
  gpu.config={sourceWidth:640,sourceHeight:360,inputWidth:320,inputHeight:180};
  gpu.workgroup=[8,8];gpu.sourceTexture={createView:()=>({})};
  gpu.prePipeline={getBindGroupLayout:()=>({})};
  gpu.externalPrePipeline={getBindGroupLayout:()=>({})};
  gpu.device={
    importExternalTexture(){calls.imports++;return {importId:calls.imports};},
    createBindGroup:descriptor=>descriptor,
    queue:{copyExternalImageToTexture(){calls.copies++;},writeBuffer(_b,_o,data){calls.writes.push([...data]);},submit(){}},
    createCommandEncoder:()=>({beginComputePass:()=>({setPipeline(p){calls.pipelines.push(p);},setBindGroup(){},dispatchWorkgroups(){},end(){}}),finish(){}}),
  };
  return {gpu,calls,video:new VideoFrame()};
}

test('decoded video is imported directly, with crop bounds and no full-frame copy',t=>{
  const {gpu,calls,video}=fixture(t);
  gpu.uploadVideoFrame(video,{x:120,y:80,width:640,height:360});gpu.preprocess();
  assert.equal(calls.copies,0);assert.equal(calls.imports,1);
  assert.equal(gpu.captureMethod,'external-texture');
  assert.equal(calls.pipelines[0],gpu.externalPrePipeline);
  const values=calls.writes[0];
  assert.ok(Math.abs(values[0]-120/1920)<1e-6);
  assert.ok(Math.abs(values[4]-120.5/1920)<1e-6,'crop must clamp at its own edge');
});

test('a new frame imports a new texture and does not reuse an expired bind group',t=>{
  const {gpu,calls,video}=fixture(t);
  gpu.uploadVideoFrame(video);const first=gpu.externalPreBindGroup;
  gpu.preprocess();gpu.uploadVideoFrame(video);gpu.preprocess();
  assert.equal(calls.imports,2);assert.notEqual(gpu.externalPreBindGroup,first);
});

test('unsupported video import falls back to the existing crop copy',t=>{
  const {gpu,calls,video}=fixture(t);
  gpu.device.importExternalTexture=()=>{throw new DOMException('unsupported','NotSupportedError');};
  gpu.uploadVideoFrame(video,{x:120,y:80,width:640,height:360});gpu.preprocess();
  assert.equal(calls.copies,1);assert.equal(gpu.captureMethod,'texture-copy');
  assert.equal(calls.pipelines[0],gpu.prePipeline);
});

test('security errors are preserved instead of silently falling back',t=>{
  const {gpu,calls,video}=fixture(t);
  gpu.device.importExternalTexture=()=>{throw new DOMException('cross-origin','SecurityError');};
  assert.throws(()=>gpu.uploadVideoFrame(video),{name:'SecurityError'});
  assert.equal(calls.copies,0);
});

test('fluid compositor imports decoded frames and clears their external binding after submit',t=>{
  const {calls,video,gpu}=fixture(t);
  const renderer=Object.create(NativeGpuRenderer.prototype);
  renderer.device=gpu.device;renderer.externalPipeline={getBindGroupLayout:()=>({})};
  renderer.paramData=new Float32Array(12);renderer.currentWidth=1;renderer.currentHeight=1;
  renderer.current={createView:()=>({}),destroy(){}};
  renderer.ai=renderer.mask={createView:()=>({})};
  renderer.context={getCurrentTexture:()=>({createView:()=>({})})};
  renderer.device.createCommandEncoder=()=>({beginRenderPass:()=>({setPipeline(p){calls.pipelines.push(p);},setBindGroup(){},draw(){},end(){}}),finish(){}});
  renderer.setFrame(video,{x:120,y:80,width:640,height:360});
  renderer.draw();
  assert.equal(calls.imports,1);assert.equal(calls.copies,0);
  assert.equal(renderer.sourceMethod,'external-texture');
  assert.equal(calls.pipelines[0],renderer.externalPipeline);
  assert.equal(renderer.externalCurrent,null,'closed frames cannot be rebound later');
});
