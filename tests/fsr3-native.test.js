import test from 'node:test';
import assert from 'node:assert/strict';
import {normalizeSettings} from '../src/core/settings.js';
import {validateNativeService} from '../src/engine/native-service-contract.js';
import {NativeEngine} from '../src/engine/native-engine.js';

test('FSR3 requires its own D3D12 backend; unrelated models cannot select it',()=>{
  const settings=normalizeSettings({model:'fsr3-native',backend:'rocm'});
  assert.equal(settings.model,'fsr3-native');assert.equal(settings.backend,'rocm');
  assert.equal(settings.fluidVideo,false);
  assert.equal(normalizeSettings({model:'adore',backend:'d3d12'}).backend,'d3d12');
});
test('FSR3 handshake verifies backend, actual official provider and native precision',()=>{
  const result={protocol:1,model:'fsr3-native',backend:'d3d12',precision:'native',provider:'3.1.5',
    models:['fsr3-native'],revisions:{'fsr3-native':'a'.repeat(64)},metadata:{'fsr3-native':{status:'experimental'}}};
  validateNativeService(result,{model:'fsr3-native',backend:'d3d12'});
  for(const patch of [{backend:'rocm'},{provider:'4.1.1'},{provider:'2.3.4'},{precision:'fp16'},{model:'adore'}]) {
    assert.throws(()=>validateNativeService({...result,...patch},{model:'fsr3-native',backend:'d3d12'}));
  }
  validateNativeService({protocol:1,model:'adore',backend:'rocm',precision:'fp16',models:['adore'],
    revisions:{adore:'b'.repeat(64)},metadata:{adore:{id:'adore',status:'external-baseline'}}},{model:'adore',backend:'rocm'});
});
test('native handshake requires advertised model revision and matching status metadata',()=>{
  const result={protocol:1,model:'adore',backend:'rocm',precision:'fp16',models:['adore'],
    revisions:{adore:'b'.repeat(64)},metadata:{adore:{id:'adore',status:'external-baseline'}}};
  validateNativeService(result,{model:'adore',backend:'rocm'});
  for(const patch of [
    {models:[]},
    {revisions:{}},
    {revisions:{adore:''}},
    {metadata:{}},
    {metadata:{adore:{id:'other',status:'external-baseline'}}},
    {metadata:{adore:{id:'adore',status:''}}},
  ]) assert.throws(()=>validateNativeService({...result,...patch},{model:'adore',backend:'rocm'}));
});
test('standalone OpenCL handshake requires native precision and its provider metadata',()=>{
  const result={protocol:1,model:'anime4kcpp-acnet-f8b4',backend:'opencl',precision:'native',
    models:['anime4kcpp-acnet-f8b4'],revisions:{'anime4kcpp-acnet-f8b4':'anime4kcpp-3.2.0-opencl'},
    metadata:{'anime4kcpp-acnet-f8b4':{id:'anime4kcpp-acnet-f8b4',status:'experimental-opencl',provider:'Anime4KCPP OpenCL'}}};
  validateNativeService(result,{model:'anime4kcpp-acnet-f8b4',backend:'opencl'});
  assert.throws(()=>validateNativeService({...result,precision:'fp16'},{model:'anime4kcpp-acnet-f8b4',backend:'opencl'}));
  assert.throws(()=>validateNativeService({...result,metadata:{'anime4kcpp-acnet-f8b4':{...result.metadata['anime4kcpp-acnet-f8b4'],provider:'ROCm'}}},{model:'anime4kcpp-acnet-f8b4',backend:'opencl'}));
});
test('FSR3 quality controls preserve output and never pre-enlarge a 720p source to 1080p',()=>{
  const engine=new NativeEngine();engine.nativeInfo={backend:'d3d12'};
  engine.settings=normalizeSettings({model:'fsr3-native',preset:'4k',fsrQuality:'native-aa'});
  engine.configure({videoWidth:1280,videoHeight:720},0,null);
  assert.equal(engine.currentSize.inputWidth,1280);assert.equal(engine.currentSize.inputHeight,720);
  assert.equal(engine.currentSize.outputWidth,3840);assert.equal(engine.currentSize.outputHeight,2160);
  assert.equal(engine.currentSize.modelOutputWidth,3840);
  engine.settings=normalizeSettings({model:'fsr3-native',preset:'1080p',fsrQuality:'performance'});
  engine.configure({videoWidth:1920,videoHeight:1080},0,null);
  assert.equal(engine.currentSize.inputWidth,960);assert.equal(engine.currentSize.outputWidth,1920);
});
