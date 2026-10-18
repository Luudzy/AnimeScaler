import test from 'node:test';
import assert from 'node:assert/strict';
import {ProfileSamples,restoreProfile,profileKey} from '../src/core/performance-profile.js';
test('optimized RealESR starts calibration separately from its old latency profile',()=>{
  const settings={model:'realesr-anime',preset:'4k',maxFps:30};
  const original={backend:'rocm',gpu:'RX5500XT',revisions:{'realesr-anime':'same-weights'}};
  const optimized={...original,metadata:{'realesr-anime':{capture_graph_max_pixels:640*360}}};
  assert.notEqual(profileKey(original,settings,1920,1080),profileKey(optimized,settings,1920,1080));
  assert.equal(profileKey(original,settings,1920,1080),profileKey({...optimized,metadata:{'realesr-anime':{capture_graph_max_pixels:0}}},settings,1920,1080));
  const trtOriginal={...original,backend:'tensorrt'},trtOptimized={...optimized,backend:'tensorrt'};
  assert.equal(profileKey(trtOriginal,settings,1920,1080),profileKey(trtOptimized,settings,1920,1080));
});
test('unrelated models retain their existing calibration keys',()=>{
  const settings={model:'adore-lite-v3',preset:'1080p',maxFps:30};
  const info={backend:'rocm',gpu:'RX5500XT'};
  assert.equal(profileKey(info,settings,1280,720),profileKey({...info,metadata:{'realesr-anime':{capture_graph_max_pixels:640*360}}},settings,1280,720));
});
test('calibration requires sustained headroom, isolates changed rungs and expires',()=>{
  const samples=new ProfileSamples();
  for(let i=0;i<29;i++)assert.equal(samples.record(10,1,30),null);
  assert.equal(samples.record(10,1,30).level,1);
  for(let i=0;i<30;i++)assert.equal(samples.record(40,0,30),null);
  assert.equal(restoreProfile({level:2,savedAt:1000},2000),2);
  assert.equal(restoreProfile({level:2,savedAt:1000},8*86400000),0);
  assert.equal(restoreProfile({level:NaN,savedAt:1000},2000),0);
  assert.notEqual(profileKey({gpu:'a'}, {model:'adore'},640,360),profileKey({gpu:'b'}, {model:'adore'},640,360));
});
