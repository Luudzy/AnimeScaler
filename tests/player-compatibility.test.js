import test from 'node:test';
import assert from 'node:assert/strict';
import { findBestVideo } from '../src/content/video-finder.js';
import { findCaptureTargetElement } from '../src/content/capture-target.js';
import { AnimeScalerController } from '../src/content/controller.js';

function root(videos = [], children = [], frames = []) {
  return { querySelectorAll: (selector) => selector === 'video' ? videos : selector === '*' ? children : frames };
}
function video(extra = {}) {
  return { isConnected: true, videoWidth: 1280, videoHeight: 720, readyState: 4,
    paused: false, ended: false, style: { opacity: '1' },
    getBoundingClientRect: () => ({ left: 0, top: 0, right: 800, bottom: 450, width: 800, height: 450 }),
    requestVideoFrameCallback: () => 1, cancelVideoFrameCallback() {}, ...extra };
}
function environment(t) {
  for (const [key, value] of Object.entries({
    window: { innerWidth: 1920, innerHeight: 1080, removeEventListener() {} },
    innerWidth:1920,innerHeight:1080,
    getComputedStyle: (element) => element.style || {},
    MutationObserver: class { disconnect() {} }, document: root(),
  })) {
    const old = globalThis[key]; globalThis[key] = value;
    t.after(() => { globalThis[key] = old; });
  }
}

test('viewer pause, seek and resume keep controlling the source selected by capture geometry',async t=>{
  environment(t);
  const preview=video({paused:true,currentTime:5,duration:100,pause(){this.paused=true;},async play(){this.paused=false;}});
  const main=video({paused:false,currentTime:30,duration:100,pause(){this.paused=true;},async play(){this.paused=false;}});
  globalThis.document=root([preview,main]);
  const controller=new AnimeScalerController();
  controller.getCaptureGeometry({frameId:0},'viewer-a');
  await controller.playerCommand({type:'toggle'},'viewer-a');
  assert.equal(main.paused,true);
  await controller.playerCommand({type:'seek',seconds:10},'viewer-a');
  assert.equal(main.currentTime,40,'seek must stay on the paused main video');
  assert.equal(preview.currentTime,5);
  await controller.playerCommand({type:'toggle'},'viewer-a');
  assert.equal(main.paused,false);assert.equal(preview.paused,true);
});

test('paused source retains its capture geometry until removal or a new viewer session',t=>{
  environment(t);
  const preview=video({paused:true}),main=video({paused:false});
  main.getBoundingClientRect=()=>({left:100,top:50,right:900,bottom:500,width:800,height:450});
  globalThis.document=root([preview,main]);
  const controller=new AnimeScalerController();
  assert.equal(controller.getCaptureGeometry({frameId:0},'viewer-a').rect.left,100);
  main.paused=true;preview.paused=false;
  assert.equal(controller.getCaptureGeometry({frameId:0},'viewer-a').rect.left,100);
  assert.equal(controller.getCaptureGeometry({frameId:0},'viewer-b').rect.left,0);
  preview.isConnected=false;
  globalThis.document=root([main]);
  assert.equal(controller.getCaptureGeometry({frameId:0},'viewer-b').rect.left,100);
});

test('a hidden bound source is unavailable instead of switching capture to another video',t=>{
  environment(t);
  const preview=video({paused:true}),main=video({paused:false});
  globalThis.document=root([preview,main]);
  const controller=new AnimeScalerController();
  assert.ok(controller.getCaptureGeometry({frameId:0},'viewer-a'));
  main.style.visibility='hidden';
  assert.equal(controller.getCaptureGeometry({frameId:0},'viewer-a'),null);
});

test('paused seek refreshes once even inside the frame cap and ignores repeated callbacks',async t=>{
  environment(t);
  const controller=new AnimeScalerController(),target=video({currentTime:40});
  controller.state='running';controller.video=controller.displayTarget=target;
  controller.settings={...controller.settings,maxFps:12,adaptive:false};
  const rendered=[];
  controller.engine={needsReconfigure:()=>false,async processFrame(v){rendered.push(v.currentTime);return {};},destroy(){}};
  await controller.onVideoFrame(0,{mediaTime:40});
  target.paused=true;target.currentTime=50;
  await controller.onVideoFrame(10,{mediaTime:49.958});
  await controller.onVideoFrame(16,{mediaTime:49.958});
  await controller.onVideoFrame(32,{mediaTime:50});
  assert.deepEqual(rendered,[40,50]);
  await controller.stop();
});

test('source frame rate observes callbacks even when processing is capped',async t=>{
  environment(t);
  const target=video({currentTime:0});
  const controller=new AnimeScalerController();
  controller.state='running';controller.video=controller.displayTarget=target;
  controller.settings={...controller.settings,maxFps:12,adaptive:false};
  controller.engine={needsReconfigure:()=>false,async processFrame(){return {totalMs:5,inferenceMs:4};},destroy(){}};
  const source=[];
  controller.diagnostics.sourceFrame=(mediaTime,now)=>source.push([mediaTime,now]);
  await controller.onVideoFrame(0,{mediaTime:0});
  target.currentTime=1/30;
  await controller.onVideoFrame(33,{mediaTime:1/30});
  assert.deepEqual(source,[[0,0],[1/30,33]]);
  assert.equal(controller.processedFps,0);
  await controller.stop();
});

test('an initial paused frame is processed only once in animation-frame fallback',async t=>{
  environment(t);
  const controller=new AnimeScalerController(),target=video({paused:true,currentTime:5});
  controller.state='running';controller.video=controller.displayTarget=target;
  controller.settings={...controller.settings,adaptive:false};
  let rendered=0;
  controller.engine={needsReconfigure:()=>false,async processFrame(){rendered++;return {};},destroy(){}};
  for(const now of [0,16,32,1000])await controller.onVideoFrame(now,{mediaTime:5});
  assert.equal(rendered,1);
  await controller.stop();
});

test('paused seeks during in-flight processing refresh only the latest position',async t=>{
  environment(t);
  const controller=new AnimeScalerController(),target=video({paused:true,currentTime:40});
  controller.state='running';controller.video=controller.displayTarget=target;
  controller.settings={...controller.settings,adaptive:false};
  const rendered=[];let finish;
  controller.engine={needsReconfigure:()=>false,async processFrame(v){
    rendered.push(v.currentTime);if(rendered.length===1)await new Promise(resolve=>{finish=resolve;});return {};
  },destroy(){}};
  const first=controller.onVideoFrame(0,{mediaTime:40});
  target.currentTime=50;await controller.onVideoFrame(10,{mediaTime:50});
  target.currentTime=60;await controller.onVideoFrame(16,{mediaTime:60});
  finish();await first;await new Promise(resolve=>setImmediate(resolve));
  assert.deepEqual(rendered,[40,60]);
  await controller.stop();
});

test('stopping cancels the deferred paused-seek refresh',async t=>{
  environment(t);
  const controller=new AnimeScalerController(),target=video({paused:true,currentTime:40});
  controller.state='running';controller.video=controller.displayTarget=target;
  controller.settings={...controller.settings,adaptive:false};
  let finish,rendered=0;
  controller.engine={needsReconfigure:()=>false,async processFrame(){rendered++;await new Promise(resolve=>{finish=resolve;});return {};},destroy(){}};
  const first=controller.onVideoFrame(0,{mediaTime:40});
  target.currentTime=50;await controller.onVideoFrame(10,{mediaTime:50});
  await controller.stop();finish();await first;await new Promise(resolve=>setImmediate(resolve));
  assert.equal(rendered,1);assert.equal(controller.state,'stopped');
});

test('finds video in nested open shadow roots and ignores hidden host', (t) => {
  environment(t);
  const target = video();
  const host = { style: {}, shadowRoot: root([target]) };
  target.getRootNode = () => ({ host });
  const page = root([], [{ shadowRoot: root([], [host]) }]);
  assert.equal(findBestVideo(page), target);
  host.style.display = 'none';
  assert.equal(findBestVideo(page), null);
});

test('prefers playing video to equally sized paused preview', (t) => {
  environment(t);
  const playing = video();
  assert.equal(findBestVideo(root([video({ paused: true }), playing])), playing);
});

test('finds external iframe inside shadow root for capture', (t) => {
  environment(t);
  const target = video({ src: 'https://player.example/embed' });
  const page = root([], [{ shadowRoot: root([], [], [target]) }]);
  assert.equal(findCaptureTargetElement(page, { frameId: 2, url: target.src,
    viewport: { width: 1920, height: 1080 } }), target);
});

test('waits for replacement metadata then reconnects without rebuilding engine', async (t) => {
  environment(t);
  const controller = new AnimeScalerController();
  const originalEngine = { destroy() {} };
  controller.engine = originalEngine;
  controller.state = 'running';
  controller.video = controller.displayTarget = video({ isConnected: false });
  controller.ensureVideoStillExists();
  assert.equal(controller.state, 'running');
  const replacement = video({ videoWidth: 0 });
  globalThis.document = root([replacement]);
  controller.ensureVideoStillExists();
  assert.notEqual(controller.video, replacement);
  replacement.videoWidth = 1280;
  controller.ensureVideoStillExists();
  assert.equal(controller.video, replacement);
  assert.equal(controller.displayTarget, replacement);
  assert.equal(controller.engine, originalEngine);
  await controller.stop();
  controller.ensureVideoStillExists();
  assert.equal(controller.state, 'stopped');
});

test('does not rebind while a GPU frame is in progress', (t) => {
  environment(t);
  const controller = new AnimeScalerController();
  controller.state = 'running'; controller.busy = true;
  const previous = video({ isConnected: false });
  controller.video = controller.displayTarget = previous;
  globalThis.document = root([video()]);
  controller.ensureVideoStillExists();
  assert.equal(controller.state, 'running');
  assert.equal(controller.video, previous);
});

test('resumes callbacks when the site reinserts the same video', async (t) => {
  environment(t);
  const controller = new AnimeScalerController();
  controller.state = 'running';
  const target = video({ isConnected: false });
  controller.video = controller.displayTarget = target;
  await controller.onVideoFrame(0);
  assert.equal(controller.waitingForPlayer, true);
  target.isConnected = true;
  globalThis.document = root([target]);
  controller.ensureVideoStillExists();
  assert.equal(controller.waitingForPlayer, false);
  assert.notEqual(controller.frameHandle, null);
  await controller.stop();
});

test('frame cap skips GPU inference between allowed frames', async (t) => {
  environment(t);
  const controller = new AnimeScalerController();
  controller.state = 'running';
  controller.settings = { ...controller.settings, maxFps: 12, adaptive: false };
  controller.video = controller.displayTarget = video();
  let frames = 0;
  controller.engine = { needsReconfigure: () => false, processFrame: async () => { frames++; return {}; }, destroy() {} };
  for (const now of [0, 16, 32, 48, 64, 80, 96]) await controller.onVideoFrame(now);
  assert.equal(frames, 2);
  await controller.stop();
});

test('30 FPS cap preserves cadence with slightly early video callbacks', async (t) => {
  environment(t);
  const controller = new AnimeScalerController();
  controller.state = 'running';
  controller.settings = {...controller.settings, maxFps:30, adaptive:false};
  controller.video = controller.displayTarget = video();
  let frames = 0;
  controller.engine = {needsReconfigure:()=>false, processFrame:async()=>{frames++;return {};}, destroy(){}};
  // 60 Hz callbacks with sub-millisecond jitter at each 30 Hz boundary.
  for (let i=0;i<60;i++) await controller.onVideoFrame(i * 1000/60 - (i % 2 === 0 ? .2 : 0));
  assert.ok(frames >= 29 && frames <= 31, `expected about 30 frames, got ${frames}`);
  await controller.stop();
});
