import test from 'node:test';
import assert from 'node:assert/strict';
import {TemporalPresenter} from '../src/engine/temporal-presenter.js';

function withPresenter(operation){
  const previous=globalThis.OffscreenCanvas;
  const pixels=new Uint8ClampedArray(80*45*4).fill(100);
  globalThis.OffscreenCanvas=class{constructor(w,h){this.width=w;this.height=h;}getContext(){return {drawImage(){},getImageData(){return {data:pixels.slice()};}};}};
  try{
    const presenter=new TemporalPresenter({width:1,height:1},null,{uploadAI(){},setFrame(){}});
    presenter.draw=()=>{};
    operation(presenter);
  }finally{globalThis.OffscreenCanvas=previous;}
}

test('presentation after a reset cannot relabel an in-flight old-scene reference',()=>{
  withPresenter(p=>{
    const frame={timestamp:100},crop={x:0,y:0,width:1,height:1};
    const reference=p.sample(frame,crop);
    p.resetHistory();p.present(frame,crop,{outputWidth:1,outputHeight:1});
    assert.equal(reference.epoch,0,'captured epoch must remain immutable');
    assert.equal(p.accept({width:1,height:1},reference),false);
  });
});

test('repeated source timestamps reuse pixels, not stale capture times or epoch objects',()=>{
  withPresenter(p=>{
    const frame={timestamp:100},crop={x:0,y:0,width:1,height:1};
    const old=p.sample(frame,crop);old.capturedAt-=1000;
    const fresh=p.sample(frame,crop);
    assert.notEqual(old,fresh);
    assert.equal(old.pixels,fresh.pixels,'repeat must avoid another thumbnail readback');
    assert.ok(fresh.capturedAt-old.capturedAt>=1000);
  });
});

test('seeks reset hybrid history even when both frames have the same colors',()=>{
  withPresenter(p=>{
    const crop={x:0,y:0,width:1,height:1},size={outputWidth:1,outputHeight:1};
    p.present({timestamp:1000000},crop,size);
    const reference=p.sample({timestamp:1000000},crop);
    assert.equal(p.accept({width:1,height:1},reference),true);
    p.present({timestamp:500000},crop,size);
    assert.equal(p.reference,null);
    assert.equal(p.accept({width:1,height:1},reference),false);
    const newReference=p.sample({timestamp:500000},crop);
    assert.equal(p.accept({width:1,height:1},newReference),true);
    p.present({timestamp:2000000},crop,size);
    assert.equal(p.reference,null);
  });
});

test('slow inference can be used on a confirmed unchanged source without changing capture age',()=>{
  withPresenter(p=>{
    const crop={x:0,y:0,width:1,height:1};
    p.present({timestamp:100},crop,{outputWidth:1,outputHeight:1});
    const reference=p.sample({timestamp:100},crop);
    reference.capturedAt-=1000;
    const originalAge=reference.capturedAt;
    assert.equal(p.accept({width:1,height:1},reference),true);
    assert.equal(reference.capturedAt,originalAge);
    // An actual change is still expired; a late frame cannot cover motion.
    p.current={...p.current,pixels:p.current.pixels.slice()};
    p.current.pixels[0]++;
    assert.equal(p.accept({width:1,height:1},reference),false);
  });
});

test('coverage diagnostics distinguish missing history from actual AI contribution',()=>{
  const presenter=Object.create(TemporalPresenter.prototype);
  presenter.metrics={presentedFrames:0,lowConfidenceFrames:0};
  presenter.recordCoverage(null,0,350);
  assert.equal(presenter.metrics.lowConfidenceFrames,1);
  assert.equal(presenter.metrics.historyAgeMs,350);
  presenter.recordCoverage({mask:new Uint8ClampedArray([255,255,255,255,255,255,255,0])},.5,40);
  assert.equal(presenter.metrics.aiCoverage,.25);
  assert.equal(presenter.metrics.presentedFrames,2);
  assert.equal(presenter.metrics.lowConfidenceFrames,1);
});

test('coverage exposes motion, confidence recovery and age weight separately',()=>{
  const p=Object.create(TemporalPresenter.prototype);
  p.metrics={presentedFrames:0,lowConfidenceFrames:0};
  const mask=a=>({mask:new Uint8ClampedArray([255,255,255,a,255,255,255,a])});
  p.recordCoverage(mask(128),.5,260,mask(255));
  assert.equal(p.metrics.motionCoverage,1);
  assert.equal(p.metrics.confidenceCoverage,128/255);
  assert.equal(p.metrics.temporalWeight,.5);
  assert.equal(p.metrics.aiCoverage,64/255);
  p.recordCoverage(null,0,350);
  assert.equal(p.metrics.motionCoverage,0);
  assert.equal(p.metrics.confidenceCoverage,0);
  assert.equal(p.metrics.temporalWeight,0);
});

test('inference completion uploads a result without drawing outside presentation clock',()=>{
  const previous=globalThis.OffscreenCanvas;
  globalThis.OffscreenCanvas=class{constructor(w,h){this.width=w;this.height=h;}getContext(){return {};}};
  try {
    let uploads=0,draws=0;
    const presenter=new TemporalPresenter({},null,{uploadAI(){uploads++;}});
    presenter.draw=()=>draws++;
    presenter.accept({width:1,height:1},{capturedAt:performance.now()-20});
    assert.equal(uploads,1);assert.equal(draws,0);
    assert.ok(presenter.reference);
  } finally {globalThis.OffscreenCanvas=previous;}
});

test('old epochs, expired results and older accepted frames never upload',()=>{
  const previous=globalThis.OffscreenCanvas;
  globalThis.OffscreenCanvas=class{constructor(w,h){this.width=w;this.height=h;}getContext(){return {};}};
  try {
    let uploads=0;
    const presenter=new TemporalPresenter({},null,{uploadAI(){uploads++;}});
    const image={width:1,height:1},now=performance.now();
    assert.equal(presenter.accept(image,{epoch:0,capturedAt:now-10}),true);
    assert.equal(presenter.accept(image,{epoch:0,capturedAt:now-20}),false);
    presenter.resetHistory();
    assert.equal(presenter.accept(image,{epoch:0,capturedAt:now}),false);
    assert.equal(presenter.accept(image,{epoch:1,capturedAt:now-400}),false);
    assert.equal(uploads,1);
    assert.equal(presenter.metrics.rejectedResults,3);
  } finally {globalThis.OffscreenCanvas=previous;}
});

test('repeated pixels refresh the source reference without reuploading accepted AI',()=>{
  const previous=globalThis.OffscreenCanvas;
  globalThis.OffscreenCanvas=class{constructor(w,h){this.width=w;this.height=h;}getContext(){return {};}};
  try {
    let uploads=0;
    const p=new TemporalPresenter({},null,{uploadAI(){uploads++;}});
    const image={width:1,height:1};
    p.accept(image,{capturedAt:performance.now()-20});
    const next={capturedAt:performance.now()};
    p.accept(image,next,true);
    assert.equal(uploads,1);assert.equal(p.reference,next);
    // A response rejected earlier was never uploaded; a repeat must upload it.
    const other={width:1,height:1};
    p.accept(other,{capturedAt:performance.now()-400});
    p.accept(other,{capturedAt:performance.now()},true);
    assert.equal(uploads,2);
  } finally {globalThis.OffscreenCanvas=previous;}
});

test('capture before presentation belongs to the new seek epoch',()=>{
  withPresenter(p=>{
    const crop={x:0,y:0,width:1,height:1},size={outputWidth:1,outputHeight:1};
    p.present({timestamp:1000000},crop,size);
    const old=p.sample({timestamp:1000000},crop);
    const next=p.sample({timestamp:500000},crop);
    p.present({timestamp:500000},crop,size);
    assert.equal(p.accept({width:1,height:1},old),false);
    assert.equal(p.accept({width:1,height:1},next),true,'capture already contains the new scene');
  });
});

test('capture geometry change invalidates old results before the next presentation',()=>{
  withPresenter(p=>{
    const size={outputWidth:1,outputHeight:1},crop={x:0,y:0,width:1,height:1};
    p.present({timestamp:100},crop,size);
    const old=p.sample({timestamp:100},crop);
    const next=p.sample({timestamp:200},{...crop,x:1});
    assert.equal(p.accept({width:1,height:1},old),false,'obsolete crop must not upload');
    assert.equal(p.accept({width:1,height:1},next),true,'fresh crop can upload before display');
    p.present({timestamp:200},{...crop,x:1},size);
    assert.equal(p.reference,next);
  });
});
