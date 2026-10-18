import test from 'node:test';
import assert from 'node:assert/strict';
import {TemporalPresenter} from '../src/engine/temporal-presenter.js';
import {temporalStrength} from '../src/engine/temporal-budget.js';

// Actual sampling, cut detection, matching and confidence history run in CPU.
// Only canvas pixel access and the final GPU upload/draw are replaced.
function sequence(t) {
  let now=0,uploads=0;
  const previous=globalThis.OffscreenCanvas;
  globalThis.OffscreenCanvas=class {
    constructor(width,height){this.width=width;this.height=height;}
    getContext(){let image;return {
      drawImage(frame){image=frame;},
      getImageData(){return {data:image.pixels.slice()};},
    };}
  };
  t.after(()=>{globalThis.OffscreenCanvas=previous;});
  t.mock.method(performance,'now',()=>now);
  const draws=[];
  const gpu={resize(){},setFrame(){},uploadAI(){uploads++;},draw(reuse,direct,strength){draws.push({reuse,direct,strength});}};
  const p=new TemporalPresenter({width:1,height:1},null,gpu);
  const crop={x:0,y:0,width:80,height:45},size={outputWidth:160,outputHeight:90};
  const frame=(value,time)=>({timestamp:time,pixels:new Uint8ClampedArray(80*45*4).fill(value)});
  const image={width:160,height:90};
  return {p,crop,size,frame,image,draws,setTime(value){now=value;},uploads:()=>uploads};
}

test('moving content obeys the source-age ceiling even when delivery is delayed',t=>{
  const s=sequence(t),{p,crop,size,frame,image}=s;
  p.present(frame(100,0),crop,size);
  const reference=p.sample(frame(100,0),crop);
  s.setTime(40);assert.equal(p.accept(image,reference),true);
  for(const age of [40,180,250,299,300,400]) {
    s.setTime(age);p.present(frame(102,Math.round(age*1000)),crop,size);
    assert.equal(p.metrics.temporalWeight,temporalStrength(age,40));
    assert.equal(p.metrics.historyAgeMs,age);
    if(age>=300)assert.equal(p.metrics.aiCoverage,0);
  }
});

test('paused identical source retains AI while recording its real age',t=>{
  const s=sequence(t),{p,crop,size,frame,image}=s;
  p.present(frame(100,0),crop,size);
  const reference=p.sample(frame(100,0),crop);
  s.setTime(40);p.accept(image,reference);
  s.setTime(2000);p.present(frame(100,0),crop,size);
  assert.equal(p.metrics.aiCoverage,1);
  assert.equal(p.metrics.historyAgeMs,2000);
  assert.equal(p.metrics.unchangedSource,true);
});

test('new-scene inference before presentation survives once, while old results never upload',t=>{
  const s=sequence(t),{p,crop,size,frame,image}=s;
  p.present(frame(20,0),crop,size);
  const old=p.sample(frame(20,0),crop);
  s.setTime(20);const fresh=p.sample(frame(230,40000),crop);
  assert.equal(p.metrics.sceneCuts,1);
  assert.equal(p.accept(image,old),false);
  assert.equal(s.uploads(),0);
  s.setTime(40);assert.equal(p.accept(image,fresh),true);
  assert.equal(s.draws.length,1,'completion cannot add a draw');
  p.present(frame(230,40000),crop,size);
  assert.equal(p.metrics.sceneCuts,1,'presentation must not count the same cut twice');
  assert.equal(p.metrics.aiCoverage,1);
  assert.equal(fresh.epoch,p.epoch);
  assert.equal(p.accept(image,old),false);
  assert.equal(s.uploads(),1);
});

test('alternating matches stay suppressed and a quiet source eventually recovers',t=>{
  const s=sequence(t),{p,crop,size,frame,image}=s;
  p.present(frame(100,0),crop,size);
  const reference=p.sample(frame(100,0),crop);p.accept(image,reference);
  for(let n=1;n<=20;n++) {
    s.setTime(n*33);p.present(frame(n%2?120:100,n*33000),crop,size);
    assert.equal(p.metrics.aiCoverage,0,`invalid history flashed on frame ${n}`);
  }
  for(let n=21;n<=40;n++) {
    s.setTime(n*33);p.present(frame(100,n*33000),crop,size);
  }
  assert.equal(p.metrics.aiCoverage,1);
  assert.equal(p.metrics.motionCoverage,1);
});

test('resize discards the earlier-size response and accepts a matching new capture',t=>{
  const s=sequence(t),{p,crop,size,frame,image}=s;
  p.present(frame(100,0),crop,size);
  const old=p.sample(frame(100,0),crop);
  p.present(frame(100,0),crop,{outputWidth:80,outputHeight:45});
  assert.equal(p.accept(image,old),false);
  const fresh=p.sample(frame(100,0),crop);
  assert.equal(p.accept({width:80,height:45},fresh),true);
  p.present(frame(100,0),crop,{outputWidth:80,outputHeight:45});
  assert.equal(p.metrics.aiCoverage,1);
});

for(const timestamps of [true,false])test(`delayed old capture cannot undo a fresh cut (timestamps=${timestamps})`,t=>{
  const {p,crop,size,frame,image,draws}=sequence(t);
  const source=(value,time)=>{const result=frame(value,time);if(!timestamps)delete result.timestamp;return result;};
  p.present(source(20,1000000),crop,size,1);
  const fresh=p.sample(source(230,1040000),crop,3);
  const epoch=p.epoch,drawCount=draws.length;
  assert.equal(p.present(source(20,1000000),crop,{outputWidth:80,outputHeight:45},2),false);
  assert.equal(p.base.width,160,'obsolete presentation cannot resize the output');
  assert.equal(p.sample(source(20,1000000),crop,2),null);
  assert.equal(p.epoch,epoch);assert.equal(draws.length,drawCount);
  assert.equal(p.accept(image,fresh),true);
  p.present(source(230,1040000),crop,size,4);
  assert.equal(p.metrics.sceneCuts,1);assert.equal(p.metrics.aiCoverage,1);
  // Media time may move backwards in a genuine later seek. Capture order wins.
  p.present(source(20,500000),crop,size,5);
  assert.ok(p.epoch>epoch);assert.equal(p.accept(image,fresh),false);
});
