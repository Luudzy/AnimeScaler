import test from 'node:test';
import assert from 'node:assert/strict';
import { motionReuse } from '../src/engine/temporal-motion.js';
import { motionReuse as baseline } from '../docs/motion-evidence-20261011/candidate-temporal-motion.js';

function pattern(width,height,seed=1) {
  const out=new Uint8ClampedArray(width*height*4);
  for(let i=0;i<out.length;i++) {
    seed=(Math.imul(seed,1664525)+1013904223)>>>0;
    out[i]=i%4===3?255:seed>>>26;
  }
  return out;
}
function warp(reference,width,height,dx,dy) {
  const out=new Uint8ClampedArray(reference.length);
  for(let y=0;y<height;y++)for(let x=0;x<width;x++) {
    const xx=x+dx,yy=y+dy;
    if(xx<0||yy<0||xx>width-1||yy>height-1)continue;
    const x0=Math.floor(xx),y0=Math.floor(yy),x1=Math.min(width-1,x0+1),y1=Math.min(height-1,y0+1),fx=xx-x0,fy=yy-y0;
    for(let c=0;c<4;c++)out[(y*width+x)*4+c]=
      (reference[(y0*width+x0)*4+c]*(1-fx)+reference[(y0*width+x1)*4+c]*fx)*(1-fy)
      +(reference[(y1*width+x0)*4+c]*(1-fx)+reference[(y1*width+x1)*4+c]*fx)*fy;
  }
  return out;
}
test('motion search avoids repeated reference reads while preserving its complete result',()=>{
  const width=80,height=45,reference=pattern(width,height),current=warp(reference,width,height,1.5,.5);
  function run(fn) {
    let reads=0;
    const observed=new Proxy(reference,{get(target,key){
      if(typeof key==='string'&&/^\d+$/.test(key))reads++;
      return Reflect.get(target,key,target);
    }});
    return {result:fn(observed,current,width,height),get reads(){return reads;}};
  }
  const before=run(baseline),after=run(motionReuse);
  assert.deepEqual(after.result,before.result);
  assert.ok(after.reads<before.reads*.9,`reference reads ${after.reads} vs baseline ${before.reads}`);
});

test('score reuse preserves masks and tiles across offsets, edges, content changes and calls',()=>{
  for(const [width,height] of [[80,45],[17,11],[1,1]]) {
    const reference=pattern(width,height,54321);
    for(const [dx,dy] of [[0,0],[3,-2],[-4.75,.25],[4.75,-.75],[1.5,.5],[-.25,2.75]]) {
      const current=warp(reference,width,height,dx,dy);
      assert.deepEqual(motionReuse(reference,current,width,height),baseline(reference,current,width,height),`${width}x${height}, ${dx},${dy}`);
      const changed=current.slice();
      for(let i=0;i<changed.length;i+=28)changed[i]=255-changed[i];
      assert.deepEqual(motionReuse(reference,changed,width,height),baseline(reference,changed,width,height));
    }
    const cut=new Uint8ClampedArray(reference.length).fill(255);
    assert.deepEqual(motionReuse(reference,cut,width,height),baseline(reference,cut,width,height));
  }
});
