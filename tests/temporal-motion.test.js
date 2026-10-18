import test from 'node:test';
import assert from 'node:assert/strict';
import * as motion from '../src/engine/temporal-motion.js';

test('small brightness changes cross the old threshold without binary flashing',()=>{
  const before=new Uint8ClampedArray(80*48*4).fill(100);
  const alpha=delta=>motion.motionReuse(before,new Uint8ClampedArray(before.length).fill(100+delta),80,48).mask[(20*80+30)*4+3];
  assert.ok(alpha(5)>alpha(7));
  assert.ok(alpha(7)>0);
  assert.ok(alpha(5)-alpha(7)<80);
  assert.equal(alpha(30),0);
});

function pattern(w=80,h=48) {
  const data=new Uint8ClampedArray(w*h*4);
  for(let y=0;y<h;y++)for(let x=0;x<w;x++) {
    const i=(y*w+x)*4;
    data[i]=(x*17+y*31+x*y*3)%220;
    data[i+1]=(x*7+y*19)%220;data[i+2]=(x*11+y*13)%220;data[i+3]=255;
  }
  return data;
}
function shift(data,w,h,dx,dy) {
  const out=new Uint8ClampedArray(data.length);
  for(let y=0;y<h;y++)for(let x=0;x<w;x++) {
    const xx=x-dx,yy=y-dy;
    if(xx>=0&&xx<w&&yy>=0&&yy<h) out.set(data.subarray((yy*w+xx)*4,(yy*w+xx)*4+4),(y*w+x)*4);
  }
  return out;
}
test('camera translation reuses aligned AI and rejects newly exposed borders',()=>{
  assert.equal(typeof motion.motionReuse,'function');
  const before=pattern(),after=shift(before,80,48,3,-2);
  const result=motion.motionReuse(before,after,80,48);
  const tile=result.tiles.find(t=>t.x===24&&t.y===16);
  assert.equal(tile.dx,-3);assert.equal(tile.dy,2);
  assert.equal(result.mask[(24*80+32)*4+3],255);
  assert.equal(result.mask[(24*80+1)*4+3],0);
});
test('moving object can reuse AI independently from stationary background',()=>{
  const before=pattern(),after=before.slice();
  const moved=shift(before,80,48,2,1);
  for(let y=8;y<40;y++)for(let x=16;x<48;x++) after.set(moved.subarray((y*80+x)*4,(y*80+x)*4+4),(y*80+x)*4);
  const result=motion.motionReuse(before,after,80,48);
  assert.equal(result.tiles.find(t=>t.x===24&&t.y===16).dx,-2);
  assert.equal(result.tiles.find(t=>t.x===64&&t.y===16).dx,0);
  assert.equal(result.mask[(20*80+28)*4+3],255);
});
test('scene cuts and unmatched details do not reuse old AI',()=>{
  const before=pattern(),after=new Uint8ClampedArray(before.length).fill(255);
  const result=motion.motionReuse(before,after,80,48);
  assert.equal(result.tiles.length,0);
  assert.equal(result.mask.some((v,i)=>i%4===3&&v>0),false);
});
test('flat identical frames stay still and invalid references are rejected',()=>{
  const data=new Uint8ClampedArray(80*48*4).fill(100);
  assert.equal(motion.motionReuse(null,data,80,48),null);
  const result=motion.motionReuse(data,data,80,48);
  assert.ok(result.tiles.every(t=>t.dx===0&&t.dy===0));
});

test('fractional movement is refined instead of rejected as an integer tie',()=>{
  const before=pattern(),after=new Uint8ClampedArray(before.length);
  for(let y=0;y<48;y++)for(let x=0;x<78;x++)for(let c=0;c<4;c++)
    after[(y*80+x)*4+c]=(before[(y*80+x+1)*4+c]+before[(y*80+x+2)*4+c])/2;
  const result=motion.motionReuse(before,after,80,48);
  const tile=result.tiles.find(t=>t.x===24&&t.y===16);
  assert.ok(tile,'fractional shift should retain a confident tile');
  assert.equal(tile.dx,1.5);assert.equal(tile.dy,0);
});

test('ambiguous repeating patterns fall back instead of choosing an arbitrary direction',()=>{
  const before=new Uint8ClampedArray(80*48*4),after=before.slice();
  for(let y=0;y<48;y++)for(let x=0;x<80;x++)for(let c=0;c<3;c++) {
    before[(y*80+x)*4+c]=x%2?200:30;
    after[(y*80+x)*4+c]=x%2?30:200;
  }
  assert.equal(motion.motionReuse(before,after,80,48).tiles.length,0);
});

test('fractional repeating patterns reject equally good refined directions',()=>{
  for(const axis of ['x','y'])for(const phase of [.5,.75,.25]) {
    const before=new Uint8ClampedArray(80*45*4),after=before.slice();
    for(let y=0;y<45;y++)for(let x=0;x<80;x++) {
      const i=(y*80+x)*4,periodic=axis==='x'?x:y,other=axis==='x'?y:x;
      before[i]=[30,80,120,60][periodic%4];
      before[i+1]=(other*19)%180;before[i+2]=(other*7)%180;before[i+3]=255;
    }
    for(let y=0;y<44;y++)for(let x=0;x<79;x++)for(let c=0;c<4;c++) {
      const i=(y*80+x)*4+c,next=i+(axis==='x'?4:80*4);
      after[i]=before[i]*(1-phase)+before[next]*phase;
    }
    const result=motion.motionReuse(before,after,80,45);
    assert.equal(result.tiles.some(t=>t.x===24&&t.y===16),false,
      `${axis} phase ${phase}: periodic refined directions must remain ambiguous`);
    assert.equal(result.mask[(20*80+28)*4+3],0);
  }
});

test('nearby refinement seeds converge before deciding an integer match is ambiguous',()=>{
  const before=new Uint8ClampedArray(80*45*4),after=before.slice();
  for(let y=0;y<45;y++)for(let x=0;x<80;x++) {
    const i=(y*80+x)*4;
    before[i]=before[i+2]=40+2*x;before[i+1]=20+4*y;before[i+3]=255;
  }
  for(let y=0;y<45;y++)for(let x=0;x<77;x++)for(let c=0;c<4;c++)
    after[(y*80+x)*4+c]=before[(y*80+x+3)*4+c];
  const result=motion.motionReuse(before,after,80,45),tile=result.tiles.find(t=>t.x===24&&t.y===16);
  assert.ok(tile,'nearby seeds still represent the same exact integer translation');
  assert.equal(tile.dx,3);assert.equal(tile.dy,0);
  assert.equal(result.mask[(20*80+28)*4+3],255);
});
