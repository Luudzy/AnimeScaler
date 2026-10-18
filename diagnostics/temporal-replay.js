import { TemporalPresenter } from '../src/engine/temporal-presenter.js';
import { FramePacer } from '../src/core/frame-pacer.js';

// Artificial callback cadence, one request in flight, fixed delivery latency.
// This is a scheduling scenario, not an emulation of NativeEngine transport.
export function cadenceEvents({frameCount,sourceFps=24,maxFps,latencyMs}) {
  if(!Number.isInteger(frameCount)||frameCount<1||frameCount>1200||
    ![sourceFps,maxFps,latencyMs].every(v=>Number.isFinite(v)&&v>0))throw new Error('Invalid cadence scenario');
  const pacer=new FramePacer(),events=[];let completeAt=-Infinity,id=0;
  for(let frameIndex=0;frameIndex<frameCount;frameIndex++) {
    const atMs=frameIndex*1000/sourceFps;
    events.push({type:'present',atMs,frameIndex});
    if(atMs<completeAt||!pacer.take(atMs,maxFps))continue;
    events.push({type:'capture',atMs,frameIndex,id:++id});completeAt=atMs+latencyMs;
    events.push({type:'complete',atMs:completeAt,id});
  }
  const priority={complete:0,present:1,capture:2};
  return events.sort((a,b)=>a.atMs-b.atMs||priority[a.type]-priority[b.type]);
}

// Synchronous: the virtual clock and canvas are always restored before return.
// Actual cut/epoch/age/matching/confidence policy runs; AI pixels and drawing do not.
export function replayTemporal({frames,events,sourceFps=24,onRow}) {
  if(!Array.isArray(frames)||!frames.length||frames.length>1200||
    !frames.every(f=>f instanceof Uint8ClampedArray&&f.length===80*45*4)||
    !Number.isFinite(sourceFps)||sourceFps<=0||!Array.isArray(events)||events.length>10000)
    throw new Error('Invalid replay input');
  let previousAt=-Infinity;
  for(const event of events) {
    if(!['present','capture','complete'].includes(event.type)||!Number.isFinite(event.atMs)||event.atMs<0||event.atMs<previousAt)
      throw new Error('Events must be valid and chronologically ordered');
    if(event.type!=='complete'&&(!Number.isInteger(event.frameIndex)||!frames[event.frameIndex]))throw new Error('Invalid source frame');
    if(event.type!=='present'&&(!Number.isInteger(event.id)||event.id<1))throw new Error('Invalid capture id');
    previousAt=event.atMs;
  }
  const oldCanvas=Object.getOwnPropertyDescriptor(globalThis,'OffscreenCanvas');
  const oldNow=Object.getOwnPropertyDescriptor(performance,'now');
  let now=0,draws=0,uploads=0,sequence=0;
  const restore=(target,key,descriptor)=>{if(descriptor)Object.defineProperty(target,key,descriptor);else delete target[key];};
  try {
    Object.defineProperty(performance,'now',{configurable:true,value:()=>now});
    Object.defineProperty(globalThis,'OffscreenCanvas',{configurable:true,value:class {
      constructor(width,height){this.width=width;this.height=height;}
      getContext(){let image;return {drawImage(frame){image=frame;},getImageData(){return {data:image.pixels.slice()};}};}
    }});
    const sink={resize(){},setFrame(){},uploadAI(){uploads++;},draw(){draws++;}};
    const presenter=new TemporalPresenter({width:1,height:1},null,sink);
    const crop={x:0,y:0,width:80,height:45},size={outputWidth:160,outputHeight:90};
    const references=new Map(),usedIds=new Set(),rows=[];
    for(const event of events) {
      now=event.atMs;const row={...event};
      if(event.type==='complete') {
        if(!references.has(event.id))throw new Error('Completion without a pending capture');
        const reference=references.get(event.id);references.delete(event.id);
        row.accepted=presenter.accept({width:160,height:90},reference);
        if(!row.accepted)row.reason=presenter.metrics.lastRejected;
      } else {
        const frame={pixels:frames[event.frameIndex],timestamp:event.frameIndex*1000000/sourceFps};
        if(event.type==='capture') {
          if(usedIds.has(event.id))throw new Error('Duplicate capture id');
          usedIds.add(event.id);references.set(event.id,presenter.sample(frame,crop,++sequence));
        } else {
          presenter.present(frame,crop,size,++sequence);
          row.metrics={...presenter.metrics};row.epoch=presenter.epoch;
        }
      }
      rows.push(row);onRow?.(row);
    }
    return {rows,draws,uploads,pendingCaptures:references.size,finalMetrics:{...presenter.metrics}};
  } finally {
    restore(globalThis,'OffscreenCanvas',oldCanvas);restore(performance,'now',oldNow);
  }
}
