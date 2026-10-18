// Local diagnostic only: exercise the production worker/compositor without
// requiring a running native service. Server integration has separate tests.
class FixtureSocket {
  static OPEN=1;
  constructor(){this.readyState=1;this.frames=0;queueMicrotask(()=>this.onopen?.());}
  close(){this.readyState=3;}
  send(data){
    if(typeof data==='string') {
      queueMicrotask(()=>this.onmessage({data:JSON.stringify({protocol:1,model:'adore-lite',precision:'fp16',backend:'rocm'})}));
      return;
    }
    const header=new DataView(data),id=header.getUint32(4,true),w=header.getUint32(16,true),h=header.getUint32(20,true);
    this.frames++;
    const repeated=this.frames===2||this.frames===5;
    const buffer=new ArrayBuffer(24+(repeated?0:w*h*3)),view=new DataView(buffer);
    [repeated?0x34435341:0x33435341,id,w,h,w,h].forEach((v,i)=>view.setUint32(i*4,v,true));
    if(!repeated)new Uint8Array(buffer,24).fill(this.frames===3?200:120);
    queueMicrotask(()=>{
      this.onmessage({data:JSON.stringify({processingMs:1,warmup:false,reused:repeated})});
      this.onmessage({data:buffer});
    });
  }
}
self.WebSocket=FixtureSocket;
if(new URL(self.location.href).searchParams.has('canvas2d'))Object.defineProperty(self.navigator,'gpu',{value:undefined});
await import('../src/engine/native-worker.js');
self.postMessage({fixtureReady:true});
