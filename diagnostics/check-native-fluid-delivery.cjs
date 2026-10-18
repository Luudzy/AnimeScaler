const fs=require('node:fs');
const {writeDiagnosticReport}=require('./report-output.cjs');
const {chromium,executablePath}=require('./browser-runtime.cjs').browserRuntime();
const baseline=process.argv.includes('--baseline');
// Simulated inference avoids loading the user's HIP backend. The real engine,
// transferable VideoFrames, worker, RGB unpack and WebGPU renderer are tested.
const fakeSocket=`
globalThis.WebSocket=class {
 static OPEN=1;
 constructor(url){this.url=url;this.readyState=1;queueMicrotask(()=>this.onopen());}
 send(value){
  if(typeof value==='string'){
   this.onmessage({data:JSON.stringify({backend:'rocm',protocol:1,model:new URL(this.url).searchParams.get('model'),precision:'fp16'})});return;
  }
  const view=new DataView(value),id=view.getUint32(4,true),w=view.getUint32(16,true),h=view.getUint32(20,true);
  setTimeout(()=>{
   const result=new ArrayBuffer(24+w*h*3),header=new DataView(result);
   [0x33435341,id,w,h,w,h].forEach((n,i)=>header.setUint32(i*4,n,true));
   const bytes=new Uint8Array(result,24);for(let i=0;i<bytes.length;i+=3)bytes.set([255,0,0],i);
   this.onmessage({data:JSON.stringify({processingMs:TEST_DELAY,warmup:false})});
   this.onmessage({data:result});
  },TEST_DELAY);
 }
 close(){this.readyState=3;}
};
`;
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath,args:['--enable-unsafe-webgpu']});
 try{
  const results=[];
  for(const {delay,primed} of [40,350].flatMap(delay=>[false,true].map(primed=>({delay,primed})))){
   const context=await browser.newContext();
   await context.route('**/src/engine/native-worker.js',async route=>{
    let code=fs.readFileSync('src/engine/native-worker.js','utf8');
    if(baseline)code=code.replace('if (!temporal.current) temporal.present(bitmap,crop,size);','');
    await route.fulfill({contentType:'text/javascript',body:`const TEST_DELAY=${delay};\n${fakeSocket}\n${code}`});
   });
   if(baseline){
    await context.route('**/src/engine/native-engine.js',async route=>{
     const code=fs.readFileSync('src/engine/native-engine.js','utf8').replace(
      /if \(!still && !this.stillPresentation && this.settings.fluidVideo && timings.temporal &&[\s\S]*?\n      }/,'')
      .replace(/if \(!still && !this.stillPresentation && this.settings.fluidVideo && timings.temporal\) {[\s\S]*?\n      }/,'');
     await route.fulfill({contentType:'text/javascript',body:code});
    });
    await context.route('**/src/engine/temporal-presenter.js',async route=>{
     const code=fs.readFileSync('src/engine/temporal-presenter.js','utf8')
      .replace('>=300 && !this.unchangedSource(reference)','>=300')
      .replace('temporalStrength(unchanged?0:age','temporalStrength(age');
     await route.fulfill({contentType:'text/javascript',body:code});
    });
   }
   const page=await context.newPage();
   page.on('pageerror',error=>{throw error;});
   await page.goto('http://127.0.0.1:8765/diagnostics/temporal.html');
   const row=await page.evaluate(async ({delay,primed})=>{
    const {NativeEngine}=await import('/src/engine/native-engine.js');
    const {normalizeSettings}=await import('/src/core/settings.js');
    const source=new OffscreenCanvas(160,90);source.getContext('2d').fillRect(0,0,160,90);
    const output=document.createElement('canvas');document.body.append(output);
    const engine=new NativeEngine();engine.diagnostics={event(){},frame(){},flush(){}};
    await engine.initialize({canvas:output,video:{videoWidth:160,videoHeight:90},
     settings:normalizeSettings({backend:'rocm',model:'adore-lite',fluidVideo:true,adaptive:false}),
     connection:{url:'http://127.0.0.1:8766',token:'0'.repeat(64)}});
    engine.currentSize={inputWidth:160,inputHeight:90,outputWidth:160,outputHeight:90};
    engine.makeFrame=async()=>new VideoFrame(source,{timestamp:1000000});
    const video={},crop={x:0,y:0,width:160,height:90};
    const snapshot=new OffscreenCanvas(160,90),ctx=snapshot.getContext('2d');
    const pixel=()=>{ctx.drawImage(output,0,0);return Array.from(ctx.getImageData(80,45,1,1).data).slice(0,3);};
    const settle=async()=>{for(let i=0;i<2;i++)await new Promise(requestAnimationFrame);};
    const displayedPixel=async()=>{
      const started=performance.now();let value;
      // GPU submission does not guarantee OffscreenCanvas commitment in two
      // main-thread rAFs on a cold device. Wait for the real visible pixels,
      // without forcing another presentation or invoking comparison.
      do {await settle();value=pixel();}while(value[0]!==255 && performance.now()-started<500);
      return {value,waitMs:performance.now()-started};
    };
    if(primed)await engine.presentFrame(video,crop);
    const first=await engine.processFrame(video,crop);
    if(primed)await new Promise(resolve=>setTimeout(resolve,130));
    const cold=await displayedPixel(),coldPixel=cold.value;
    const coldCoverage=engine.lastPresentationMetrics?.aiCoverage??first.temporal?.aiCoverage;
    await engine.processFrame(video,crop,{still:true});
    await settle();
    const comparisonPixel=pixel();
    engine.lastPresentationAt=-Infinity;
    await engine.presentFrame(video,crop);
    engine.lastPresentationAt=-Infinity;
    const after=await engine.processFrame(video,crop);
    const resumed=await displayedPixel(),resumedPixel=resumed.value;
    await engine.destroy();output.remove();
    return {delayMs:delay,clockPreviouslyRunning:primed,compositor:first.compositor,coldPixel,comparisonPixel,resumedPixel,
     coldCoverage,resumedCoverage:after.temporal?.aiCoverage,coldDisplayWaitMs:cold.waitMs,resumedDisplayWaitMs:resumed.waitMs,
     visibleWithoutCompare:coldPixel[0]===255,visibleAfterCompare:resumedPixel[0]===255};
   },{delay,primed});
   results.push(row);await context.close();
  }
  const report={date:new Date().toISOString(),baseline,scope:'Simulated inference; actual NativeEngine, worker, VideoFrame and WebGPU. No HIP/model performance claim.',results};
  writeDiagnosticReport(`diagnostics/native-fluid-delivery-${baseline?'baseline':'fixed'}-20261010.json`,report);
  console.log(JSON.stringify(report,null,2));
  if(!baseline&&results.some(row=>row.compositor!=='webgpu'||!row.visibleWithoutCompare||!row.visibleAfterCompare||row.coldCoverage!==1||row.resumedCoverage!==1))throw Error('Native fluid result was not displayed');
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
