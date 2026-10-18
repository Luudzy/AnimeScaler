const fs=require('node:fs');
const {writeDiagnosticReport}=require('./report-output.cjs');
const {chromium,executablePath}=require('./browser-runtime.cjs').browserRuntime();
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath,args:['--enable-unsafe-webgpu']});
 try{
  const page=await browser.newPage(),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  const results=await page.goto('http://127.0.0.1:8765/diagnostics/temporal.html').then(()=>page.evaluate(async()=>{
   const {GpuPrePost}=await import('/src/engine/gpu-prepost.js');
   const {createNativeGpuRenderer}=await import('/src/engine/native-gpu-renderer.js');
   const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'}),device=await adapter.requestDevice();
   device.addEventListener('uncapturederror',e=>{throw new Error(e.error.message);});
   const canvas=new OffscreenCanvas(1920,1080),ctx=canvas.getContext('2d');
   const out=new OffscreenCanvas(1,1),gpu=new GpuPrePost({device,canvas:out});
   const configs=[{sourceWidth:1920,sourceHeight:1080,inputWidth:640,inputHeight:360},
    {sourceWidth:640,sourceHeight:360,inputWidth:800,inputHeight:450}];
   let copies=0;
   const copy=device.queue.copyExternalImageToTexture.bind(device.queue);
   device.queue.copyExternalImageToTexture=(...args)=>{copies++;return copy(...args);};
   const tensor=async()=>{
    const size=gpu.getInputBuffer().size,read=device.createBuffer({size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    const encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(gpu.getInputBuffer(),0,read,0,size);device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);const data=new Float32Array(read.getMappedRange()).slice(0,gpu.config.inputWidth*gpu.config.inputHeight*3);read.unmap();read.destroy();return data;
   };
   const parity=[];
   for(const config of configs){
    await gpu.configure(config);
    if(!gpu.externalPrePipeline)throw Error('Direct compute shader failed to compile');
    const crop=config.sourceWidth===1920?null:{x:123,y:81,width:640,height:360};
    for(const format of ['RGBA','NV12']){
     ctx.fillStyle='#2360ad';ctx.fillRect(0,0,1920,1080);
     for(let i=0;i<40;i++){ctx.fillStyle=`rgb(${i*6},${220-i*4},${i*3})`;ctx.fillRect(i*48,0,48,1080);}
     let frame;
     if(format==='RGBA')frame=new VideoFrame(canvas,{timestamp:0});
     else {
      const bytes=new Uint8Array(1920*1080*3/2);bytes.fill(128,1920*1080);
      for(let y=0;y<1080;y++)for(let x=0;x<1920;x++)bytes[y*1920+x]=16+(x+y)%220;
      for(let y=0;y<540;y++)for(let x=0;x<960;x++){const i=1920*1080+(y*960+x)*2;bytes[i]=64+x%128;bytes[i+1]=96+y%96;}
      frame=new VideoFrame(bytes,{format:'NV12',codedWidth:1920,codedHeight:1080,timestamp:0,colorSpace:{matrix:'bt709',primaries:'bt709',transfer:'bt709',fullRange:false}});
     }
     const pipeline=gpu.externalPrePipeline;
     gpu.externalPrePipeline=null;gpu.uploadVideoFrame(frame,crop);gpu.preprocess();const before=await tensor();
     gpu.externalPrePipeline=pipeline;const count=copies;gpu.uploadVideoFrame(frame,crop);gpu.preprocess();const after=await tensor();
     let max=0,sum=0;for(let i=0;i<after.length;i++){const error=Math.abs(after[i]-before[i]);max=Math.max(max,error);sum+=error;}
     parity.push({format,crop:!!crop,maxAbsError:max,meanAbsError:sum/after.length,directCopies:copies-count,sampleRGB:[after[0],after[config.inputWidth*config.inputHeight],after[2*config.inputWidth*config.inputHeight]]});frame.close();
    }
   }
   await gpu.configure(configs[0]);
   const liveCanvas=document.createElement('canvas');liveCanvas.width=1920;liveCanvas.height=1080;
   liveCanvas.getContext('2d').drawImage(canvas,0,0);
   const stream=liveCanvas.captureStream(0),video=document.createElement('video');video.muted=true;video.srcObject=stream;
   stream.getVideoTracks()[0].requestFrame();await video.play();
   const videoPipeline=gpu.externalPrePipeline;
   gpu.externalPrePipeline=null;gpu.uploadVideoFrame(video);gpu.preprocess();const videoBefore=await tensor();
   gpu.externalPrePipeline=videoPipeline;const videoCopies=copies;gpu.uploadVideoFrame(video);gpu.preprocess();const videoAfter=await tensor();
   let videoMax=0,videoSum=0;for(let i=0;i<videoAfter.length;i++){const error=Math.abs(videoAfter[i]-videoBefore[i]);videoMax=Math.max(videoMax,error);videoSum+=error;}
   parity.push({format:'HTMLVideoElement',crop:false,maxAbsError:videoMax,meanAbsError:videoSum/videoAfter.length,directCopies:copies-videoCopies});
   video.pause();for(const track of stream.getTracks())track.stop();video.srcObject=null;
   const realCanvas=new OffscreenCanvas(1280,720),realContext=realCanvas.getContext('2d');
   await gpu.configure({sourceWidth:1280,sourceHeight:720,inputWidth:640,inputHeight:360});
   const realParity=[];
   for(const clip of ['clip-60s-360-to-720','clip-600s-360-to-720','clip-1200s-360-to-720']){
    let max=0,sum=0,channels=0,directCopies=0;
    for(let n=0;n<48;n++){
     const response=await fetch(`/diagnostics/video-sequences/rezero-real-01/${clip}/hr/${String(n).padStart(8,'0')}.png`);
     if(!response.ok)throw Error('Real frame unavailable');
     const image=await createImageBitmap(await response.blob());realContext.drawImage(image,0,0);image.close();
     const frame=new VideoFrame(realCanvas,{timestamp:n*41667}),pipeline=gpu.externalPrePipeline;
     gpu.externalPrePipeline=null;gpu.uploadVideoFrame(frame);gpu.preprocess();const before=await tensor();
     gpu.externalPrePipeline=pipeline;const count=copies;gpu.uploadVideoFrame(frame);gpu.preprocess();const after=await tensor();
     directCopies+=copies-count;
     for(let i=0;i<after.length;i++){const error=Math.abs(after[i]-before[i]);max=Math.max(max,error);sum+=error;}
     channels+=after.length;frame.close();
    }
    realParity.push({clip,frames:48,maxAbsError:max,meanAbsError:sum/channels,directCopies});
   }
   await gpu.configure(configs[0]);
   const summarize=a=>{const s=[...a].sort((x,y)=>x-y);return {meanMs:a.reduce((x,y)=>x+y,0)/a.length,medianMs:s[Math.ceil(s.length*.5)-1],p95Ms:s[Math.ceil(s.length*.95)-1]};};
   const timings={};
   const ext=gpu.externalPrePipeline;
   // Alternate paths per distinct synthetic frame; completion synchronization
   // measures the full upload+preprocess, not just CPU queue submission.
   const samples={copy:[],direct:[]},counts={copy:0,direct:0};
   for(let n=0;n<65;n++){
    ctx.fillStyle=`rgb(${n*3%255},80,170)`;ctx.fillRect(n*13%1800,20,120,120);
    const frame=new VideoFrame(canvas,{timestamp:n*41667});
    for(const mode of (n%2?['direct','copy']:['copy','direct'])){
     gpu.externalPrePipeline=mode==='direct'?ext:null;
     const count=copies,start=performance.now();gpu.uploadVideoFrame(frame);gpu.preprocess();await device.queue.onSubmittedWorkDone();
     if(n>=15){samples[mode].push(performance.now()-start);counts[mode]+=copies-count;}
    }
    frame.close();
   }
   for(const mode of ['copy','direct'])timings[mode]={...summarize(samples[mode]),copies:counts[mode],frames:50};
   const avoidedFullSourceTextureBytes=1920*1080*4;
   gpu.destroy();device.destroy();
   const target=new OffscreenCanvas(160,90),renderer=await createNativeGpuRenderer(target);
   if(!renderer?.externalPipeline)throw Error('Direct native compositor failed to compile');
   let renderCopies=0;const originalCopy=renderer.device.queue.copyExternalImageToTexture.bind(renderer.device.queue);
   renderer.device.queue.copyExternalImageToTexture=(...args)=>{renderCopies++;return originalCopy(...args);};
   renderer.resize(160,90);ctx.fillStyle='rgb(20,60,180)';ctx.fillRect(0,0,1920,1080);
   const frame=new VideoFrame(canvas,{timestamp:0});renderer.setFrame(frame,{x:123,y:81,width:640,height:360});renderer.draw();frame.close();await renderer.device.queue.onSubmittedWorkDone();
   const snapshot=new OffscreenCanvas(160,90),read=snapshot.getContext('2d');read.drawImage(target,0,0);
   const compositor={sourceMethod:renderer.sourceMethod,copies:renderCopies,pixel:Array.from(read.getImageData(80,45,1,1).data).slice(0,3)};
   renderer.destroy();return {adapter:{vendor:adapter.info.vendor,architecture:adapter.info.architecture,device:adapter.info.device,description:adapter.info.description},parity,realParity,timings,avoidedFullSourceTextureBytes,compositor};
  }));
  const report={date:new Date().toISOString(),scope:'GPU-completed upload/preprocess only, synthetic distinct VideoFrames. No model inference or player FPS. 15 warmups + 50 measured frames per path.',errors,...results};
  writeDiagnosticReport('diagnostics/direct-video-20261010.json',report);
  console.log(JSON.stringify(report,null,2));
  if(errors.length||[...results.parity,...results.realParity].some(p=>p.directCopies!==0||p.maxAbsError>2/255)||results.compositor.copies||results.compositor.pixel.some((v,i)=>Math.abs(v-[20,60,180][i])>1))throw Error('Direct video parity/copy check failed');
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
