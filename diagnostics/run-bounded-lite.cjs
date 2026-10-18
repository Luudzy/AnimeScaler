// One owned native service, one browser, fixed low-load experiment, no retries.
const fs=require('node:fs');
const path=require('node:path');
const net=require('node:net');
const {spawn}=require('node:child_process');
const {once}=require('node:events');
const {createHash}=require('node:crypto');
const {browserRuntime}=require('./browser-runtime.cjs');
const {boundedWait:timeout,cleanupOwnedProbe}=require('./probe-lifecycle.cjs');
const root=path.resolve(__dirname,'..');
const directory=path.resolve(root,process.env.ANIMESCALER_REPORT_DIR||'diagnostics/lite-bounded-output');
const python=process.env.ANIMESCALER_PYTHON||process.env.ANIMESCALER_ROCM_PYTHON;
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const write=(name,value)=>fs.writeFileSync(path.join(directory,name),JSON.stringify(value,null,2)+'\n',{flag:'wx'});
const sha=buffer=>createHash('sha256').update(buffer).digest('hex');

(async()=>{
  if(!python)throw new Error('Set ANIMESCALER_PYTHON or ANIMESCALER_ROCM_PYTHON to the ROCm Python executable');
  if(!fs.existsSync(python))throw new Error('Existing ROCm Python not found');
  fs.mkdirSync(directory,{recursive:true});
  if(fs.existsSync(path.join(directory,'probe-start.json')))throw new Error('Choose a fresh report directory');
  for(const port of [8765,8766])await new Promise((resolve,reject)=>{
    const check=net.createServer();check.once('error',()=>reject(new Error(`Port ${port} already occupied; existing service preserved`)));
    check.listen(port,'127.0.0.1',()=>check.close(resolve));
  });
  const connectionPath=path.join(root,'native/connection.json');
  const oldConnection=fs.existsSync(connectionPath)?fs.readFileSync(connectionPath):null;
  write('probe-start.json',{date:new Date().toISOString(),seconds:20,maxFps:12,model:'adore-lite-v3',preset:'720p',
    media:'diagnostics/stability-media/rezero-ep19-360p.mp4',mediaSha256:sha(fs.readFileSync(path.join(root,'diagnostics/stability-media/rezero-ep19-360p.mp4'))),
    limits:'One serial service/browser; 20 seconds measured after startup, 12 inference FPS, HIP graphs/autotune disabled. 75-second workload deadline plus bounded cleanup; no retries or package changes.'});
  const stdout=fs.openSync(path.join(directory,'native.stdout.txt'),'wx'),stderr=fs.openSync(path.join(directory,'native.stderr.txt'),'wx');
  let service,browser,server,nativeError=null,stopped=false,serviceClosed;
  try {
    const run=(async()=>{
      service=spawn(python,['-u','native/server.py'],{cwd:root,windowsHide:true,
        env:{...process.env,ANIMESCALER_BACKEND:'rocm',ANIMESCALER_HIP_GRAPH:'0',ANIMESCALER_MIOPEN_BENCHMARK:'0',
          ANIMESCALER_CHANNELS_LAST:'0',MIOPEN_FIND_MODE:'FAST',MIOPEN_DISABLE_CACHE:'1',OMP_NUM_THREADS:'2',MKL_NUM_THREADS:'2',
          PYTHONDONTWRITEBYTECODE:'1',MIOPEN_USER_DB_PATH:path.join(directory,'db'),MIOPEN_CUSTOM_CACHE_DIR:path.join(directory,'kernels')},
        stdio:['ignore','pipe','pipe']});
      serviceClosed=new Promise(resolve=>service.once('close',resolve));
      service.on('error',error=>{nativeError=error;});
      for(const [stream,fd] of [[service.stdout,stdout],[service.stderr,stderr]])stream.on('data',chunk=>{
        fs.writeSync(fd,chunk);
        if(/unspecified launch failure|GPU processing failed|Traceback \(most recent call last\)/i.test(chunk.toString()))nativeError=new Error('Native runtime reported a failure; no retry');
      });
      const start=Date.now();let info;
      while(Date.now()-start<20000) {
        if(stopped)throw new Error('Probe stopped');
        if(nativeError)throw nativeError;
        if(service.exitCode!==null)throw new Error('Native service exited during startup');
        try {
          const connection=JSON.parse(fs.readFileSync(connectionPath,'utf8'));
          const response=await fetch('http://127.0.0.1:8766/health',{headers:{Authorization:'Bearer '+connection.token},signal:AbortSignal.timeout(500)});
          if(response.ok){info=await response.json();break;}
        } catch { /* Startup only; never retry a failed GPU workload. */ }
        await delay(250);
      }
      if(stopped)throw new Error('Probe stopped');
      if(!info)throw new Error('Native startup exceeded 20 seconds');
      if(!info.hip||!info.models?.includes('adore-lite-v3'))throw new Error('Requested ROCm/Lite V3 runtime unavailable');
      write('runtime.json',{backend:info.backend,hip:info.hip,gpu:info.gpu,architecture:info.architecture,
        precision:info.precision,weightSha256:info.revisions['adore-lite-v3'],hipGraphs:false,autotune:false});
      const {createDiagnosticServer}=await import('../scripts/serve-diagnostics.mjs');
      // The native origin allowlist deliberately admits only this diagnostic
      // origin. An ephemeral browser origin is rejected before inference.
      server=createDiagnosticServer();server.listen(8765,'127.0.0.1');await once(server,'listening');
      const runtime=browserRuntime();
      browser=await runtime.chromium.launch({headless:true,executablePath:runtime.executablePath,
        args:['--enable-unsafe-webgpu','--autoplay-policy=no-user-gesture-required'],timeout:8000});
      if(stopped){await browser.close();throw new Error('Probe stopped');}
      const page=await browser.newPage({viewport:{width:1000,height:700}}),errors=[];
      page.on('pageerror',error=>errors.push(error.message));
      await page.goto(`http://127.0.0.1:${server.address().port}/diagnostics/playback.html`,{timeout:5000});
      await page.waitForFunction(()=>typeof window.runPlayback==='function',null,{timeout:2000});
      const result=await timeout(page.evaluate(()=>window.runPlayback({model:'adore-lite-v3',
        media:'/diagnostics/stability-media/rezero-ep19-360p.mp4',seconds:20,preset:'720p',fluidVideo:true,maxFps:12,interventionsEnabled:false})),32000,'Bounded playback');
      result.date=new Date().toISOString();result.browser=browser.version();result.browserErrors=errors;
      write('playback-adore-lite-v3.json',result);
      if(nativeError||result.state!=='complete'||errors.length||result.processedFrames<50)throw nativeError||new Error('Bounded playback failed validation');
      console.log(JSON.stringify({state:result.state,frames:result.processedFrames,fps:result.processedFps,
        coverage:result.aiCoverage.mean,lossMean:result.contributionAnalysis.lossMean,boundaries:result.boundaries},null,2));
    })();
    await timeout(run,75000,'Overall probe');
  } catch(error) {write('failure.json',{date:new Date().toISOString(),error:error.message});throw error;}
  finally {
    stopped=true;
    await cleanupOwnedProbe({service,serviceClosed,browser,server,
      logClosers:[()=>service?.stdout?.destroy(),()=>service?.stderr?.destroy(),()=>fs.closeSync(stdout),()=>fs.closeSync(stderr)],
      restoreConnection(){
        if(oldConnection)fs.writeFileSync(connectionPath,oldConnection);
        else if(fs.existsSync(connectionPath))fs.unlinkSync(connectionPath);
        return oldConnection?sha(fs.readFileSync(connectionPath))===sha(oldConnection):!fs.existsSync(connectionPath);
      },record:report=>write('probe-cleanup.json',report)});
  }
})().catch(error=>{console.error(error.message);process.exitCode=1;});
