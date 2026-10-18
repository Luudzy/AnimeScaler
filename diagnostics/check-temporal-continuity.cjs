// Small serial WebGPU pixel regression test; no Torch, HIP, model or native service.
const {execFileSync}=require('node:child_process');
const {once}=require('node:events');
const {browserRuntime}=require('./browser-runtime.cjs');
const {writeDiagnosticReport}=require('./report-output.cjs');

(async()=>{
  const baselineRef='63f6a57';
  const baseline=execFileSync('git',['show',`${baselineRef}:src/engine/temporal-presenter.js`],{encoding:'utf8'});
  const {createDiagnosticServer}=await import('../scripts/serve-diagnostics.mjs');
  const server=createDiagnosticServer();
  let browser;
  try {
    server.listen(0,'127.0.0.1');await once(server,'listening');
    const {chromium,executablePath}=browserRuntime();
    browser=await chromium.launch({headless:true,executablePath,args:['--enable-unsafe-webgpu'],timeout:15000});
    const results=[];
    // Same pixels, dimensions, checks and browser. Only the presenter changes.
    for(const revision of ['baseline','working-tree']) {
      const context=await browser.newContext();
      try {
        if(revision==='baseline')await context.route('**/src/engine/temporal-presenter.js',route=>route.fulfill({contentType:'text/javascript',body:baseline}));
        const page=await context.newPage();
        const errors=[];page.on('pageerror',error=>errors.push(error.message));
        await page.goto(`http://127.0.0.1:${server.address().port}/diagnostics/temporal.html`,{timeout:15000});
        await page.waitForFunction(()=>typeof document.querySelector('#run').onclick==='function',null,{timeout:10000});
        const result=await page.evaluate(async()=>{
          document.querySelector('#gpu').checked=true;
          let testError=null;
          const run=Promise.resolve().then(()=>document.querySelector('#run').onclick());
          let timer;
          try {
            await Promise.race([run,new Promise((_,reject)=>{timer=setTimeout(()=>reject(new Error('Bounded pixel test exceeded 15 seconds')),15000);})]);
          } catch(error){testError=error.message;}
          finally {clearTimeout(timer);}
          const text=document.querySelector('#result').textContent;
          let checks;try{checks=JSON.parse(text);}catch{throw new Error(`GPU pixel report unavailable: ${text}`);}
          const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});
          const info=adapter?.info;
          return {checks,testError,adapter:info?{vendor:info.vendor,architecture:info.architecture,device:info.device,description:info.description}:null};
        });
        results.push({revision,errors,...result});
      } finally {await context.close();}
    }
    const count=checks=>Object.values(checks).filter(value=>typeof value==='boolean');
    const failures=checks=>Object.entries(checks).filter(([,value])=>value===false).map(([key])=>key);
    const report={date:new Date().toISOString(),baselineRef,browserVersion:browser.version(),
      scope:'160x90 synthetic pixels with actual VideoFrame and WebGPU compositor. Serial baseline/candidate, no Torch/HIP/model or native service. No model quality, FPS, VRAM peak or PC stability claim.',
      results:results.map(row=>({...row,checkCount:count(row.checks).length,failedChecks:failures(row.checks)}))};
    const candidate=report.results[1];
    report.state=candidate.failedChecks.length||candidate.errors.length||candidate.testError||
      candidate.checks.compositor!=='webgpu'||candidate.checkCount<20?'failed':'passed';
    writeDiagnosticReport('diagnostics/temporal-continuity-20261011.json',report);
    console.log(JSON.stringify({state:report.state,results:report.results.map(row=>({revision:row.revision,checkCount:row.checkCount,failedChecks:row.failedChecks,adapter:row.adapter}))},null,2));
    if(report.state!=='passed')throw new Error('Candidate WebGPU continuity checks failed');
  } finally {
    await browser?.close();
    await new Promise(resolve=>server.close(resolve));
  }
})().catch(error=>{console.error(error.message);process.exitCode=1;});
