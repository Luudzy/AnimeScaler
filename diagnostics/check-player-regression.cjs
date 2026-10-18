const fs=require('node:fs');
const path=require('node:path');
const {once}=require('node:events');
const {execFileSync}=require('node:child_process');
const {createHash}=require('node:crypto');
const {browserRuntime}=require('./browser-runtime.cjs');

async function bounded(promise,ms,label){let timer;try{return await Promise.race([promise,new Promise((_,reject)=>{timer=setTimeout(()=>reject(new Error(label+' timed out')),ms);})]);}finally{clearTimeout(timer);}}
(async()=>{
  const root=path.resolve(__dirname,'..'),output=path.resolve(root,process.argv[2]??'diagnostics/player-browser-20261011');
  if(fs.existsSync(output))throw new Error('Use a new output directory');
  fs.mkdirSync(output,{recursive:true});
  const baseline=execFileSync('git',['show','c236644:src/content/controller.js'],{cwd:root,encoding:'utf8'});
  const {createDiagnosticServer}=await import('../scripts/serve-diagnostics.mjs');
  const server=createDiagnosticServer(),args=['--disable-gpu','--disable-accelerated-video-decode','--disable-accelerated-2d-canvas','--autoplay-policy=no-user-gesture-required'];
  let browser;
  try {
    server.listen(0,'127.0.0.1');await bounded(once(server,'listening'),3000,'Server');
    const runtime=browserRuntime();browser=await runtime.chromium.launch({headless:true,executablePath:runtime.executablePath,args,timeout:8000});
    const results=[];
    for(const revision of ['baseline','working-tree']) {
      const context=await browser.newContext();
      try {
        if(revision==='baseline')await context.route('**/src/content/controller.js',route=>route.fulfill({status:200,contentType:'text/javascript',body:baseline}));
        const page=await context.newPage();
        await page.goto(`http://127.0.0.1:${server.address().port}/diagnostics/player-regression.html`,{timeout:5000});
        await page.waitForFunction(()=>typeof window.runPlayerRegression==='function',null,{timeout:3000});
        results.push({revision,...await bounded(page.evaluate(()=>window.runPlayerRegression()),7000,'Player checks')});
      } finally {await bounded(context.close(),3000,'Context close');}
    }
    const sha=data=>createHash('sha256').update(data).digest('hex');
    const report={date:new Date().toISOString(),browser:browser.version(),args,baselineRef:'c236644',
      baselineControllerSha256:sha(baseline),candidateControllerSha256:sha(fs.readFileSync(path.join(root,'src/content/controller.js'))),
      mediaSha256:sha(fs.readFileSync(path.join(root,'diagnostics/stability-media/rezero-ep19-360p.mp4'))),
      modelInference:false,nativeRuntimeLoaded:false,results};
    fs.writeFileSync(path.join(output,'results.json'),JSON.stringify(report,null,2)+'\n');
    console.log(JSON.stringify(report,null,2));
    if(results[1].failed||!results[0].failed)throw new Error('Expected failing baseline and passing candidate');
  } finally {
    try {if(browser)await bounded(browser.close(),5000,'Browser close');}
    finally {server.closeAllConnections();await bounded(new Promise(resolve=>server.close(resolve)),3000,'Server close');}
  }
})().catch(error=>{console.error(error.message);process.exitCode=1;});
