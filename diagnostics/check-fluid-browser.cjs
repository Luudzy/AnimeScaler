const fs=require('node:fs');
const {writeDiagnosticReport}=require('./report-output.cjs');
const {chromium,executablePath}=require('./browser-runtime.cjs').browserRuntime();
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath,args:['--enable-unsafe-webgpu']});
 try{
  const page=await browser.newPage(),errors=[];
  page.on('pageerror',error=>errors.push(error.message));
  page.on('console',message=>{if(message.type()==='error')errors.push(message.text());});
  await page.goto('http://127.0.0.1:8765/diagnostics/temporal.html');
  const results=[];
  for(const gpu of [true,false]){
   results.push(await page.evaluate(async gpu=>{document.querySelector('#gpu').checked=gpu;await document.querySelector('#run').onclick();return JSON.parse(document.querySelector('#result').textContent);},gpu));
  }
  writeDiagnosticReport('diagnostics/fluid-browser-20261010.json',{date:new Date().toISOString(),scope:'Synthetic pixel sequences with actual VideoFrame and compositor. Not model quality or live-site FPS.',errors,results});
  if(errors.length||results.some(row=>Object.values(row).some(value=>value===false)))throw Error('Fluid compositor checks failed');
  console.log(JSON.stringify({errors,results},null,2));
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
