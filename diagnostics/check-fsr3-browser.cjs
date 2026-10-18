const fs=require('node:fs');
const {chromium}=require(process.env.ANIMESCALER_PLAYWRIGHT || 'C:/Users/mybad/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright-core');
(async()=>{
  const browser=await chromium.launch({headless:true,executablePath:process.env.ANIMESCALER_CHROMIUM || 'C:/Users/mybad/AppData/Local/ms-playwright/chromium-1223/chrome-win64/chrome.exe',args:['--enable-unsafe-webgpu']});
  try {
    const page=await browser.newPage();const errors=[];
    const connection=JSON.parse(fs.readFileSync('native/fsr3-connection.json','utf8'));
    await page.addInitScript(value=>{window.__fsr3Connection=value;},connection);
    page.on('pageerror',error=>errors.push(error.message));
    page.on('console',message=>{if(message.type()==='error')errors.push(message.text());});
    await page.goto('http://127.0.0.1:8765/diagnostics/fsr3-native-viewer.html');
    await page.waitForFunction(()=>/^(PASS|FAIL)/.test(document.querySelector('#result').textContent),null,{timeout:60000});
    const result=await page.locator('#result').textContent();
    fs.writeFileSync('diagnostics/fsr3-browser-worker.json',JSON.stringify({date:new Date().toISOString(),errors,result},null,2));
    console.log(result);
    if(!result.startsWith('PASS')||errors.length)throw Error('Native browser verification failed');
  }finally{await browser.close();}
})().catch(error=>{console.error(error.message);process.exitCode=1;});
