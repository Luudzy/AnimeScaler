const fs=require('node:fs');
const path=require('node:path');
const {browserRuntime}=require('./browser-runtime.cjs');
const {chromium,executablePath}=browserRuntime();
(async()=>{
 const root=path.resolve(__dirname,'..'), output=process.env.ANIMESCALER_VIEWER_DEBUG_OUTPUT || 'diagnostics/stability-20261011/viewer-start-debug';
 fs.mkdirSync(output,{recursive:false});
 const context=await chromium.launchPersistentContext(path.join(output,'profile'),{headless:true,executablePath,args:[`--disable-extensions-except=${root}`,`--load-extension=${root}`,'--enable-unsafe-webgpu','--autoplay-policy=no-user-gesture-required']});
 const report={scope:'Read-only traces of actual Chrome tab/session startup; no API substitution',snapshots:[]};
 try{
  const worker=context.serviceWorkers()[0]||await context.waitForEvent('serviceworker');
  const source=await context.newPage();await source.goto('http://127.0.0.1:8765/diagnostics/playback.html');
  await source.evaluate(async()=>{const v=document.querySelector('video');v.src='/diagnostics/stability-media/rezero-ep19-360p.mp4';await v.play();});
  const sourceId=await worker.evaluate(async()=>(await chrome.tabs.query({url:'http://127.0.0.1:8765/diagnostics/playback.html'}))[0].id);
  await worker.evaluate(async id=>{await chrome.tabs.update(id,{active:true});await chrome.action.openPopup();},sourceId);
  const popup=await context.newPage();await popup.goto(worker.url().replace(/background\.js$/,'popup/popup.html'));
  await worker.evaluate(async id=>chrome.tabs.update(id,{active:true}),sourceId);
  await popup.locator('#model').selectOption('adore-lite-v3');await popup.locator('#preset').selectOption('720p');await popup.locator('#fluidVideo').check();
  await popup.locator('#toggle').click();
  for(let n=0;n<52;n++){
   const state=await worker.evaluate(async()=>{
    const tabs=await chrome.tabs.query({}), stored=await chrome.storage.session.get(null);
    const diagnostic=(await chrome.storage.local.get('animeScalerDiagnostics')).animeScalerDiagnostics;
    const viewers=[];
    for(const tab of tabs.filter(item=>item.url?.includes('/viewer/') || !item.url)){
     let navigation;try{navigation=await chrome.webNavigation.getFrame({tabId:tab.id,frameId:0});}catch(error){navigation={error:error.message};}
     viewers.push({id:tab.id,status:tab.status,url:tab.url,navigation});
    }
    return {tabs:tabs.map(({id,status,url,active})=>({id,status,url,active})),viewers,sessionKeys:Object.keys(stored),errors:(diagnostic?.events||[]).filter(item=>item.type==='error').slice(-3)};
   });
   state.elapsedSeconds=n*.25;
   state.popupStatus=await popup.locator('#status').innerText().catch(()=> 'closed');state.popupMetrics=await popup.locator('#metrics').innerText().catch(()=> 'closed');
   const viewer=context.pages().find(page=>page.url().includes('/viewer/'));
   if(viewer) state.viewerStatus=await viewer.locator('#status').innerText({timeout:500}).catch(()=> 'unavailable');
   report.snapshots.push(state);
   await source.waitForTimeout(250);
  }
 }catch(error){report.error=error.message;}finally{await context.close();fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));}
 console.log(JSON.stringify({snapshots:report.snapshots.length,error:report.error}));
})().catch(error=>{console.error(error);process.exitCode=1;});
