const {chromium,executablePath}=require('./browser-runtime.cjs').browserRuntime();
const assert = require('node:assert/strict');
(async () => {
  const browser = await chromium.launch({headless:true,executablePath});
  try {
    const page = await browser.newPage();
    const errors=[]; page.on('pageerror', error=>errors.push(error.message));
    await page.addInitScript(() => {
      globalThis.__saved = {animeScalerSettings:JSON.parse(sessionStorage.getItem('settings') || '{"model":"fsr1-spatial"}')};
      globalThis.chrome = {
        storage:{session:{get:async()=>({})},local:{get:async key=>typeof key==='string'?{[key]:__saved[key]}:__saved,set:async values=>{Object.assign(__saved,values);sessionStorage.setItem('settings',JSON.stringify(__saved.animeScalerSettings));}}},
        tabs:{query:async()=>[{id:1}],sendMessage:async()=>({state:'stopped',hasVideo:false})},
        webNavigation:{getAllFrames:async()=>[{frameId:0}]},
        runtime:{sendMessage:async()=>({}),onMessage:{addListener(){}}},
      };
    });
    await page.goto(`${process.env.ANIMESCALER_DIAGNOSTIC_URL || 'http://127.0.0.1:8765'}/popup/popup.html`);
    await page.waitForFunction(()=>document.querySelector('#model').value==='fsr1-spatial');
    assert.equal(await page.locator('#fsrQualityField').isVisible(),true);
    for(const mode of ['performance','balanced','quality','ultra-quality','native-aa']) {
      await page.selectOption('#fsrQuality',mode);
      await page.waitForFunction(mode=>__saved.animeScalerSettings.fsrQuality===mode,mode);
    }
    await page.selectOption('#spatialDenoise','strong');
    await page.waitForFunction(()=>__saved.animeScalerSettings.spatialDenoise==='strong');
    await page.reload();
    await page.waitForFunction(()=>document.querySelector('#spatialDenoise').value==='strong');
    await page.selectOption('#model','asr-anime-temporal');
    await page.waitForFunction(()=>__saved.animeScalerSettings.model==='asr-anime-temporal');
    assert.equal(await page.locator('#fsrQualityField').isVisible(),true);
    assert.match(await page.locator('#modelNote').textContent(),/temporal/);
    await page.selectOption('#model','fsr3-native');
    await page.waitForFunction(()=>__saved.animeScalerSettings.model==='fsr3-native' && __saved.animeScalerSettings.backend==='d3d12');
    assert.equal(await page.locator('#fsrQualityField').isVisible(),true);
    assert.equal(await page.locator('#spatialDenoiseField').isVisible(),false);
    assert.equal(await page.locator('#fluidVideo').isDisabled(),true);
    assert.match(await page.locator('#modelNote').textContent(),/experimental/i);
    await page.reload();
    await page.waitForFunction(()=>document.querySelector('#model').value==='fsr3-native');
    assert.equal(await page.locator('#backend').inputValue(),'d3d12');
    await page.selectOption('#model','adore');
    await page.waitForFunction(()=>document.querySelector('#fsrQualityField').hidden);
    assert.equal(await page.locator('#fsrQualityField').isVisible(),false);
    assert.equal(await page.locator('#spatialDenoiseField').isVisible(),false);
    assert.notEqual(await page.locator('#backend').inputValue(),'d3d12');
    assert.deepEqual(errors,[]);
    console.log('PASS: five spatial presets selectable/persisted; denoise persisted; controls hidden for other models.');
  } finally { await browser.close(); }
})().catch(error=>{console.error(error);process.exitCode=1;});
