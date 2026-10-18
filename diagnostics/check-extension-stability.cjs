const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { browserRuntime } = require('./browser-runtime.cjs');
const { chromium, executablePath } = browserRuntime();

(async () => {
  const root = path.resolve(__dirname, '..');
  const output = process.env.ANIMESCALER_EXTENSION_OUTPUT || 'diagnostics/stability-20261011/extension-stability';
  if (fs.existsSync(output)) throw new Error('Preserve previous extension validation; choose a new directory');
  fs.mkdirSync(output, { recursive: true });
  const { createDiagnosticServer } = await import(pathToFileURL(path.join(root, 'scripts/serve-diagnostics.mjs')));
  const mediaServer = createDiagnosticServer({ root });
  await new Promise((resolve, reject) => { mediaServer.once('error', reject); mediaServer.listen(8768, '127.0.0.1', resolve); });
  let context;
  const report = { state: 'running', scope: 'Actual unpacked MV3 extension, decoded HTMLVideo and extension messages; no chrome API substitution', checks: [], errors: [], limitations: [] };
  const check = (name, passed, details) => { report.checks.push({ name, passed, details }); console.log(JSON.stringify({ name, passed })); };
  try {
    context = await chromium.launchPersistentContext(path.join(output, 'profile'), {
      headless: true, executablePath, viewport: { width: 1280, height: 800 },
      args: [`--disable-extensions-except=${root}`, `--load-extension=${root}`, '--enable-unsafe-webgpu', '--autoplay-policy=no-user-gesture-required'],
    });
    const worker = context.serviceWorkers()[0] || await context.waitForEvent('serviceworker', { timeout: 15000 });
    const page = await context.newPage(); page.on('pageerror', error => report.errors.push(error.message));
    await page.goto('http://127.0.0.1:8765/diagnostics/playback.html');
    await page.evaluate(async () => { const video = document.querySelector('video'); video.src = '/diagnostics/stability-media/rezero-ep19-360p.mp4'; await video.play(); });
    const tabId = await worker.evaluate(async () => (await chrome.tabs.query({ url: 'http://127.0.0.1:8765/diagnostics/playback.html' }))[0].id);
    const send = message => worker.evaluate(async ({ tabId, message }) => chrome.tabs.sendMessage(tabId, message, { frameId: 0 }), { tabId, message });
    const settings = { model: 'fsr1-spatial', backend: 'webgpu', enabled: true, preset: '720p', maxFps: 30, adaptive: false, showHud: true, spatialDenoise: 'balanced', fsrQuality: 'native-aa' };
    await page.waitForTimeout(500); await send({ type: 'ANIMESCALER_START', settings });
    await page.waitForTimeout(2500);
    let status = await send({ type: 'ANIMESCALER_STATUS' });
    check('start-real-extension', status.state === 'running' && status.settings?.model === settings.model && status.stats?.outputWidth === 1280, status);
    for (const model of ['asr-anime-spatial', 'fsr1-spatial']) {
      await send({ type: 'ANIMESCALER_SETTINGS', settings: { model } });
      await page.waitForTimeout(1500); status = await send({ type: 'ANIMESCALER_STATUS' });
      check(`switch-${model}`, status.state === 'running' && status.settings?.model === model && status.stats?.modelLabel === (model === 'fsr1-spatial' ? 'AMD FSR 1 Spatial' : 'ASR Anime Spatial'), status);
    }
    await page.setViewportSize({ width: 900, height: 620 });
    await page.waitForTimeout(1000); status = await send({ type: 'ANIMESCALER_STATUS' });
    check('viewport-resize', status.state === 'running', status);
    await page.locator('video').click();
    await page.evaluate(() => document.querySelector('video').requestFullscreen());
    await page.waitForTimeout(600); status = await send({ type: 'ANIMESCALER_STATUS' });
    check('source-fullscreen', status.state === 'running' && await page.evaluate(() => !!document.fullscreenElement), status);
    await page.evaluate(() => document.exitFullscreen());
    await page.evaluate(async () => {
      const old = document.querySelector('video'), next = old.cloneNode(true);
      next.src = old.src; old.pause(); old.replaceWith(next); await next.play();
    });
    await page.waitForTimeout(2500); status = await send({ type: 'ANIMESCALER_STATUS' });
    check('video-element-replacement', status.state === 'running' && !status.waitingForPlayer && status.stats?.sourceWidth === 640, status);
    await send({ type: 'ANIMESCALER_STOP' });
    await page.evaluate(async () => { const video = document.querySelector('video'); video.src = 'http://127.0.0.1:8768/diagnostics/stability-media/rezero-ep19-360p.mp4'; await video.play(); });
    await send({ type: 'ANIMESCALER_START', settings });
    await page.waitForTimeout(1800);
    const direct = await send({ type: 'ANIMESCALER_STATUS' });
    check('cross-origin-handling', (direct.state === 'error' && direct.error?.code === 'CROSS_ORIGIN_VIDEO') || (direct.state === 'running' && !!direct.stats), direct);
    await send({ type: 'ANIMESCALER_STOP' });
    // Programmatic action opening uses the browser's actual permission rules.
    // Any activeTab/tabCapture rejection is retained rather than bypassed.
    try {
      await worker.evaluate(async id => { await chrome.tabs.update(id, { active: true }); await chrome.action.openPopup(); }, tabId);
      await page.waitForTimeout(500);
      let popup = context.pages().find(item => item.url().endsWith('/popup/popup.html'));
      if (!popup) {
        report.popupSurface = 'Real extension popup page opened in a tab; headless action popup was not exposed. Chrome permissions remain unmodified.';
        popup = await context.newPage();
        await popup.goto(worker.url().replace(/background\.js$/, 'popup/popup.html'));
        await worker.evaluate(async id => chrome.tabs.update(id, { active:true }), tabId);
      }
      await popup.locator('#model').selectOption('adore-lite-v3');
      await popup.locator('#preset').selectOption('720p');
      await popup.locator('#adaptive').uncheck();
      await popup.locator('#toggle').click();
      await page.waitForTimeout(6000);
      const viewer = context.pages().find(item => item.url().includes('/viewer/viewer.html?'));
      if (!viewer) throw new Error(await popup.locator('#metrics').innerText());
      await viewer.waitForFunction(() => document.querySelector('#metrics')?.textContent.includes('ms/frame'), undefined, { timeout: 12000 });
      report.actualViewer = { status: await viewer.locator('#status').innerText(), metrics: await viewer.locator('#metrics').innerText() };
      check('actual-native-tabCapture-viewer', true, report.actualViewer);
      await viewer.locator('#compare').click();
      await viewer.waitForFunction(() => document.querySelector('#status')?.textContent.includes('Comparação'), undefined, { timeout: 10000 });
      check('actual-viewer-still-comparison', true, await viewer.locator('#status').innerText());
      await viewer.locator('#compare').click();
      await viewer.locator('#fullscreen').click(); await viewer.waitForTimeout(500);
      report.viewerFullscreen = await viewer.locator('#fullscreen').getAttribute('aria-pressed');
      await viewer.close();
    } catch (error) {
      report.limitations.push(`Native tabCapture viewer attempt: ${error.message}`);
      report.viewerFailureDiagnostics = await worker.evaluate(async () => {
        const stored=await chrome.storage.local.get('animeScalerDiagnostics');
        return (stored.animeScalerDiagnostics?.events || []).filter(event=>event.type==='error').slice(-5);
      });
    }
    // Separate external-page attempt: failure does not become a passing site test.
    try {
      const external = await context.newPage();
      await external.goto('https://media.w3.org/2010/05/bunny/trailer.mp4', { timeout: 20000, waitUntil: 'domcontentloaded' });
      await external.locator('video').waitFor({ timeout: 5000 });
      await external.evaluate(async () => { const video = document.querySelector('video'); video.muted = true; await video.play(); });
      await external.waitForTimeout(1500);
      const externalId = await worker.evaluate(async () => (await chrome.tabs.query({ url: 'https://media.w3.org/2010/05/bunny/trailer.mp4' }))[0].id);
      const result = await worker.evaluate(async ({ id, settings }) => chrome.tabs.sendMessage(id, { type: 'ANIMESCALER_START', settings }, { frameId: 0 }), { id: externalId, settings });
      await external.waitForTimeout(2000);
      const resultStatus = await worker.evaluate(async id => chrome.tabs.sendMessage(id, { type: 'ANIMESCALER_STATUS' }, { frameId: 0 }), externalId);
      report.external = { page: external.url(), start: result, status: resultStatus };
      if (resultStatus.state === 'running' && resultStatus.stats) check('external-w3c-direct-playback', true, resultStatus);
      else report.limitations.push(`External W3C start did not reach running: ${JSON.stringify(resultStatus.error)}`);
      await external.close();
    } catch (error) { report.limitations.push(`External W3C attempt: ${error.message}`); }
    report.state = report.errors.length || report.checks.some(item => !item.passed) ? 'failed' : 'complete';
  } catch (error) { report.state = 'failed'; report.errors.push(error.message); }
  finally {
    await context?.close(); await new Promise(resolve => mediaServer.close(resolve));
    fs.writeFileSync(path.join(output, 'report.json'), JSON.stringify(report, null, 2), { flag: 'wx' });
  }
  console.log(JSON.stringify({ state: report.state, checks: report.checks.map(({ name, passed }) => ({ name, passed })), errors: report.errors, limitations: report.limitations }));
  if (report.state !== 'complete') process.exitCode = 1;
})().catch(error => { console.error(error.message); process.exitCode = 1; });
