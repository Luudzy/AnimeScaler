const path = require('node:path');
const home = require('node:os').homedir();
// Local diagnostic helper; override these paths on other machines.
const { chromium } = require(process.env.ANIMESCALER_PLAYWRIGHT || path.join(home, '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright-core'));
const fs = require('node:fs');
(async () => {
  const browser = await chromium.launch({ headless: true, executablePath: process.env.ANIMESCALER_CHROMIUM || path.join(home, 'AppData/Local/ms-playwright/chromium-1223/chrome-win64/chrome.exe'), args: ['--enable-unsafe-webgpu'] });
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on('console', message => { if (message.type() === 'error' || message.type() === 'warning') { errors.push(message.text()); console.log(message.type(), message.text()); } });
    page.on('pageerror', error => { errors.push(error.message); console.log('pageerror', error.message); });
    await page.goto(process.argv[2] || 'http://127.0.0.1:8123/diagnostics/asr-video-quality.html?debug=1');
    await page.waitForFunction(() => /^(PASS|FAIL)/.test(document.querySelector('#result').textContent), { timeout: 120000 });
    const text = await page.locator('#result').textContent();
    console.log(text);
    const gpuErrors = errors.filter(message => /invalid.*(bindgroup|bind group|commandbuffer|command buffer)|validation|device.*lost/i.test(message));
    const adapter = await page.evaluate(async () => {
      const adapter = await navigator.gpu.requestAdapter();
      return adapter ? { ...Object.fromEntries(['vendor', 'architecture', 'device', 'description'].map(key => [key, adapter.info[key]])) } : null;
    });
    if (process.argv[3]) fs.writeFileSync(process.argv[3], JSON.stringify({ date: new Date().toISOString(), adapter, errors, result: text }, null, 2));
    if (process.argv[4]) await page.screenshot({ path: process.argv[4], fullPage: true });
    if (!text.startsWith('PASS') || gpuErrors.length) process.exitCode = 1;
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
