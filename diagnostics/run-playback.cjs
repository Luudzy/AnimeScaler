const fs = require('node:fs');
const path = require('node:path');
const { browserRuntime } = require('./browser-runtime.cjs');
const { chromium, executablePath } = browserRuntime();
const seconds = Number(process.env.ANIMESCALER_PLAYBACK_SECONDS || 300);
const models = (process.env.ANIMESCALER_PLAYBACK_MODELS || 'fsr1-spatial,asr-anime-spatial,adore-lite-v3,zenkai-animevideo-30k').split(',');
const directory = process.env.ANIMESCALER_PLAYBACK_OUTPUT || 'diagnostics/stability-20261011/playback-baseline';
if (fs.existsSync(directory)) throw new Error('Preserve prior playback output; choose a new output directory');
fs.mkdirSync(directory, { recursive: true });

(async () => {
  const browser = await chromium.launch({ headless: true, executablePath, args: ['--enable-unsafe-webgpu', '--autoplay-policy=no-user-gesture-required'] });
  try {
    for (const model of models) {
      const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
      const errors = [], warnings = [];
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); if (message.type() === 'warning') warnings.push(message.text()); });
      await page.goto('http://127.0.0.1:8765/diagnostics/playback.html');
      const running = page.evaluate(options => window.runPlayback(options), { model, seconds, preset: process.env.ANIMESCALER_PLAYBACK_PRESET || '720p', media: '/diagnostics/stability-media/rezero-ep19-360p.mp4', fluidVideo: true });
      const timer = setInterval(async () => {
        try { const progress = await page.evaluate(() => window.playbackProgress?.()); if (progress) { fs.writeFileSync(path.join(directory, 'progress.json'), JSON.stringify(progress, null, 2)); console.log(JSON.stringify(progress)); } } catch {}
      }, 30000);
      let result;
      try { result = await running; } finally { clearInterval(timer); }
      result = { date: new Date().toISOString(), browser: await browser.version(), ...result, browserErrors: errors, browserWarnings: [...new Set(warnings)] };
      fs.writeFileSync(path.join(directory, model + '.json'), JSON.stringify(result, null, 2));
      console.log(JSON.stringify({ model, state: result.state, fps: result.processedFps, p95: result.frameMs.p95, aiCoverage: result.aiCoverage.mean, errors }));
      await page.close();
      if (result.state !== 'complete' || errors.length || result.processedFrames < 20) throw new Error('Playback validation failed for ' + model);
    }
  } finally { await browser.close(); }
})().catch(error => { fs.writeFileSync(path.join(directory, 'failure.json'), JSON.stringify({ date: new Date().toISOString(), error: error.message }, null, 2)); console.error(error); process.exitCode = 1; });
