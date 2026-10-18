const fs = require('node:fs');
const path = require('node:path');
const { browserRuntime } = require('./browser-runtime.cjs');
const { chromium, executablePath } = browserRuntime();

const root = path.resolve(__dirname, '..');
const durationMs = Number(process.env.ANIMESCALER_VIEWER_BENCHMARK_MS || 20000);
const model = process.env.ANIMESCALER_VIEWER_BENCHMARK_MODEL || 'adore-lite-v3';
const output = path.resolve(root, process.env.ANIMESCALER_VIEWER_BENCHMARK_OUTPUT ||
  `diagnostics/native-viewer-window-${new Date().toISOString().replace(/[:.]/g, '-')}.json`);
const profile = path.resolve(root, `diagnostics/native-viewer-profile-${process.pid}`);
if (fs.existsSync(output) || fs.existsSync(profile)) throw new Error('Preserve prior benchmark output; choose a new output and profile path');

(async () => {
  const browser = await chromium.launch({ executablePath, headless: true, args: ['--autoplay-policy=no-user-gesture-required'] });
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  const report = { state: 'running', scope: 'Real NativeEngine and local service with animated synthetic canvas capture; no site tabCapture',
    model, input: '640x360', preset: '720p', durationMs, samples: [], errors: [] };
  page.on('pageerror', error => report.errors.push(error.message));
  try {
    await page.goto('http://127.0.0.1:8765/diagnostics/native-viewer.html');
    await page.locator('#model').selectOption(model);
    await page.locator('#start').click();
    await page.waitForFunction(() => {
      const text = document.querySelector('#status')?.textContent ?? '';
      return /Viewer ROCM ativo|Erro no viewer|Falha ao iniciar|Usando WebGPU/.test(text);
    }, null, { timeout: 20000 });
    report.started = await page.locator('#status').textContent();
    report.startedAt = Date.now();
    const endAt = Date.now() + durationMs;
    while (Date.now() < endAt) {
      await new Promise(resolve => setTimeout(resolve, Math.min(5000, endAt - Date.now())));
      report.samples.push({ atMs: Date.now() - report.startedAt,
        status: await page.locator('#status').textContent(),
        metrics: await page.locator('#metrics').textContent(),
        performance: await page.locator('#report').textContent() });
    }
    report.state = report.errors.length || report.samples.some(sample => !sample.status.includes('Viewer ROCM ativo'))
      ? 'failed' : 'complete';
  } catch (error) {
    report.state = 'failed'; report.errors.push(error.message);
  } finally {
    await page.close().catch(() => {});
    await browser.close();
    fs.writeFileSync(output, JSON.stringify(report, null, 2), { flag: 'wx' });
  }
  console.log(JSON.stringify({ state: report.state, model: report.model, started: report.started,
    samples: report.samples, errors: report.errors, output }));
  if (report.state !== 'complete') process.exitCode = 1;
})();
