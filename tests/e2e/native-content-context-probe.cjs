const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const { browserRuntime } = require('../../diagnostics/browser-runtime.cjs');
const { chromium, executablePath } = browserRuntime();

const root = path.resolve(__dirname, '../..');
const output = path.resolve(root, process.env.ANIMESCALER_PROBE_OUTPUT ||
  `diagnostics/stability-20261015/native-content-context-probe/${process.env.ANIMESCALER_PROBE_BROWSER || 'chromium'}`);
const browserName = process.env.ANIMESCALER_PROBE_BROWSER || 'chromium';
const diagnosticUrl = 'http://127.0.0.1:8765/diagnostics/playback.html';
let extensionRoot;
let context;
let probeTimer;
const report = { browser: browserName, state: 'running', scope: 'Temporary unpacked extension; one local ROCm frame; sanitized telemetry only' };

async function copy(relative) {
  await fs.cp(path.join(root, relative), path.join(extensionRoot, relative), { recursive: true });
}

(async () => {
extensionRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'animescaler-content-probe-'));
try {
  for (const relative of ['manifest.json', 'background.js', 'content-bootstrap.js', 'native-inline.html', 'assets', 'popup', 'src']) {
    await copy(relative);
  }
  await fs.appendFile(path.join(extensionRoot, 'background.js'), `\nchrome.runtime.onMessage.addListener((message, sender, sendResponse) => {\n  if (message?.type !== 'ANIMESCALER_TEST_NATIVE_CONNECTION' || sender.id !== chrome.runtime.id) return;\n  fetch(chrome.runtime.getURL('native/connection.json'), { cache: 'no-store' })\n    .then(response => response.ok ? response.json() : Promise.reject(new Error('CONFIG_UNAVAILABLE')))\n    .then(sendResponse, () => sendResponse({ error: 'CONFIG_UNAVAILABLE' }));\n  return true;\n});\n`);
  await fs.mkdir(path.join(extensionRoot, 'native'), { recursive: true });
  await copy('native/connection.json');
  const manifestPath = path.join(extensionRoot, 'manifest.json');
  const manifest = JSON.parse(await fs.readFile(manifestPath, 'utf8'));
  manifest.content_scripts.push({
    matches: [diagnosticUrl],
    js: ['native-content-context-probe.js'],
    run_at: 'document_idle',
    all_frames: false,
  });
  await fs.writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
  await fs.writeFile(path.join(extensionRoot, 'native-content-context-probe.js'), String.raw`(() => {
    const report = { extensionId: chrome.runtime.id, configReadByServiceWorker: false, checks: {} };
    const OriginalWorker = globalThis.Worker;
    let engine;
    globalThis.Worker = class extends OriginalWorker {
      constructor(url, options) {
        report.workerUrl = String(url);
        super(url, options);
      }
    };
    const finish = value => {
      globalThis.Worker = OriginalWorker;
      window.postMessage({ type: 'ANIMESCALER_NATIVE_CONTEXT_PROBE', report: value }, '*');
    };
    (async () => {
      const deadline = Date.now() + 30000;
      let video = document.querySelector('#source');
      while ((!video || video.readyState < 2 || !video.videoWidth) && Date.now() < deadline) {
        await new Promise(resolve => setTimeout(resolve, 100));
        video = document.querySelector('#source');
      }
      if (!video || video.readyState < 2 || !video.videoWidth) throw Object.assign(new Error('VIDEO_NOT_READY'), { code: 'VIDEO_NOT_READY' });
      const { NativeEngine } = await import(chrome.runtime.getURL('src/engine/native-engine.js'));
      const connection = await chrome.runtime.sendMessage({ type: 'ANIMESCALER_TEST_NATIVE_CONNECTION' });
      if (!connection || connection.error) throw Object.assign(new Error('PRIVATE_CONFIG_UNAVAILABLE'), { code: 'PRIVATE_CONFIG_UNAVAILABLE' });
      report.configReadByServiceWorker = true;
      engine = new NativeEngine();
      const canvas = document.createElement('canvas');
      canvas.width = 1;
      canvas.height = 1;
      const settings = { backend: 'rocm', model: 'adore-lite-v3', preset: '720p', maxFps: 24,
        adaptive: false, fluidVideo: false, fsrQuality: 'native-aa', spatialDenoise: 'balanced' };
      await engine.initialize({ canvas, video, settings, connection });
      report.nativeInfo = {
        backend: engine.nativeInfo?.backend,
        gpu: engine.nativeInfo?.gpu,
        architecture: engine.nativeInfo?.architecture,
        precision: engine.nativeInfo?.precision,
      };
      const stats = await engine.processFrame(video);
      report.frame = {
        completed: true,
        inputWidth: stats.sourceWidth,
        inputHeight: stats.sourceHeight,
        outputWidth: stats.outputWidth,
        outputHeight: stats.outputHeight,
        totalMs: stats.totalMs,
        inferenceMs: stats.inferenceMs,
      };
      report.checks.workerUsesExtensionOrigin = new URL(report.workerUrl).protocol === 'chrome-extension:';
      report.checks.privateConfigReadInExtensionContext = report.configReadByServiceWorker;
      report.checks.nativeServiceAuthenticated = report.frame.completed && report.nativeInfo?.backend === 'rocm';
    })().catch(error => {
      report.errorCode = error?.code || 'PROBE_FAILED';
      report.errorName = error?.name || 'Error';
      report.errorMessage = String(error?.message || '').replace(/\b[a-f0-9]{64}\b/gi, '[redacted]');
    }).finally(async () => {
      try { await engine?.destroy(); } catch { report.cleanupError = true; }
      finish(report);
    });
  })();`);

  await fs.mkdir(output, { recursive: true });
  const profile = path.join(output, 'profile');
  context = await chromium.launchPersistentContext(profile, {
    headless: true,
    executablePath,
    viewport: { width: 1280, height: 800 },
    args: [`--disable-extensions-except=${extensionRoot}`, `--load-extension=${extensionRoot}`,
      '--enable-unsafe-webgpu', '--autoplay-policy=no-user-gesture-required'],
  });
  report.browserVersion = context.browser()?.version() || 'unknown';
  const page = await context.newPage();
  await page.addInitScript(() => {
    window.__animeScalerPrivateConfigFetches = 0;
    const fetch = window.fetch.bind(window);
    window.fetch = (...args) => {
      if (String(args[0]?.url ?? args[0] ?? '').includes('/native/connection.json')) {
        window.__animeScalerPrivateConfigFetches++;
      }
      return fetch(...args);
    };
  });
  let resolveProbe;
  const probeResult = new Promise(resolve => { resolveProbe = resolve; });
  probeTimer = setTimeout(() => resolveProbe({ errorCode: 'PROBE_TIMEOUT' }), 60000);
  page.on('console', message => {
    if (message.type() === 'error' && message.text().includes('native-content-context-probe')) {
      report.consoleError = 'CONTENT_PROBE_ERROR';
    }
  });
  await page.exposeFunction('__probeFinished', value => { clearTimeout(probeTimer); resolveProbe(value); });
  await page.addInitScript(() => window.addEventListener('message', event => {
    if (event.source === window && event.data?.type === 'ANIMESCALER_NATIVE_CONTEXT_PROBE') {
      void window.__probeFinished(event.data.report);
    }
  }));
  await page.goto(diagnosticUrl);
  await page.evaluate(async source => {
    const video = document.querySelector('#source');
    video.removeAttribute('muted');
    video.muted = false;
    video.src = source;
    await video.play();
  }, '/diagnostics/stability-media/rezero-ep19-360p.mp4');
  const content = await probeResult;
  report.content = content;
  const pageObservedPrivateFetch = await page.evaluate(() => window.__animeScalerPrivateConfigFetches > 0);
  const privateRequest = await page.evaluate(async extensionId => {
    try {
      const response = await fetch(`chrome-extension://${extensionId}/native/connection.json`, { cache: 'no-store' });
      return { accessible: response.ok, status: response.status };
    } catch { return { accessible: false, blocked: true }; }
  }, content.extensionId);
  report.pageProbe = {
    privateConfigAccessible: privateRequest.accessible,
    serviceWorkerConfigFetchObservedByPage: pageObservedPrivateFetch,
  };
  report.checks = {
    ...(content.checks || {}),
    privateConfigNotAccessibleToPage: privateRequest.accessible === false,
    serviceWorkerConfigFetchNotObservedByPage: report.pageProbe.serviceWorkerConfigFetchObservedByPage === false,
    noProbeError: !content.errorCode,
    cleanupCompleted: content.cleanupError !== true,
  };
  report.state = Object.values(report.checks).every(Boolean) ? 'complete' : 'failed';
} catch (error) {
  report.state = 'failed';
  report.error = error.message;
} finally {
  clearTimeout(probeTimer);
  await context?.close();
  report.finishedAt = new Date().toISOString();
  const reportPath = path.join(output, 'report.json');
  await fs.writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`, { flag: 'wx' });
  if (extensionRoot) await fs.rm(extensionRoot, { recursive: true, force: true });
}

console.log(JSON.stringify({ state: report.state, browser: browserName,
  checks: report.checks, frame: report.content?.frame, error: report.content?.errorCode || report.error }));
if (report.state !== 'complete') process.exitCode = 1;
})().catch(error => { console.error(error.message); process.exitCode = 1; });
