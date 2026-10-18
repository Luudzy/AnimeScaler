const fs = require('node:fs');
const path = require('node:path');
const { browserRuntime } = require('./browser-runtime.cjs');
const { chromium, executablePath } = browserRuntime();

(async () => {
  const root = path.resolve(__dirname, '..');
  const runId = new Date().toISOString().replace(/[:.]/g, '-');
  const output = path.resolve(root, process.env.ANIMESCALER_EXTENSION_SMOKE_OUTPUT ??
    `diagnostics/stability-20261015/extension-smoke-${runId}.json`);
  const profile = path.resolve(root, process.env.ANIMESCALER_EXTENSION_SMOKE_PROFILE ??
    `diagnostics/stability-20261015/extension-profile-${runId}`);
  const mediaUrl = process.env.ANIMESCALER_PLAYBACK_MEDIA_URL ?? '/diagnostics/stability-media/rezero-ep19-360p.mp4';
  const model = process.env.ANIMESCALER_PLAYBACK_MODEL ?? 'fsr1-spatial';
  const expectAudio = process.env.ANIMESCALER_PLAYBACK_EXPECT_AUDIO === '1';
  if (fs.existsSync(output) || fs.existsSync(profile)) throw new Error('Preserve prior smoke output; choose new output and profile paths');
  const context = await chromium.launchPersistentContext(profile, {
    headless: true, executablePath, viewport: { width: 1280, height: 800 },
    args: [`--disable-extensions-except=${root}`, `--load-extension=${root}`, '--enable-unsafe-webgpu', '--autoplay-policy=no-user-gesture-required'],
  });
  const report = { scope: 'Actual unpacked MV3 extension on a local decoded-video page; not an external streaming site.', checks: [], errors: [] };
  try {
    const worker = context.serviceWorkers()[0] || await context.waitForEvent('serviceworker', { timeout: 15000 });
    const page = await context.newPage();
    page.on('pageerror', error => report.errors.push(error.message));
    await page.goto('http://127.0.0.1:8765/diagnostics/playback.html');
    await page.evaluate(async ({ mediaUrl, expectAudio }) => {
      const video = document.querySelector('video');
      video.src = mediaUrl;
      if (expectAudio) {
        video.muted = false;
        video.volume = 1;
        const context = new AudioContext();
        const analyser = context.createAnalyser();
        analyser.fftSize = 2048;
        context.createMediaElementSource(video).connect(analyser);
        analyser.connect(context.destination);
        await context.resume();
        window.__playbackAudio = { context, analyser };
      }
      await video.play();
    }, { mediaUrl, expectAudio });
    report.media = { url: mediaUrl, audioExpected: expectAudio };
    const readAudio = () => page.evaluate(() => {
      const video = document.querySelector('video');
      const audio = window.__playbackAudio;
      if (!audio) return null;
      const samples = new Float32Array(audio.analyser.fftSize);
      audio.analyser.getFloatTimeDomainData(samples);
      const rms = Math.sqrt(samples.reduce((sum, sample) => sum + sample * sample, 0) / samples.length);
      return { rms, contextState: audio.context.state, muted: video.muted, paused: video.paused, currentTime: video.currentTime };
    });
    if (expectAudio) report.audioBefore = await readAudio();
    const tabId = await worker.evaluate(async () => (await chrome.tabs.query({ url: 'http://127.0.0.1:8765/diagnostics/playback.html' }))[0].id);
    const send = message => worker.evaluate(async ({ tabId, message }) => chrome.tabs.sendMessage(tabId, message, { frameId: 0 }), { tabId, message });
    await page.waitForTimeout(700);
    report.probe = await send({ type: 'ANIMESCALER_PROBE' });
    const settings = { model, backend: 'webgpu', enabled: true, preset: '720p', maxFps: 30, adaptive: false, showHud: true, spatialDenoise: 'balanced', fsrQuality: 'native-aa' };
    report.start = await send({ type: 'ANIMESCALER_START', settings });
    await page.waitForTimeout(3000);
    report.running = await send({ type: 'ANIMESCALER_STATUS' });
    report.checks.push({ name: 'actual-extension-start', passed: report.running?.state === 'running' && report.running.stats?.outputWidth === 1280 });
    if (expectAudio) {
      report.audioDuring = await readAudio();
      report.checks.push({ name: 'source-audio-decoded-during-processing', passed: report.audioDuring?.rms > 0.001 && report.audioDuring?.contextState === 'running' && report.audioDuring?.muted === false && report.audioDuring?.paused === false });
    }
    await page.evaluate(() => document.querySelector('video').pause());
    await page.waitForTimeout(300);
    await page.evaluate(async () => {
      const video = document.querySelector('video');
      video.currentTime = Math.min(Math.max(0, video.duration - 1), video.currentTime + 2);
      await video.play();
    });
    await page.waitForTimeout(1000);
    report.afterSeek = await send({ type: 'ANIMESCALER_STATUS' });
    report.afterSeekSource = await page.evaluate(() => ({ paused: document.querySelector('video').paused,
      currentTime: document.querySelector('video').currentTime }));
    report.checks.push({ name: 'pause-seek-resume', passed: report.afterSeek?.state === 'running' &&
      report.afterSeekSource.paused === false && report.afterSeekSource.currentTime > 0 });
    if (expectAudio) report.audioAfterSeek = await readAudio();
    report.stop = await send({ type: 'ANIMESCALER_STOP' });
    report.checks.push({ name: 'stop', passed: report.stop?.state === 'stopped' });
    report.state = report.errors.length || report.checks.some(check => !check.passed) ? 'failed' : 'complete';
  } catch (error) { report.state = 'failed'; report.errors.push(error.message); }
  finally { await context.close(); fs.writeFileSync(output, JSON.stringify(report, null, 2), { flag: 'wx' }); }
  console.log(JSON.stringify({ state: report.state, checks: report.checks, errors: report.errors }));
  if (report.state !== 'complete') process.exitCode = 1;
})();
