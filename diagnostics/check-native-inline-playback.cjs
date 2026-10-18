const fs = require('node:fs');
const path = require('node:path');
const { browserRuntime } = require('./browser-runtime.cjs');
const { chromium, executablePath } = browserRuntime();

const root = path.resolve(__dirname, '..');
const output = path.resolve(root, process.env.ANIMESCALER_INLINE_OUTPUT ||
  `diagnostics/stability-20261015/native-inline-${Date.now()}`);
const media = process.env.ANIMESCALER_INLINE_MEDIA || '/diagnostics/stability-media/rezero-ep19-360p.mp4';
const model = process.env.ANIMESCALER_INLINE_MODEL || 'adore-lite-v3';
const backend = process.env.ANIMESCALER_INLINE_BACKEND || 'rocm';
const durationMs = Number(process.env.ANIMESCALER_INLINE_DURATION_MS || 60000);
const syntheticAudio = process.env.ANIMESCALER_INLINE_SYNTHETIC_AUDIO === 'true';
const sourceAudio = process.env.ANIMESCALER_INLINE_SOURCE_AUDIO === 'true';
if (fs.existsSync(output)) throw new Error('Preserve prior playback output; choose a new output directory');
fs.mkdirSync(output, { recursive: true });

(async () => {
  let context;
  const report = { state: 'running', scope: 'Unpacked extension direct-video path, local video, live native service',
    audioFixture: syntheticAudio ? '440 Hz Web Audio tone routed to the page output plus a live synthetic audio element track'
      : sourceAudio ? 'AAC audio decoded from the source video element and measured through Web Audio' : 'none',
    model, durationMs, samples: [], errors: [], checks: {} };
  try {
    context = await chromium.launchPersistentContext(path.join(output, 'profile'), {
      headless: true, executablePath, viewport: { width: 1280, height: 800 },
      args: [`--disable-extensions-except=${root}`, `--load-extension=${root}`,
        '--enable-unsafe-webgpu', '--autoplay-policy=no-user-gesture-required'],
    });
    const worker = context.serviceWorkers()[0] || await context.waitForEvent('serviceworker', { timeout: 15000 });
    const page = await context.newPage();
    page.on('pageerror', error => report.errors.push(error.message));
    await page.goto('http://127.0.0.1:8765/diagnostics/playback.html');
    await page.evaluate(async ({ source, syntheticAudio, sourceAudio }) => {
      const video = document.querySelector('#source');
      video.removeAttribute('muted');
      video.muted = !sourceAudio;
      video.volume = 1;
      video.loop = true;
      video.src = source;
      if (!syntheticAudio && !sourceAudio) { await video.play(); return; }

      const audioContext = new AudioContext();
      const signalAnalyser = audioContext.createAnalyser();
      signalAnalyser.fftSize = 2048;
      let audioTrack = null;
      let audio = null;
      if (sourceAudio) {
        audioContext.createMediaElementSource(video).connect(signalAnalyser);
        signalAnalyser.connect(audioContext.destination);
      }
      if (syntheticAudio) {
        const oscillator = audioContext.createOscillator();
        const gain = audioContext.createGain();
        const audioDestination = audioContext.createMediaStreamDestination();
        audioTrack = audioDestination.stream.getAudioTracks()[0];
        audio = document.createElement('audio');
        audio.autoplay = true;
        audio.srcObject = audioDestination.stream;
        document.body.append(audio);
        gain.gain.value = 0.04;
        oscillator.frequency.value = 440;
        oscillator.connect(gain);
        gain.connect(audioDestination);
        gain.connect(signalAnalyser);
        signalAnalyser.connect(audioContext.destination);
        oscillator.start();
        if (!audioTrack) throw new Error('SYNTHETIC_AUDIO_TRACK_UNAVAILABLE');
      }
      await audioContext.resume();
      await video.play();
      if (audio) await audio.play();
      const samples = new Float32Array(signalAnalyser.fftSize);
      window.__audioProbe = {
        track: audioTrack,
        element: audio,
        measure() {
          signalAnalyser.getFloatTimeDomainData(samples);
          const rms = Math.sqrt(samples.reduce((sum, sample) => sum + sample * sample, 0) / samples.length);
          return { rms, trackState: audioTrack?.readyState ?? null, contextState: audioContext.state,
            elementPaused: audio?.paused ?? video.paused, elementMuted: audio?.muted ?? video.muted,
            sourceReadyState: video.readyState, currentTime: video.currentTime };
        },
      };
    }, { source: media, syntheticAudio, sourceAudio });
    await page.waitForFunction(() => document.querySelector('#source')?.readyState >= 2);
    const tabId = await worker.evaluate(async url => (await chrome.tabs.query({ url }))[0]?.id,
      'http://127.0.0.1:8765/diagnostics/playback.html');
    if (!Number.isInteger(tabId)) throw new Error('TEST_TAB_NOT_FOUND');
    const send = message => worker.evaluate(({ id, value }) =>
      chrome.tabs.sendMessage(id, value, { frameId: 0 }), { id: tabId, value: message });
    const settings = { backend, model, enabled: true, preset: '720p', maxFps: 24,
      adaptive: false, fluidVideo: process.env.ANIMESCALER_INLINE_FLUID_VIDEO === 'true', showHud: false };
    report.settings = settings;
    const start = await send({ type: 'ANIMESCALER_START', settings });
    report.start = start;
    const startedAt = Date.now();
    while (Date.now() - startedAt < durationMs) {
      await new Promise(resolve => setTimeout(resolve, 5000));
      const [status, source] = await Promise.all([
        send({ type: 'ANIMESCALER_STATUS' }),
        page.evaluate(() => ({ currentTime: document.querySelector('#source').currentTime,
          paused: document.querySelector('#source').paused, muted: document.querySelector('#source').muted,
          audio: window.__audioProbe?.measure() ?? null,
          overlayDisplay: document.querySelector('canvas[data-animescaler="canvas"]')?.style.display ?? null })),
      ]);
      report.samples.push({ atMs: Date.now() - startedAt, state: status.state, error: status.error?.code,
        stats: status.stats ?? null, source });
    }
    report.checks.sourcePlaybackContinued = report.samples.length > 0 &&
      report.samples.every(sample => !sample.source.paused) &&
      report.samples.at(-1).source.currentTime > report.samples[0].source.currentTime;
    report.checks.sourceElementUnmuted = report.samples.length > 0 &&
      report.samples.every(sample => !sample.source.muted);
    report.checks.audioSignalContinued = syntheticAudio
      ? report.samples.length > 0 && report.samples.at(-1)?.source.audio?.trackState === 'live' &&
        report.samples.at(-1)?.source.audio?.contextState === 'running' && report.samples.at(-1)?.source.audio?.rms > 0.005
        && report.samples.at(-1)?.source.audio?.elementPaused === false && report.samples.at(-1)?.source.audio?.elementMuted === false
      : sourceAudio ? report.samples.length > 0 &&
        report.samples.every(sample => sample.source.audio?.contextState === 'running' &&
          sample.source.audio?.elementPaused === false && sample.source.audio?.elementMuted === false &&
          sample.source.audio?.sourceReadyState >= 2) &&
        report.samples.filter(sample => sample.source.audio?.rms > 0.001).length >= 2
      : null;
    report.checks.inlineSessionStayedRunning = report.samples.length > 0 &&
      report.samples.every(sample => sample.state === 'running');
    const guardEvent = (await worker.evaluate(async () => {
      const data = await chrome.storage.local.get('animeScalerDiagnostics');
      return (data.animeScalerDiagnostics?.events ?? []).findLast(event => event.type === 'throughput_guard') ?? null;
    }));
    report.throughputGuard = guardEvent?.data ?? null;
    report.checks.throughputGuardStoppedEnhancement = Boolean(guardEvent) &&
      report.samples.at(-1)?.state === 'error' && report.samples.at(-1)?.error === 'THROUGHPUT_BELOW_TARGET';
    report.checks.enhancementHiddenAfterGuard = report.checks.throughputGuardStoppedEnhancement &&
      report.samples.at(-1)?.source.overlayDisplay === 'none';
    report.checks.sessionBehaviorExpected = report.checks.inlineSessionStayedRunning ||
      (report.checks.throughputGuardStoppedEnhancement && report.checks.enhancementHiddenAfterGuard);
    report.checks.processingProducedStats = report.samples.some(sample => sample.stats &&
      Number.isFinite(sample.stats.totalMs ?? sample.stats.inferenceMs ?? sample.stats.fps));
    const readDiagnostics = async () => worker.evaluate(async () => {
      const data = await chrome.storage.local.get('animeScalerDiagnostics');
      return data.animeScalerDiagnostics?.events ?? [];
    });
    report.runtimeDiagnostics = await readDiagnostics();
    report.runtimeErrors = report.runtimeDiagnostics.filter(event => event.type === 'error');
    report.performance = report.runtimeDiagnostics.filter(event => event.type === 'performance')
      .map(event => ({ time: event.time, ...event.data }));
    await send({ type: 'ANIMESCALER_STOP' });
    report.finalDiagnostics = await readDiagnostics();
    report.teardownEvents = report.finalDiagnostics.slice(report.runtimeDiagnostics.length);
    report.nativeReady = report.runtimeDiagnostics.some(event => event.type === 'native_ready' &&
      event.data?.backend === backend);
    report.errors.push(...report.runtimeErrors.filter(event => event.data?.code !== 'THROUGHPUT_BELOW_TARGET')
      .map(event => `${event.data?.code ?? 'UNKNOWN'}:${event.data?.message ?? ''}`));
    report.state = report.checks.sourcePlaybackContinued && report.checks.sourceElementUnmuted &&
      report.checks.sessionBehaviorExpected && report.checks.processingProducedStats &&
      (!syntheticAudio || report.checks.audioSignalContinued) && report.nativeReady && report.errors.length === 0
      ? 'complete' : 'failed';
  } catch (error) {
    report.state = 'failed';
    report.errors.push(error.stack || error.message);
  } finally {
    await context?.close();
    report.finishedAt = new Date().toISOString();
    fs.writeFileSync(path.join(output, 'report.json'), JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
  }
  console.log(JSON.stringify({ state: report.state, checks: report.checks,
    samples: report.samples.length, nativeReady: report.nativeReady, errors: report.errors }));
  if (report.state !== 'complete') process.exitCode = 1;
})().catch(error => { console.error(error.message); process.exitCode = 1; });
