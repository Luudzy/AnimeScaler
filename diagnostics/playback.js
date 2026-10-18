import { NativeEngine } from '../src/engine/native-engine.js';
import { createProcessingEngine } from '../src/engine/engine-factory.js';
import { normalizeSettings } from '../src/core/settings.js';
import { FramePacer } from '../src/core/frame-pacer.js';
import { analyzePresentation, summarize } from './playback-metrics.js';

const video = document.querySelector('#source');
const canvas = document.querySelector('#output');
const mean = values => values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;

window.runPlayback = async ({ model, media, seconds = 300, preset = '720p', fluidVideo = true, maxFps = 30, interventionsEnabled = true }) => {
  if (!Number.isFinite(seconds) || seconds < 1 || seconds > 1800) throw new Error('Playback duration must be 1–1800 seconds');
  if(![12,24,30].includes(maxFps))throw new Error('Playback inference limit must be 12, 24 or 30 FPS');
  const settings = normalizeSettings({ model, preset, fluidVideo, enabled:true, backend: 'rocm', adaptive: false, maxFps });
  video.src = media; video.loop = true;
  await video.play();
  const native = settings.backend === 'rocm';
  const engine = native ? new NativeEngine() : createProcessingEngine(settings);
  engine.diagnostics = { event() {}, frame() {}, flush() {}, error(error) { errors.push(error.message); } };
  const errors = [], startup = performance.now();
  const connection = native ? await fetch('/diagnostics/native-config').then(response => { if (!response.ok) throw new Error('Native connection unavailable'); return response.json(); }) : undefined;
  await engine.initialize({ canvas, video, settings, connection });
  const first = await engine.processFrame(video);
  const startupMs = performance.now() - startup;
  const started = performance.now(), qualityStart = video.getVideoPlaybackQuality();
  const timings = [], capture = [], transport = [], inference = [], render = [], coverage = [], age = [], presentationIntervals = [], processingIntervals = [];
  const boundaries=Object.fromEntries(['sourceFrameMs','workerMs','workerRoundTripMs','workerOverheadMs'].map(key=>[key,[]]));
  const presentationRows=[];
  const pacer = new FramePacer();
  let active = true, busy = false, latest = false, callbacks = 0, skippedWhileBusy = 0, handle, pendingProcess = Promise.resolve();
  const presentations=new Set();
  let lastProcessed = null, lastPresented = null, latestStats = first;
  let reusedFrames = 0, freshProcessedFrames = 0;
  const processing = () => {
    if (!active || busy || !latest || !pacer.take(performance.now(), settings.maxFps)) return;
    latest = false; busy = true;
    pendingProcess = (async () => {
      try {
        const row = await engine.processFrame(video);
        latestStats = row;
        if (!row.warmup) {
          timings.push(row.totalMs);
          for (const [key, samples] of [['captureMs',capture],['transportMs',transport],['inferenceMs',inference],['renderMs',render]]) {
            if (Number.isFinite(row[key])) samples.push(row[key]);
          }
          for(const [key,samples] of Object.entries(boundaries))if(Number.isFinite(row[key]))samples.push(row[key]);
          if (row.reused) reusedFrames++; else freshProcessedFrames++;
          const now = performance.now(); if (lastProcessed !== null) processingIntervals.push(now - lastProcessed); lastProcessed = now;
        }
      } catch (error) { errors.push(error.message); active = false; }
      finally { busy = false; if (active && latest) queueMicrotask(processing); }
    })();
  };
  const callback = () => {
    if (!active) return;
    callbacks++; if (busy) skippedWhileBusy++; latest = true;
    if (native && settings.fluidVideo) {
      const task = engine.presentFrame(video, { x: 0, y: 0, width: video.videoWidth, height: video.videoHeight }).then(presented => {
        if (!presented) return;
        const row = engine.lastPresentationMetrics;
        if (row) {
          coverage.push(row.aiCoverage); age.push(row.historyAgeMs);
          const sample={elapsedMs:performance.now()-started,mediaTimeSeconds:video.currentTime};
          for(const key of ['aiCoverage','historyAgeMs','motionCoverage','confidenceCoverage','temporalWeight'])
            if(Number.isFinite(row[key]))sample[key]=row[key];
          if(typeof row.unchangedSource==='boolean')sample.unchangedSource=row.unchangedSource;
          presentationRows.push(sample);
        }
        const now = performance.now(); if (lastPresented !== null) presentationIntervals.push(now - lastPresented); lastPresented = now;
      }).catch(error => { errors.push(error.message); active = false; });
      presentations.add(task);void task.finally(()=>presentations.delete(task));
    }
    processing(); handle = video.requestVideoFrameCallback(callback);
  };
  handle = video.requestVideoFrameCallback(callback);
  const interventions = [];
  const interventionTimer = interventionsEnabled ? setTimeout(async () => {
    try {
      video.pause(); interventions.push({ action: 'pause', atSeconds: (performance.now() - started) / 1000 });
      await new Promise(resolve => setTimeout(resolve, 700));
      video.currentTime = Math.min(video.duration - 2, video.currentTime + 10);
      await video.play(); interventions.push({ action: 'seek+10/resume', atSeconds: (performance.now() - started) / 1000 });
      canvas.style.width = '480px'; interventions.push({ action: 'display-resize', width: 480 });
    } catch (error) { errors.push(error.message); }
  }, Math.min(60000, seconds * 500)) : null;
  window.playbackProgress = () => ({ model, elapsedSeconds: (performance.now() - started) / 1000, processedFrames: timings.length, callbacks, aiCoverageMean: mean(coverage), errors: [...errors] });
  const end = started + seconds * 1000;
  while (active && performance.now() < end) await new Promise(resolve => setTimeout(resolve, 250));
  active = false; clearTimeout(interventionTimer); video.cancelVideoFrameCallback(handle);
  await Promise.allSettled([pendingProcess, ...presentations]);
  const elapsedMs = performance.now() - started, qualityEnd = video.getVideoPlaybackQuality();
  const result = {
    state: errors.length ? 'failed' : 'complete', scope: 'Decoded local HTMLVideoElement -> actual engine/worker/native service -> canvas. No tabCapture, remote site, or monitor scanout.',
    model, settings, startupMs, elapsedMs, source: { width: video.videoWidth, height: video.videoHeight, callbacks, decodedFrames: qualityEnd.totalVideoFrames - qualityStart.totalVideoFrames, droppedFrames: qualityEnd.droppedVideoFrames - qualityStart.droppedVideoFrames },
    processedFrames: timings.length, processedFps: timings.length * 1000 / elapsedMs, skippedWhileBusy,
    reusedFrames, freshProcessedFrames, freshProcessedFps:freshProcessedFrames * 1000 / elapsedMs,
    measurementNotes:['Processed FPS includes exact repeated frames reused by the native service; fresh processing is separate.', 'Missing stage timings have no samples and null summaries, rather than invented zero costs.', 'Canvas submission and hybrid contribution are measured; monitor scanout is not measured.'],
    frameMs: summarize(timings), captureMs: summarize(capture), transportIncludingServiceMs: summarize(transport), nativeServiceMs: summarize(inference), renderSubmissionMs: summarize(render),
    boundaries:Object.fromEntries(Object.entries(boundaries).map(([key,values])=>[key,summarize(values)])),
    contributionAnalysis:analyzePresentation(presentationRows),presentationRows:presentationRows.slice(0,1200),
    presentationRowsTruncated:Math.max(0,presentationRows.length-1200),
    processingIntervalsMs: summarize(processingIntervals), presentationIntervalsMs: summarize(presentationIntervals), presentedSamples: coverage.length,
    aiCoverage: summarize(coverage), historyAgeMs: summarize(age), latestTemporal: engine.lastPresentationMetrics || latestStats.temporal || null,
    dimensions: engine.currentSize, interventions, errors,
    nativeRuntime:engine.nativeInfo?Object.fromEntries(['backend','model','precision','gpu','architecture','hip'].map(key=>[key,engine.nativeInfo[key]])):null,
  };
  video.pause(); await engine.destroy();
  document.querySelector('#report').textContent = JSON.stringify(result, null, 2);
  return result;
};
