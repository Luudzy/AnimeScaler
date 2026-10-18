import { PresentationStats } from './presentation-stats.js';

export const DIAGNOSTIC_KEY = 'animeScalerDiagnostics';
const KEYS = new Set(('code message mode backend compositor fluidVideo exclusiveViewer captureMethod model preset maxFps adaptive precision modelPrecision layout sourceWidth sourceHeight inputWidth inputHeight modelOutputWidth modelOutputHeight outputWidth outputHeight adaptiveLevel frames elapsedMs fps processedFrames processedFps inferenceFrames inferenceFps reuseKnownFrames repeatKnownFrames repeatedFrames sourceCallbacks sourceFrames sourceElapsedMs sourceFps meanMs p50Ms p95Ms maxMs sessionRunMeanMs captureMeanMs transportMeanMs renderMeanMs vendor architecture device description shaderF16 timestampQuery sceneCuts rejectedResults acceptedResults presentedFrames lowConfidenceFrames aiCoverage historyAgeMs reusedFrames responseBytesTotal measuredFps targetFps minimumFps windowMs').split(' '));
for (const key of ['staleOrigin', 'sessionTag', 'messageType', 'sequence', 'expectedSequence', 'receivedSequence',
  'revision', 'expectedRevision', 'generation', 'closed', 'hasPending', 'hasInFlight']) KEYS.add(key);
const cleanText = value => String(value).replace(/(?:https?|blob|file|chrome-extension):\S+/gi, '[url]').slice(0, 500);
KEYS.add('fsrQuality');
KEYS.add('spatialDenoise');
KEYS.add('temporalHistoryAvailable');
KEYS.add('sourceMethod');
for (const key of ['presentationFps', 'aiCoverageMean', 'lowConfidenceRatio', 'historyAgeP95Ms']) KEYS.add(key);
for (const key of ['motionCoverage','confidenceCoverage','temporalWeight','unchangedSource',
  'motionCoverageMean','confidenceCoverageMean','temporalWeightMean',
  'sourceFrameMeanMs','workerMeanMs','workerRoundTripMeanMs','workerOverheadMeanMs']) KEYS.add(key);
const BOUNDARIES={sourceFrameMs:'sourceFrameMeanMs',workerMs:'workerMeanMs',
  workerRoundTripMs:'workerRoundTripMeanMs',workerOverheadMs:'workerOverheadMeanMs'};
function cleanData(data) {
  return Object.fromEntries(Object.entries(data ?? {}).filter(([key, value]) => KEYS.has(key)
    && (typeof value === 'string' || typeof value === 'boolean' || (typeof value === 'number' && Number.isFinite(value))))
    .map(([key, value]) => [key, typeof value === 'string' ? cleanText(value) : value]));
}
const blank = () => ({ schemaVersion: 1, dropped: 0, events: [] });

// Only the service worker writes. Its queue prevents lost updates across tabs.
export function createDiagnosticStore(storage, maxEvents = 1200) {
  let pending = Promise.resolve();
  const serial = operation => {
    const result = pending.then(operation);
    pending = result.catch(() => {});
    return result;
  };
  const read = async () => (await storage.get(DIAGNOSTIC_KEY))[DIAGNOSTIC_KEY] ?? blank();
  return {
    read: () => serial(read),
    clear: () => serial(() => storage.set({[DIAGNOSTIC_KEY]: blank()})),
    append: event => serial(async () => {
      const history = await read();
      history.events.push({time: new Date().toISOString(), type: cleanText(event.type).slice(0, 40),
        session: cleanText(event.session).slice(0, 80), data: cleanData(event.data)});
      const excess = Math.max(0, history.events.length - maxEvents);
      history.events.splice(0, excess);
      history.dropped += excess;
      // Bound bytes as well as count (storage.local is shared with settings).
      while (JSON.stringify(history).length > 500000 && history.events.length > 1) {
        history.events.shift(); history.dropped++;
      }
      await storage.set({[DIAGNOSTIC_KEY]: history});
    }),
  };
}

export class DiagnosticLog {
  constructor({ send = event => globalThis.chrome?.runtime?.sendMessage({type: 'ANIMESCALER_DIAGNOSTIC', event}),
    now = () => performance.now(), intervalMs = 5000 } = {}) {
    this.send = send; this.now = now; this.intervalMs = intervalMs;
    this.session = globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`;
    this.started = now(); this.samples = []; this.runMs = 0; this.context = null;
    this.inferenceFrames=0;this.reusedFrames=0;this.reuseKnownFrames=0;
    this.repeatedFrames=0;this.repeatKnownFrames=0;this.responseBytesTotal=0;
    this.stages = Object.fromEntries(['captureMs','transportMs','renderMs'].map(key=>[key,{sum:0,count:0}]));
    this.boundaries = {};
    this.sourceWindow={startedAt:null,callbacks:0,frames:0,lastMediaTime:null,context:null,contextKey:null};
    this.lastError = null; this.errorAt = -Infinity;
    this.presentation = new PresentationStats({ startedAt: now() });
  }
  event(type, data = {}) {
    if (type === 'start' || type === 'initializing') { this.flush(); this.started = this.now(); }
    try { Promise.resolve(this.send({session: this.session, type, data: cleanData(data)})).catch(() => {}); }
    catch { /* Diagnostics must never interrupt playback. */ }
  }
  error(error) {
    const data = {code: error?.code ?? error?.name ?? 'ERROR', message: error?.message ?? String(error),
      ...(error?.diagnostic ?? {})};
    const key = JSON.stringify(data);
    if (key === this.lastError && this.now() - this.errorAt < this.intervalMs) return;
    this.flush(); this.lastError = key; this.errorAt = this.now();
    this.event('error', data);
  }
  sourceFrame(mediaTime, timestamp = this.now(), context = {}) {
    if (!Number.isFinite(timestamp)) return false;
    const cleaned=cleanData(context),key=JSON.stringify(cleaned);
    if(this.sourceWindow.contextKey!==null&&key!==this.sourceWindow.contextKey) this.flushSource(timestamp);
    const window=this.sourceWindow;
    if(window.startedAt===null) {
      window.startedAt=timestamp;window.context=cleaned;window.contextKey=key;
    }
    window.callbacks++;
    if(Number.isFinite(mediaTime)&&mediaTime>=0&&mediaTime!==window.lastMediaTime) {
      window.frames++;window.lastMediaTime=mediaTime;
    }
    if(timestamp-window.startedAt>=this.intervalMs||window.callbacks>=600) this.flushSource(timestamp,true);
    return true;
  }
  flushSource(now=this.now(),continueWindow=false) {
    const window=this.sourceWindow;
    if(window.startedAt===null) return null;
    if(!window.callbacks) {
      if(!continueWindow) this.sourceWindow={startedAt:null,callbacks:0,frames:0,lastMediaTime:null,context:null,contextKey:null};
      return null;
    }
    const sourceElapsedMs=Math.max(1,now-window.startedAt);
    const data={...window.context,sourceCallbacks:window.callbacks,sourceElapsedMs,
      ...(window.frames?{sourceFrames:window.frames}:{}),
      ...(window.frames>=2&&sourceElapsedMs>=100?{sourceFps:window.frames*1000/sourceElapsedMs}:{})};
    this.event('source_rate',data);
    this.sourceWindow=continueWindow
      ? {startedAt:now,callbacks:0,frames:0,lastMediaTime:null,context:window.context,contextKey:window.contextKey}
      : {startedAt:null,callbacks:0,frames:0,lastMediaTime:null,context:null,contextKey:null};
    return data;
  }
  frame(stats) {
    const context = cleanData(stats);
    const key = JSON.stringify(context);
    if (key !== this.contextKey) { this.flush(); this.context = context; this.contextKey = key; }
    if (!Number.isFinite(stats.totalMs)) return;
    this.samples.push(stats.totalMs);
    if(typeof stats.reused==='boolean') {this.reuseKnownFrames++;if(stats.reused)this.reusedFrames++;}
    if(typeof stats.repeated==='boolean') {this.repeatKnownFrames++;if(stats.repeated)this.repeatedFrames++;}
    if(Number.isFinite(stats.responseBytes))this.responseBytesTotal+=stats.responseBytes;
    if(Number.isFinite(stats.inferenceMs)&&stats.inferenceMs>0&&stats.reused!==true&&stats.repeated!==true) {
      this.runMs+=stats.inferenceMs;this.inferenceFrames++;
    }
    for (const key of Object.keys(this.stages)) if(Number.isFinite(stats[key])&&stats[key]>=0) {
      this.stages[key].sum+=stats[key];this.stages[key].count++;
    }
    for(const [key,mean] of Object.entries(BOUNDARIES)) if(Number.isFinite(stats[key]) && stats[key]>=0) {
      const sample=this.boundaries[mean]??={sum:0,count:0};sample.sum+=stats[key];sample.count++;
    }
    if (this.now() - this.started >= this.intervalMs || this.samples.length >= 600) this.flush();
  }
  present(metrics, context = {}) {
    const now = this.now();
    const cleaned = cleanData(context), key = JSON.stringify(cleaned);
    if (key !== this.presentationContextKey) {
      this.flushPresentation(now);
      this.presentationContext = cleaned; this.presentationContextKey = key;
    }
    if (!this.presentation.record(metrics, now)) return;
    if (now - this.presentation.startedAt >= this.intervalMs || this.presentation.frames >= 600) this.flushPresentation(now);
  }
  flushPresentation(now = this.now()) {
    const summary = this.presentation.flush(now);
    if (summary) this.event('presentation', { ...this.presentationContext, ...summary });
  }
  flush() {
    const now = this.now();
    if (this.samples.length) {
      const sorted = this.samples.sort((a, b) => a - b), frames = sorted.length;
      const elapsedMs = Math.max(1, now - this.started);
      const processedFps=frames*1000/elapsedMs;
      const stageMeans=Object.fromEntries(Object.entries(this.stages).filter(([,value])=>value.count)
        .map(([key,value])=>[key.replace('Ms','MeanMs'),value.sum/value.count]));
      this.event('performance', {...this.context,
        ...Object.fromEntries(Object.entries(this.boundaries).map(([key,value])=>[key,value.sum/value.count])),
        frames, elapsedMs, fps: processedFps,processedFrames:frames,processedFps,
        ...(this.inferenceFrames?{inferenceFrames:this.inferenceFrames,inferenceFps:this.inferenceFrames*1000/elapsedMs,
          sessionRunMeanMs:this.runMs/this.inferenceFrames}:{}),
        ...(this.reuseKnownFrames?{reusedFrames:this.reusedFrames,reuseKnownFrames:this.reuseKnownFrames}:{}),
        ...(this.repeatKnownFrames?{repeatedFrames:this.repeatedFrames,repeatKnownFrames:this.repeatKnownFrames}:{}),
        responseBytesTotal:this.responseBytesTotal,
        ...stageMeans,meanMs: sorted.reduce((a, b) => a + b, 0) / frames, p50Ms: sorted[Math.ceil(frames * .5) - 1],
        p95Ms: sorted[Math.ceil(frames * .95) - 1], maxMs: sorted[frames - 1],
      });
      this.started = now;
    }
    this.samples = []; this.runMs = 0;this.inferenceFrames=0;
    this.reusedFrames=0;this.reuseKnownFrames=0;this.repeatedFrames=0;this.repeatKnownFrames=0;this.responseBytesTotal=0;
    this.stages = Object.fromEntries(['captureMs','transportMs','renderMs'].map(key=>[key,{sum:0,count:0}]));
    this.boundaries = {};
    this.flushSource(now);
    this.flushPresentation(now);
  }
}
