import { resolveInputSize, resolveOutputSize, resolveFsrInputSize } from '../core/settings.js';
import { ADDED_MODELS } from '../core/local-models.js';
import { DiagnosticLog } from '../core/diagnostic-log.js';
import { FramePacer } from '../core/frame-pacer.js';
import { profileKey, restoreProfile, ProfileSamples } from '../core/performance-profile.js';
import { sanitizeNativeStats } from '../content/native-host-client.js';
import { nativeServiceFailure } from './native-service-contract.js';
import { sessionTag } from '../core/session-diagnostics.js';

const MAX_PNG_BYTES = 16 * 1024 * 1024;
const nativeServiceLabel = backend => backend === 'opencl' ? 'OpenCL' : backend === 'tensorrt' ? 'TensorRT' : 'ROCm';
const WORKER_ERROR_CODES = new Set([
  'BACKEND_TIMEOUT', 'BACKEND_UNAVAILABLE', 'BACKEND_INVALID_RESPONSE',
  'NATIVE_SESSION_BUSY', 'FRAME_TOO_LARGE', 'INVALID_PNG_OUTPUT',
]);
function pngDataUrl(result, sequence, mediaTime, size) {
  const buffer = result?.pngBytes;
  if (buffer instanceof ArrayBuffer && buffer.byteLength > MAX_PNG_BYTES) throw new Error('FRAME_TOO_LARGE');
  const invalid = () => { throw new Error('INVALID_PNG_OUTPUT'); };
  if (!(buffer instanceof ArrayBuffer) || buffer.byteLength < 45 || result.mime !== 'image/png' ||
      result.sequence !== sequence || result.mediaTime !== mediaTime ||
      !Number.isSafeInteger(result.width) || result.width <= 0 || !Number.isSafeInteger(result.height) || result.height <= 0 ||
      result.width !== size.outputWidth || result.height !== size.outputHeight) invalid();
  const bytes = new Uint8Array(buffer), view = new DataView(buffer);
  if ([137,80,78,71,13,10,26,10].some((value,index) => bytes[index] !== value) ||
      view.getUint32(8) !== 13 || view.getUint32(12) !== 0x49484452 ||
      view.getUint32(16) !== result.width || view.getUint32(20) !== result.height) invalid();
  let offset = 8, hasPixels = false, ended = false;
  while (offset + 12 <= bytes.length) {
    const length = view.getUint32(offset), type = view.getUint32(offset + 4);
    if (offset + length + 12 > bytes.length) invalid();
    if (type === 0x49444154 && length > 0) hasPixels = true;
    offset += length + 12;
    if (type === 0x49454e44) { ended = length === 0 && offset === bytes.length; break; }
  }
  if (!hasPixels || !ended) invalid();
  // Chunk conversion avoids an argument-list overflow on large PNG frames.
  const chunks = [];
  for (let offset = 0; offset < bytes.length; offset += 32768) chunks.push(String.fromCharCode(...bytes.subarray(offset, offset + 32768)));
  return `data:image/png;base64,${btoa(chunks.join(''))}`;
}

export class NativeEngine {
  constructor() {
    this.diagnostics = new DiagnosticLog();
    this.pending = Promise.resolve(); this.requests = new Map(); this.nextId = 0;
    this.backend = 'rocm'; this.closed = false;
    this.generation = 0;
    this.presentationPacer = new FramePacer();
    this.presentationGeneration = 0;
    this.sourceSequence = 0;
    this.sessionTag = null;
    this.profileSamples = new ProfileSamples();
  }
  exclusive(operation) {
    const generation = this.generation;
    const result = this.pending.then(async () => {
      this.assertActive(generation);
      if (this.presentationFailure) throw this.presentationFailure;
      const value = await operation();
      this.assertActive(generation);
      return value;
    });
    this.pending = result.catch(() => {});
    return result;
  }
  assertActive(generation = this.generation) {
    if (this.closed || generation !== this.generation) throw new Error(`${nativeServiceLabel(this.backend)} encerrado.`);
  }
  call(type, data, transfer = []) {
    return new Promise((resolve, reject) => {
      this.assertActive();
      const id = ++this.nextId;
      const timer = setTimeout(() => { this.requests.delete(id); reject(new Error(`${nativeServiceLabel(this.backend)} excedeu o tempo de resposta.`)); }, 35000);
      this.requests.set(id, { resolve, reject, timer });
      try { this.worker.postMessage({ id, type, ...data }, transfer); }
      catch (error) { clearTimeout(timer); this.requests.delete(id); reject(error); }
    });
  }
  async initialize({ canvas, video, settings, adaptiveLevel = 0, sourceRect = null, connection = null, sessionId = null }) {
    const generation = this.generation;
    this.assertActive(generation);
    this.backend = settings.backend;
    this.settings = settings;
    this.sessionTag = sessionTag(sessionId ?? crypto.randomUUID());
    if (!connection) {
      const fsr3 = settings.backend === 'd3d12';
      const controller = this.initializeAbort = new AbortController();
      try {
        const response = await fetch(chrome.runtime.getURL(fsr3 ? 'native/fsr3-connection.json' : 'native/connection.json'), { cache: 'no-store', signal: controller.signal });
        this.assertActive(generation);
        if (!response.ok) throw new Error(fsr3 ? 'Abra Iniciar-FSR3.cmd antes de ativar o FSR 3.' : this.backend === 'opencl' ? 'Abra Iniciar-OpenCL.cmd antes de ativar Anime4KCPP.' : this.backend === 'tensorrt' ? 'Abra Iniciar-TensorRT.cmd antes de ativar TensorRT.' : 'Abra Iniciar-ROCm.cmd antes de ativar o ROCm.');
        connection = await response.json();
        this.assertActive(generation);
      } finally { if (this.initializeAbort === controller) this.initializeAbort = null; }
    }
    this.assertActive(generation);
    this.worker = new Worker(new URL('./native-worker.js', import.meta.url), { type: 'module' });
    this.worker.onmessage = ({ data }) => {
      if (this.closed || generation !== this.generation) return;
      const request = this.requests.get(data.id);
      if (!request) return;
      this.requests.delete(data.id); clearTimeout(request.timer);
      if (data.error) {
        const code = WORKER_ERROR_CODES.has(data.errorCode) ? data.errorCode : 'BACKEND_UNAVAILABLE';
        request.reject(Object.assign(new Error(data.error), { code }));
      } else request.resolve(data.result);
    };
    this.worker.onerror = () => {
      const failure = nativeServiceFailure(this.backend, 'disconnected', this.settings?.model);
      this.failRequests(Object.assign(new Error(failure.message), { code: failure.code }));
    };
    const offscreen = canvas.transferControlToOffscreen();
    const info = await this.call('init', { canvas: offscreen, connection, model:settings.model, backend:this.backend, transport:settings.nativeTransport }, [offscreen]);
    this.assertActive(generation);
    this.diagnostics.event('native_ready', { backend: this.backend, device: info.gpu, architecture: info.architecture,
      precision: info.precision, sessionTag: this.sessionTag });
    this.nativeInfo=info;
    this.profileId=profileKey(info,settings,sourceRect?.width??video.videoWidth,sourceRect?.height??video.videoHeight);
    if(settings.adaptive && globalThis.chrome?.storage?.local) {
      try { const saved=await chrome.storage.local.get(this.profileId);adaptiveLevel=restoreProfile(saved[this.profileId]); } catch { /* Storage is optional. */ }
      this.assertActive(generation);
    }
    await this.reconfigure(video, adaptiveLevel, sourceRect);
    this.assertActive(generation);
    if (settings.backend === 'tensorrt') await this.checkTensorRTProfile(generation);
  }
  async checkTensorRTProfile(generation=this.generation) {
    const shapes=this.tensorRTProfileShapes();
    const request={model:this.settings.model,shapes};
    let result;
    for(let attempt=0;attempt<3;attempt++) {
      try {
        result=await this.call('profile-check',request);
        break;
      } catch(error) {
        if(error?.code!=='NATIVE_SESSION_BUSY'||attempt===2)throw error;
        await new Promise(resolve=>setTimeout(resolve,50*(attempt+1)));
        this.assertActive(generation);
      }
    }
    this.assertActive(generation);
    if(result?.profileCheck!==true||result.supported!==true||result.model!==this.settings.model||
       JSON.stringify(result.shapes)!==JSON.stringify(shapes)||!Array.isArray(result.missingShapes))
      throw new Error('Resposta de perfil TensorRT inválida.');
    const requested=new Set(shapes.map(([width,height])=>`${width}x${height}`));
    const missing=result.missingShapes.map(shape=>Array.isArray(shape)&&shape.length===2&&
      Number.isSafeInteger(shape[0])&&Number.isSafeInteger(shape[1])?`${shape[0]}x${shape[1]}`:null);
    if(missing.some(shape=>!shape||!requested.has(shape))||new Set(missing).size!==missing.length||
       result.available!==(missing.length===0))throw new Error('Resposta de perfil TensorRT inválida.');
    if(missing.length) {
      const commands=missing.map(shape=>`--shape ${shape}`).join(' ');
      throw new Error(`Perfis TensorRT ausentes para ${this.settings.model}: ${missing.join(', ')}. Execute native/prepare_tensorrt.py ${commands} antes de iniciar a reprodução.`);
    }
  }
  tensorRTProfileShapes() {
    const {sourceWidth,sourceHeight,inputWidth,inputHeight}=this.currentSize;
    const shapes=[],seen=new Set();
    const add=(width,height)=>{
      const key=`${width}x${height}`;
      if(!seen.has(key)){seen.add(key);shapes.push([width,height]);}
    };
    add(inputWidth,inputHeight);
    if(this.settings.adaptive) {
      const zenkaiRealPair=this.settings.model==='zenkai-realesr-clean5k'||this.settings.model==='zenkai-animevideo-30k';
      const inputPreset=zenkaiRealPair&&['1080p','1440p'].includes(this.settings.preset)?'native':this.settings.preset;
      for(let level=0;level<8;level++) {
        const size=resolveInputSize(sourceWidth,sourceHeight,inputPreset,level);
        add(size.width,size.height);
      }
    }
    return shapes;
  }
  configure(video, level, rect) {
    this.assertActive();
    const sourceWidth = rect?.width ?? video.videoWidth;
    const sourceHeight = rect?.height ?? video.videoHeight;
    const nextProfileId=profileKey(this.nativeInfo??{},this.settings,sourceWidth,sourceHeight);
    if(this.profileId!==nextProfileId) {
      this.profileId=nextProfileId;this.profileSamples=new ProfileSamples();this.savedProfileLevel=undefined;
    }
    // At high-quality native presets, keep the complete source through the
    // model so line art is not discarded before inference. The final display
    // resize happens after the x2 result. 4K keeps its explicit 1080p input
    // geometry, including the 720p-source-to-1080p preparation path.
    const zenkaiRealPair = this.settings.model === 'zenkai-realesr-clean5k' || this.settings.model === 'zenkai-animevideo-30k';
    const highQualitySource = zenkaiRealPair && ['1080p', '1440p'].includes(this.settings.preset);
    const inputPreset = highQualitySource ? 'native' : this.settings.preset;
    let { width, height } = resolveInputSize(sourceWidth, sourceHeight, inputPreset, level);
    const output = resolveOutputSize(sourceWidth,sourceHeight,this.settings.preset,width,height);
    if (this.settings.model === 'fsr3-native') {
      const input=resolveFsrInputSize(sourceWidth,sourceHeight,output.width,output.height,this.settings.fsrQuality);
      const bound=Math.min(1,1920/input.width,1920/input.height,Math.sqrt(1920*1080/(input.width*input.height)));
      width=Math.max(40,Math.floor(Math.min(input.width*bound,level ? width : Infinity)/4)*4);
      height=Math.max(40,Math.floor(width/sourceWidth*sourceHeight/4)*4);
    }
    // Anime4KCPP is a native 2x worker. For 4K, feed it exactly half of the
    // requested output so the OpenCL model writes the final dimensions and we
    // avoid the expensive CPU resize from a smaller native result.
    if (this.settings.model.startsWith('anime4kcpp-') && this.settings.preset === '4k' && level === 0) {
      width = Math.floor(output.width / 8) * 4;
      height = Math.floor(output.height / 8) * 4;
    }
    if (Math.min(width, height) < 40 || width * height > 1920 * 1080 || Math.max(width, height) > 1920) throw new Error('Resolução fora do limite do Adore ROCm.');
    clearTimeout(this.presentationWake);
    this.presentationGeneration++;
    this.presentationPacer.reset();
    this.diagnostics.flush();
    this.currentSize = { sourceWidth, sourceHeight, inputWidth: width, inputHeight: height,
      modelOutputWidth: this.settings.model==='fsr3-native' ? output.width : width * (ADDED_MODELS[this.settings.model]?.scale??2), modelOutputHeight: this.settings.model==='fsr3-native' ? output.height : height * (ADDED_MODELS[this.settings.model]?.scale??2),
      outputWidth: output.width, outputHeight: output.height, adaptiveLevel: level };
  }
  async changeConfiguration(operation) {
    const generation = this.generation;
    this.assertActive(generation);
    // A dispatched worker presentation owns the old geometry until its reply.
    // Pending capture has no worker side effects and is invalidated by configure.
    this.reconfiguring=true;
    try {
      // A timeout does not prove that a queued worker draw has stopped.
      // Preserve geometry and propagate failure instead of bypassing the barrier.
      await this.presentationFlight;
      this.assertActive(generation);
      operation();
    } finally {this.reconfiguring=false;}
  }
  reconfigure(video, level = 0, rect = null) {
    return this.exclusive(() => this.changeConfiguration(() => this.configure(video, level, rect)));
  }
  updateSettings(settings, video, level = 0, rect = null) {
    return this.exclusive(() => this.changeConfiguration(() => {
      this.settings = settings;
      this.profileId=profileKey(this.nativeInfo??{},settings,rect?.width??video.videoWidth,rect?.height??video.videoHeight);
      this.profileSamples=new ProfileSamples();
      this.savedProfileLevel=undefined;
      this.configure(video, level, rect);
    }));
  }
  needsReconfigure(video, level = 0, rect = null) {
    return !this.currentSize || this.currentSize.sourceWidth !== (rect?.width ?? video.videoWidth)
      || this.currentSize.sourceHeight !== (rect?.height ?? video.videoHeight) || this.currentSize.adaptiveLevel !== level;
  }
  async processImage(bitmap, { sequence, mediaTime, crop } = {}) {
    try {
      return await this.exclusive(async () => {
        const generation = this.generation;
        if (!bitmap || !Number.isSafeInteger(sequence) || sequence < 0 || !Number.isFinite(mediaTime) || mediaTime < 0 ||
            !Number.isSafeInteger(bitmap.width) || bitmap.width <= 0 || !Number.isSafeInteger(bitmap.height) || bitmap.height <= 0 ||
            !this.currentSize) throw new Error('INVALID_PNG_INPUT');
        crop ??= { x: 0, y: 0, width: bitmap.width, height: bitmap.height };
        if (![crop.x, crop.y, crop.width, crop.height].every(Number.isFinite) || crop.x < 0 || crop.y < 0 ||
            crop.width <= 0 || crop.height <= 0 || crop.x + crop.width > bitmap.width || crop.y + crop.height > bitmap.height) {
          throw new Error('INVALID_PNG_INPUT');
        }
        // A presentation already dispatched still owns the shared host canvas.
        clearTimeout(this.presentationWake);
        this.stillPresentation = true;
        this.inlineOutputBusy = true;
        try {
          await this.presentationFlight;
          this.assertActive(generation);
          const size = this.currentSize;
          const dispatched = performance.now();
          const result = await this.call('frame', { bitmap, crop, size, sequence, mediaTime,
            sourceSequence: ++this.sourceSequence, fluidVideo: !!this.settings.fluidVideo, pngOutput: true }, [bitmap]);
          this.assertActive(generation);
          const workerRoundTripMs = performance.now() - dispatched;
          const dataUrl = pngDataUrl(result, sequence, mediaTime, size);
          const output = { sequence, mediaTime, width: result.width, height: result.height, mime: 'image/png', dataUrl };
          const measured = sanitizeNativeStats(result);
          // Older workers may return only image bytes. Preserve their PNG contract
          // without converting unavailable measurements into zero or false.
          if (Object.keys(measured).length) {
            const opencl = this.settings.model.startsWith('anime4kcpp-');
            const fsr3 = this.settings.model === 'fsr3-native';
            output.stats = sanitizeNativeStats({ ...measured, ...size, workerRoundTripMs,
              ...(Number.isFinite(measured.workerMs) && measured.workerMs <= workerRoundTripMs
                ? { workerOverheadMs: workerRoundTripMs - measured.workerMs } : {}),
              backend: opencl ? 'opencl' : this.nativeInfo?.backend ?? this.backend,
              model: this.settings.model,
              modelPrecision: opencl || fsr3 ? 'native' : this.nativeInfo?.precision });
          }
          return output;
        } finally { this.inlineOutputBusy = false; }
      });
    } catch (error) {
      // Failed or timed-out output cannot leave a worker drawing into this session.
      void this.destroy();
      throw error;
    } finally { bitmap?.close(); }
  }
  processFrame(video, rect = null, {still = false} = {}) {
    this.stillPresentation=still;
    clearTimeout(this.presentationWake);
    return this.exclusive(async () => {
      const started = performance.now();
      const sourceSequence=++this.sourceSequence;
      // A VideoFrame can retain the decoder's backing surface; avoid making
      // an ImageBitmap copy when the browser supports transferable frames.
      const bitmap = await this.makeFrame(video);
      const bitmapMs = performance.now() - started;
      const crop = rect ?? { x: 0, y: 0, width: bitmap.displayWidth ?? bitmap.width, height: bitmap.displayHeight ?? bitmap.height };
      let timings;
      const dispatched=performance.now();
      try { timings = await this.call('frame', { bitmap, crop, size: this.currentSize, sourceSequence, fluidVideo:!still && this.settings.fluidVideo }, [bitmap]); }
      catch (error) { bitmap.close(); throw error; }
      timings.sourceFrameMs=bitmapMs;
      timings.workerRoundTripMs=performance.now()-dispatched;
      if(Number.isFinite(timings.workerMs))timings.workerOverheadMs=Math.max(0,timings.workerRoundTripMs-timings.workerMs);
      // rVFC may stop on paused/static capture. Wake the same presenter only
      // when its clock is idle; normal playback keeps a single drawing cadence.
      if (!still && !this.stillPresentation && this.settings.fluidVideo && timings.temporal &&
          performance.now()-(this.lastPresentationAt??-Infinity)>80) {
        const presented=await this.presentFrame(video,crop);
        if (presented && this.lastPresentationMetrics) timings.temporal=this.lastPresentationMetrics;
      }
      if (!still && !this.stillPresentation && this.settings.fluidVideo && timings.temporal) {
        // The last regular callback may have happened just before completion.
        // A one-shot idle check covers that pause without a second active clock.
        this.presentationWake=setTimeout(()=>{
          if (!this.closed && !this.stillPresentation &&
              performance.now()-(this.lastPresentationAt??-Infinity)>80) {
            void this.presentFrame(video,crop).catch(error=>this.diagnostics.error?.(error));
          }
        },90);
      }
      timings.captureMs += bitmapMs;
      const openclModel = this.settings.model.startsWith('anime4kcpp-');
      const fsr3Model = this.settings.model === 'fsr3-native';
      const stats = { ...this.currentSize, ...timings, totalMs: performance.now() - started,
        backend: openclModel ? 'opencl' : this.backend,
        modelPrecision: openclModel || fsr3Model ? 'native' : 'fp16', model:this.settings.model,
        modelLabel: `${ADDED_MODELS[this.settings.model]?.label??this.settings.model} ${openclModel ? 'OpenCL' : this.backend}`,
        fsrQuality:fsr3Model ? this.settings.fsrQuality : undefined,
        layout: openclModel || fsr3Model ? 'RGBA' : 'NCHW' };
      if (!stats.warmup) this.diagnostics.frame(stats);
      if(stats.temporal && performance.now()-(this.lastTemporalLogAt??0)>=5000) {
        this.lastTemporalLogAt=performance.now();
        this.diagnostics.event('temporal_history',{
          sceneCuts:stats.temporal.sceneCuts,rejectedResults:stats.temporal.rejectedResults,
          acceptedResults:stats.temporal.acceptedResults,code:stats.temporal.lastRejected??'ok',
          presentedFrames:stats.temporal.presentedFrames,lowConfidenceFrames:stats.temporal.lowConfidenceFrames,
          aiCoverage:stats.temporal.aiCoverage,historyAgeMs:stats.temporal.historyAgeMs,
          motionCoverage:stats.temporal.motionCoverage,confidenceCoverage:stats.temporal.confidenceCoverage,
          temporalWeight:stats.temporal.temporalWeight,unchangedSource:stats.temporal.unchangedSource
        });
      }
      if(!stats.warmup && !stats.reused && this.settings.adaptive && globalThis.chrome?.storage?.local) {
        const profile=this.profileSamples.record(stats.totalMs,stats.adaptiveLevel,this.settings.maxFps);
        if(profile && this.savedProfileLevel!==profile.level) {
          this.savedProfileLevel=profile.level;
          chrome.storage.local.set({[this.profileId]:profile}).catch(()=>{this.savedProfileLevel=undefined;});
        }
      }
      return stats;
    });
  }
  failRequests(error) {
    for (const request of this.requests.values()) { clearTimeout(request.timer); request.reject(error); }
    this.requests.clear();
  }
  async makeFrame(video) {
    if (typeof VideoFrame === 'function' && video.videoWidth>0) {
      try { return new VideoFrame(video,{timestamp:Math.round(video.currentTime*1e6)}); } catch { /* Compatibility path. */ }
    }
    return createImageBitmap(video);
  }
  async presentFrame(video,crop) {
    if (this.presentationFailure) throw this.presentationFailure;
    if (!this.settings?.fluidVideo || !this.currentSize || this.presentBusy || this.closed || this.reconfiguring || this.inlineOutputBusy) return false;
    // Animation sources are commonly 24/30 FPS. Bound composition separately
    // from inference so 60 Hz tab capture does not double thumbnail readbacks.
    if(!this.presentationPacer.take(performance.now(),30))return false;
    this.presentBusy=true;
    const generation=this.presentationGeneration,size=this.currentSize,sourceSequence=++this.sourceSequence;
    let bitmap;
    try {
      bitmap=await this.makeFrame(video);
      if (this.closed || this.reconfiguring || this.inlineOutputBusy || generation!==this.presentationGeneration) return false;
      this.presentationFlight=this.call('present',{bitmap,crop,size,sourceSequence},[bitmap]);
      const result=await this.presentationFlight;
      if (this.closed || generation!==this.presentationGeneration || result.presented===false) return false;
      this.lastPresentationAt=performance.now();
      this.lastPresentationMetrics=result.temporal;
      this.diagnostics.present?.(result.temporal, { ...this.currentSize, model:this.settings.model, backend:this.backend });
      return true;
    } catch(error) {
      // Keep dispatch failure even after this flight and other queued work end.
      // Only destroying/recreating the engine can restore known worker state.
      if(this.presentationFlight)this.presentationFailure=error;
      throw error;
    } finally {bitmap?.close();this.presentationFlight=null;this.presentBusy=false;}
  }
  destroy() {
    if (this.destroying) return this.destroying;
    this.closed = true;
    this.generation++;
    this.presentationGeneration++;
    this.initializeAbort?.abort();
    clearTimeout(this.presentationWake);
    this.worker?.terminate(); this.failRequests(new Error(`${nativeServiceLabel(this.backend)} encerrado.`));
    this.destroying = this.pending.then(() => { this.diagnostics.flush(); });
    return this.destroying;
  }
}
