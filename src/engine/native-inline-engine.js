import { NativeHostClient, nativePngDimensionsError, sanitizeNativeStats } from '../content/native-host-client.js';
import { sessionDiagnostic } from '../core/session-diagnostics.js';

function errorWithCode(code, cause, diagnostic) {
  const error = new Error(code, { cause });
  error.code = code;
  if (diagnostic && Object.keys(diagnostic).length) error.diagnostic = diagnostic;
  return error;
}

function staleError(engine, messageType, details = {}) {
  return errorWithCode('STALE_SESSION', undefined, sessionDiagnostic('inline_engine', engine.sessionId, {
    messageType, generation: engine.generation, closed: engine.closed, ...details,
  }));
}

export class NativeInlineEngine {
  constructor({ createClient = () => new NativeHostClient(), createCanvas = () => document.createElement('canvas') } = {}) {
    this.createClient = createClient;
    this.createCanvas = createCanvas;
    this.client = null;
    this.pending = Promise.resolve();
    this.sequence = 0;
    this.sessionId = null;
    this.generation = 0;
    this.closed = false;
    this.destroying = null;
    this.operationCount = 0;
  }

  exclusive(operation, messageType = 'UNKNOWN') {
    if (this.closed) return Promise.reject(staleError(this, messageType));
    // One active operation and one waiting operation; capture starts only on dispatch.
    if (this.operationCount >= 2) return Promise.reject(errorWithCode('NATIVE_SESSION_BUSY'));
    this.operationCount++;
    const generation = this.generation;
    const result = this.pending.then(async () => {
      if (this.closed || generation !== this.generation) throw staleError(this, messageType);
      return operation(generation);
    }).finally(() => { this.operationCount--; });
    this.pending = result.catch(() => {});
    return result;
  }

  capture(video) {
    const width = video?.videoWidth;
    const height = video?.videoHeight;
    if (!Number.isSafeInteger(width) || width <= 0 || !Number.isSafeInteger(height) || height <= 0) {
      throw errorWithCode('VIDEO_NOT_READY');
    }
    const sizeError = nativePngDimensionsError(width, height);
    if (sizeError) throw errorWithCode(sizeError);
    if (this.captureCanvas.width !== width || this.captureCanvas.height !== height) {
      this.captureCanvas.width = width;
      this.captureCanvas.height = height;
      this.captureContext = this.captureCanvas.getContext('2d', { willReadFrequently: true });
    }
    try {
      this.captureContext.drawImage(video, 0, 0, width, height);
      return { width, height, mime: 'image/png', dataUrl: this.captureCanvas.toDataURL('image/png') };
    } catch (error) {
      const code = error?.name === 'SecurityError'
        ? (this.captureMode === 'direct' ? 'CROSS_ORIGIN_VIDEO' : 'SOURCE_NOT_ISOLATED')
        : 'SOURCE_NOT_ISOLATED';
      throw errorWithCode(code, error);
    }
  }

  async initialize({ video, settings, captureMode = 'direct', presenter, adaptiveLevel = 0 }) {
    if (this.client || this.closed) throw errorWithCode('NATIVE_SESSION_BUSY');
    this.video = video;
    this.settings = settings;
    this.captureMode = captureMode;
    this.presenter = presenter;
    this.adaptiveLevel = adaptiveLevel;
    try {
      this.captureCanvas = this.createCanvas();
      this.captureContext = this.captureCanvas.getContext('2d', { willReadFrequently: true });
      if (!this.captureContext) throw errorWithCode('CAPTURE_UNAVAILABLE');
      // Check origin access before the controller reports the session as running.
      const source = this.capture(video);
      this.sourceWidth = source.width;
      this.sourceHeight = source.height;
      const client = this.client = this.createClient();
      this.sessionId = crypto.randomUUID();
      await client.start({ sessionId: this.sessionId, backend: settings.backend, model: settings.model,
        settings, captureMode: captureMode === 'tab' ? 'restricted-png' : 'direct-png',
        sourceWidth: source.width, sourceHeight: source.height });
    } catch (error) {
      const client = this.client;
      this.client = null;
      try { await client?.stop('initialize-failed'); }
      catch (cleanupError) { error.cleanupError = cleanupError; }
      finally {
        if (this.captureCanvas) { this.captureCanvas.width = 1; this.captureCanvas.height = 1; }
        this.captureContext = null;
        this.presenter = null;
        this.video = null;
      }
      throw error;
    }
  }

  needsReconfigure(video, adaptiveLevel = this.adaptiveLevel) {
    return this.sourceWidth !== video?.videoWidth || this.sourceHeight !== video?.videoHeight ||
      this.adaptiveLevel !== adaptiveLevel;
  }

  reconfigure(video, adaptiveLevel = this.adaptiveLevel) {
    return this.updateSettings(this.settings, video, adaptiveLevel);
  }

  updateSettings(settings, video, adaptiveLevel = this.adaptiveLevel) {
    return this.exclusive(async generation => {
      const source = this.capture(video);
      const result = await this.client.configure({ settings, adaptiveLevel,
        sourceWidth: source.width, sourceHeight: source.height });
      if (this.closed || generation !== this.generation) throw staleError(this, 'CONFIGURE');
      this.settings = settings;
      this.sourceWidth = source.width;
      this.sourceHeight = source.height;
      this.adaptiveLevel = adaptiveLevel;
      return result;
    }, 'CONFIGURE');
  }

  processFrame(video) {
    return this.exclusive(async generation => {
      const started = performance.now();
      const source = this.capture(video);
      const captureMs = performance.now() - started;
      const sequence = ++this.sequence;
      const mediaTime = Number.isFinite(video.currentTime) ? Math.max(0, video.currentTime) : 0;
      const output = await this.client.processFrame({ ...source, sequence, mediaTime });
      const received = performance.now();
      if (this.closed || generation !== this.generation) throw staleError(this, 'FRAME_IN', { sequence });
      if (this.presenter) {
        const isCurrent = () => !this.closed && generation === this.generation;
        await this.presenter.presentFrame(output, isCurrent);
      }
      if (this.closed || generation !== this.generation) throw staleError(this, 'FRAME_IN', { sequence });
      const totalMs = performance.now() - started;
      const nativeStats = sanitizeNativeStats(output.stats);
      const stats = { ...nativeStats, totalMs, captureMs,
        ...(nativeStats.captureMs === undefined ? {} : { nativeCaptureMs: nativeStats.captureMs }),
        roundTripMs: Math.max(0, received - started - captureMs),
        presentationMs: Math.max(0, totalMs - (received - started)),
        backend: nativeStats.backend ?? this.settings.backend, model: nativeStats.model ?? this.settings.model,
        modelLabel: nativeStats.modelLabel ?? this.settings.model,
        sourceWidth: source.width, sourceHeight: source.height,
        outputWidth: output.width, outputHeight: output.height };
      this.diagnostics?.frame?.(stats);
      return stats;
    }, 'FRAME_IN');
  }

  destroy() {
    if (this.destroying) return this.destroying;
    this.closed = true;
    this.generation++;
    const client = this.client;
    this.client = null;
    const stopping = Promise.resolve().then(() => client?.stop('engine-destroy'));
    this.destroying = stopping.finally(() => this.pending).finally(() => {
      if (this.captureCanvas) { this.captureCanvas.width = 1; this.captureCanvas.height = 1; }
      this.captureContext = null;
      this.presenter = null;
      this.video = null;
    });
    return this.destroying;
  }
}
