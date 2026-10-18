import { TEMPORAL_MAX_SOURCE_AGE_MS } from '../engine/temporal-budget.js';
import { isStaleOrigin, sessionDiagnostic } from '../core/session-diagnostics.js';

const PORT_NAME = 'animescaler-native-inline-v1';
const PNG_PREFIX = 'data:image/png;base64,';
const MAX_PNG_BYTES = 16 * 1024 * 1024;
export const MAX_NATIVE_PNG_AXIS = 8192;
export const MAX_NATIVE_PNG_PIXELS = 7680 * 4320;
// Freshness follows the temporal history budget; running inference keeps its own failure deadline.
export const MAX_NATIVE_FRAME_WAIT_MS = TEMPORAL_MAX_SOURCE_AGE_MS;

export function nativePngDimensionsError(width, height) {
  if (!Number.isSafeInteger(width) || width <= 0 || !Number.isSafeInteger(height) || height <= 0) {
    return 'SOURCE_NOT_ISOLATED';
  }
  return width > MAX_NATIVE_PNG_AXIS || height > MAX_NATIVE_PNG_AXIS ||
    width * height > MAX_NATIVE_PNG_PIXELS ? 'FRAME_TOO_LARGE' : null;
}

export function sanitizeNativeStats(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return {};
  const stats = {};
  for (const key of ['inferenceMs', 'captureMs', 'renderMs', 'transportMs', 'workerMs',
    'workerRoundTripMs', 'workerOverheadMs', 'responseBytes', 'adaptiveLevel',
    'inputWidth', 'inputHeight', 'outputWidth', 'outputHeight', 'sourceWidth', 'sourceHeight']) {
    if (Number.isFinite(value[key]) && value[key] >= 0) stats[key] = value[key];
  }
  for (const key of ['warmup', 'reused', 'repeated', 'fluidVideo']) {
    if (typeof value[key] === 'boolean') stats[key] = value[key];
  }
  const enums = {
    backend: ['rocm', 'tensorrt', 'opencl'], modelPrecision: ['native', 'fp16', 'fp32', 'int8'],
    layout: ['RGBA', 'NCHW', 'NHWC'], compositor: ['webgpu', 'canvas2d'],
    transport: ['repeat', 'rgb', 'jpeg98', 'rgba'], captureMethod: ['VideoFrame.copyTo', 'canvas'],
    sourceMethod: ['external-texture', 'texture-copy'],
  };
  for (const [key, allowed] of Object.entries(enums)) {
    if (allowed.includes(value[key])) stats[key] = value[key];
  }
  for (const key of ['model', 'modelLabel']) {
    if (typeof value[key] === 'string' && /^[\p{L}\p{N} _().+-]{1,128}$/u.test(value[key])) stats[key] = value[key];
  }
  if (value.temporal && typeof value.temporal === 'object' && !Array.isArray(value.temporal)) {
    const temporal = {};
    for (const key of ['sceneCuts', 'rejectedResults', 'acceptedResults', 'presentedFrames',
      'lowConfidenceFrames', 'aiCoverage', 'historyAgeMs', 'motionCoverage', 'confidenceCoverage', 'temporalWeight']) {
      if (Number.isFinite(value.temporal[key]) && value.temporal[key] >= 0) temporal[key] = value.temporal[key];
    }
    if (typeof value.temporal.unchangedSource === 'boolean') temporal.unchangedSource = value.temporal.unchangedSource;
    if (Object.keys(temporal).length) stats.temporal = temporal;
  }
  return stats;
}

function codedError(code, diagnostic) {
  const error = new Error(code);
  error.code = code;
  if (diagnostic && Object.keys(diagnostic).length) error.diagnostic = diagnostic;
  return error;
}

export function pngDataUrlError(dataUrl) {
  if (typeof dataUrl !== 'string' || !dataUrl.startsWith(PNG_PREFIX)) return 'SOURCE_NOT_ISOLATED';
  const encoded = dataUrl.slice(PNG_PREFIX.length);
  if (encoded.length > Math.ceil(MAX_PNG_BYTES / 3) * 4) return 'FRAME_TOO_LARGE';
  if (!encoded.length || encoded.length % 4 || !/^[A-Za-z0-9+/]*={0,2}$/.test(encoded)) return 'SOURCE_NOT_ISOLATED';
  const bytes = encoded.length / 4 * 3 - (encoded.endsWith('==') ? 2 : encoded.endsWith('=') ? 1 : 0);
  return bytes <= MAX_PNG_BYTES ? null : 'FRAME_TOO_LARGE';
}

// Validate IHDR before any browser image decoder can allocate decoded pixels.
export function inspectNativePng(dataUrl, width, height, limit = MAX_PNG_BYTES) {
  const code = nativePngDimensionsError(width, height) || pngDataUrlError(dataUrl);
  if (code) throw codedError(code);
  const encoded = dataUrl.slice(PNG_PREFIX.length);
  const byteLength = encoded.length / 4 * 3 - (encoded.endsWith('==') ? 2 : encoded.endsWith('=') ? 1 : 0);
  if (byteLength > limit) throw codedError('FRAME_TOO_LARGE');
  if (byteLength < 33) throw codedError('SOURCE_NOT_ISOLATED');
  const header = Uint8Array.from(atob(encoded.slice(0, 44)), character => character.charCodeAt(0));
  const view = new DataView(header.buffer);
  if ([137, 80, 78, 71, 13, 10, 26, 10].some((value, index) => header[index] !== value) ||
      view.getUint32(8) !== 13 || view.getUint32(12) !== 0x49484452 ||
      view.getUint32(16) !== width || view.getUint32(20) !== height) throw codedError('SOURCE_NOT_ISOLATED');
  return { byteLength, width, height };
}

export class NativeHostClient {
  constructor({ runtime = globalThis.chrome?.runtime, startTimeoutMs = 45000,
    frameTimeoutMs = 40000, stopTimeoutMs = 10000 } = {}) {
    this.runtime = runtime;
    this.startTimeoutMs = startTimeoutMs;
    this.frameTimeoutMs = frameTimeoutMs;
    this.stopTimeoutMs = stopTimeoutMs;
    this.port = null;
    this.sessionId = null;
    this.inFlight = null;
    this.pending = null;
    this.closed = false;
    this.revision = 0;
  }

  start(options) {
    if (this.port || this.sessionId) return Promise.reject(codedError('NATIVE_SESSION_BUSY'));
    if (!this.runtime?.connect) return Promise.reject(codedError('HOST_UNAVAILABLE'));
    this.sessionId = options.sessionId ?? crypto.randomUUID();
    this.closed = false;
    try { this.port = this.runtime.connect({ name: PORT_NAME }); }
    catch { return Promise.reject(codedError('HOST_UNAVAILABLE')); }
    this.port.onMessage.addListener(this.onMessage = message => this.receive(message));
    this.port.onDisconnect.addListener(this.onDisconnect = () => {
      void this.runtime.lastError;
      this.fail(codedError('HOST_UNAVAILABLE'));
    });
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => this.fail(codedError('HOST_UNAVAILABLE')), this.startTimeoutMs);
      this.startWaiter = { resolve: value => { clearTimeout(timer); resolve(value); },
        reject: error => { clearTimeout(timer); reject(error); } };
      try {
        this.port.postMessage({ protocol: 1, type: 'START', sessionId: this.sessionId,
          backend: options.backend, model: options.model, settings: options.settings,
          captureMode: options.captureMode, sourceWidth: options.sourceWidth, sourceHeight: options.sourceHeight });
      } catch { this.fail(codedError('HOST_UNAVAILABLE')); }
    });
  }

  receive(message) {
    if (this.closed || !message || message.protocol !== 1 || message.sessionId !== this.sessionId) {
      const diagnostic = sessionDiagnostic('native_client', this.sessionId, {
        messageType: typeof message?.type === 'string' ? message.type : 'UNKNOWN',
        receivedSequence: Number.isSafeInteger(message?.sequence) ? message.sequence : undefined,
        expectedSequence: this.inFlight?.sequence,
        revision: Number.isSafeInteger(message?.revision) ? message.revision : undefined,
        expectedRevision: this.configureWaiter?.revision,
        hasInFlight: !!this.inFlight,
        hasPending: !!this.pending,
      });
      this.fail(codedError('STALE_SESSION', diagnostic));
      return;
    }
    if (message.type === 'ERROR') {
      const code = message.code || 'BACKEND_UNAVAILABLE';
      const origin = isStaleOrigin(message.staleOrigin) ? message.staleOrigin : 'native_client';
      this.fail(codedError(code, code === 'STALE_SESSION' ? sessionDiagnostic(origin, this.sessionId, {
        messageType: typeof message.messageType === 'string' ? message.messageType : 'ERROR',
        sequence: Number.isSafeInteger(message.sequence) ? message.sequence : this.inFlight?.sequence,
        revision: Number.isSafeInteger(message.revision) ? message.revision : this.configureWaiter?.revision,
        hasInFlight: !!this.inFlight,
        hasPending: !!this.pending,
      }) : undefined));
      return;
    }
    if (message.type === 'STARTED') {
      if (!this.startWaiter) { this.fail(codedError('STALE_SESSION', sessionDiagnostic('native_client', this.sessionId, { messageType: message.type }))); return; }
      const waiter = this.startWaiter;
      this.startWaiter = null;
      waiter.resolve(message);
      return;
    }
    if (message.type === 'CONFIGURED') {
      const waiter = this.configureWaiter;
      if (!waiter || message.revision !== waiter.revision) { this.fail(codedError('STALE_SESSION', sessionDiagnostic('native_client', this.sessionId, {
        messageType: message.type, revision: Number.isSafeInteger(message.revision) ? message.revision : undefined,
        expectedRevision: waiter?.revision,
      }))); return; }
      this.configureWaiter = null;
      waiter.resolve(message);
      return;
    }
    if (message.type === 'FRAME_OUT') {
      if (this.stopWaiter) return;
      const request = this.inFlight;
      if (!request || message.sequence !== request.sequence || message.mediaTime !== request.mediaTime ||
          !Number.isSafeInteger(message.width) || message.width <= 0 || !Number.isSafeInteger(message.height) ||
          message.height <= 0 || message.mime !== 'image/png' || nativePngDimensionsError(message.width, message.height) ||
          pngDataUrlError(message.dataUrl)) {
        this.fail(codedError('STALE_SESSION', sessionDiagnostic('native_client', this.sessionId, {
          messageType: message.type, receivedSequence: Number.isSafeInteger(message.sequence) ? message.sequence : undefined,
          expectedSequence: request?.sequence,
        })));
        return;
      }
      request.output = { protocol: 1, type: 'FRAME_OUT', sessionId: this.sessionId,
        sequence: message.sequence, mediaTime: message.mediaTime, width: message.width, height: message.height,
        mime: 'image/png', dataUrl: message.dataUrl };
      // Client and overlay share a clock. Never serialize this deadline to the host protocol.
      Object.defineProperty(request.output, 'expiresAt', { value: request.receivedAt + MAX_NATIVE_FRAME_WAIT_MS });
      if (message.stats !== undefined) request.output.stats = sanitizeNativeStats(message.stats);
      this.completeFrame(request);
      return;
    }
    if (message.type === 'FRAME_ACK') {
      if (this.stopWaiter) return;
      const request = this.inFlight;
      if (!request || message.sequence !== request.sequence) { this.fail(codedError('STALE_SESSION', sessionDiagnostic('native_client', this.sessionId, {
        messageType: message.type, receivedSequence: Number.isSafeInteger(message.sequence) ? message.sequence : undefined,
        expectedSequence: request?.sequence,
      }))); return; }
      if (message.accepted !== true) {
        const code = message.code || 'BACKEND_UNAVAILABLE';
        const origin = isStaleOrigin(message.staleOrigin) ? message.staleOrigin : 'native_client';
        this.finishFrame(request, codedError(code, code === 'STALE_SESSION' ? sessionDiagnostic(origin, this.sessionId, {
          messageType: typeof message.messageType === 'string' ? message.messageType : message.type,
          sequence: request.sequence,
        }) : undefined));
        return;
      }
      request.accepted = true;
      this.completeFrame(request);
      return;
    }
    if (message.type === 'STOPPED') {
      const waiter = this.stopWaiter;
      if (!waiter) { this.fail(codedError('STALE_SESSION', sessionDiagnostic('native_client', this.sessionId, { messageType: message.type }))); return; }
      this.stopWaiter = null;
      this.startWaiter?.reject(codedError('STALE_SESSION'));
      this.startWaiter = null;
      waiter.resolve(message);
      this.closePort();
      return;
    }
    this.fail(codedError('BACKEND_UNAVAILABLE'));
  }

  processFrame(frame) {
    const receivedAt = performance.now();
    if (this.closed || !this.port || this.stopWaiter || this.startWaiter || this.configureWaiter) {
      return Promise.reject(codedError('STALE_SESSION', sessionDiagnostic('native_client', this.sessionId, {
        messageType: 'FRAME_IN', sequence: Number.isSafeInteger(frame?.sequence) ? frame.sequence : undefined,
        hasInFlight: !!this.inFlight, hasPending: !!this.pending,
      })));
    }
    if (!Number.isSafeInteger(frame.sequence) || frame.sequence < 0 || !Number.isFinite(frame.mediaTime) ||
        frame.mediaTime < 0 || !Number.isSafeInteger(frame.width) || frame.width <= 0 ||
        !Number.isSafeInteger(frame.height) || frame.height <= 0 || frame.mime !== 'image/png') {
      return Promise.reject(codedError('SOURCE_NOT_ISOLATED'));
    }
    const pngError = nativePngDimensionsError(frame.width, frame.height) || pngDataUrlError(frame.dataUrl);
    if (pngError) return Promise.reject(codedError(pngError));
    return new Promise((resolve, reject) => {
      const request = { ...frame, receivedAt, resolve, reject, output: null, accepted: false };
      if (this.inFlight) {
        if (this.pending) this.pending.reject(codedError('STALE_FRAME'));
        this.pending = request;
      } else this.sendFrame(request);
    });
  }

  sendFrame(request) {
    if (performance.now() - request.receivedAt >= MAX_NATIVE_FRAME_WAIT_MS) {
      request.reject(codedError('STALE_FRAME'));
      return;
    }
    this.inFlight = request;
    request.timer = setTimeout(() => this.fail(codedError('FRAME_TIMEOUT')), this.frameTimeoutMs);
    try { this.port.postMessage({ protocol: 1, type: 'FRAME_IN', sessionId: this.sessionId,
      sequence: request.sequence, mediaTime: request.mediaTime, width: request.width, height: request.height,
      mime: request.mime, dataUrl: request.dataUrl }); }
    catch { this.fail(codedError('HOST_UNAVAILABLE')); }
  }

  completeFrame(request) {
    if (!request.output || !request.accepted || this.inFlight !== request) return;
    this.finishFrame(request, null, request.output);
  }

  finishFrame(request, error, output) {
    if (this.inFlight !== request) return;
    clearTimeout(request.timer);
    this.inFlight = null;
    if (error) request.reject(error); else request.resolve(output);
    if (!this.closed && this.pending) {
      const pending = this.pending;
      this.pending = null;
      this.sendFrame(pending);
    }
  }

  configure({ settings, adaptiveLevel = 0, sourceWidth, sourceHeight }) {
    if (this.closed || !this.port || this.inFlight || this.pending || this.configureWaiter || this.startWaiter || this.stopWaiter) {
      return Promise.reject(codedError('NATIVE_SESSION_BUSY'));
    }
    const revision = ++this.revision;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => this.fail(codedError('FRAME_TIMEOUT')), this.frameTimeoutMs);
      this.configureWaiter = { revision, resolve: value => { clearTimeout(timer); resolve(value); },
        reject: error => { clearTimeout(timer); reject(error); } };
      try { this.port.postMessage({ protocol: 1, type: 'CONFIGURE', sessionId: this.sessionId, revision,
        settings, adaptiveLevel, sourceWidth, sourceHeight }); }
      catch { this.fail(codedError('HOST_UNAVAILABLE')); }
    });
  }

  stop(reason = 'stop') {
    if (!this.port || this.closed) return Promise.resolve();
    if (this.stopWaiter) return this.stopWaiter.promise;
    this.startWaiter?.reject(codedError('STALE_SESSION', sessionDiagnostic('native_client', this.sessionId, { messageType: 'START' })));
    this.startWaiter = null;
    this.configureWaiter?.reject(codedError('STALE_SESSION', sessionDiagnostic('native_client', this.sessionId, {
      messageType: 'CONFIGURE', revision: this.configureWaiter?.revision,
    })));
    this.configureWaiter = null;
    if (this.inFlight) {
      clearTimeout(this.inFlight.timer);
      this.inFlight.reject(codedError('STALE_SESSION', sessionDiagnostic('native_client', this.sessionId, {
        messageType: 'FRAME_IN', sequence: this.inFlight.sequence,
      })));
      this.inFlight = null;
    }
    if (this.pending) { this.pending.reject(codedError('STALE_SESSION', sessionDiagnostic('native_client', this.sessionId, {
      messageType: 'FRAME_IN', sequence: this.pending.sequence,
    }))); this.pending = null; }
    let resolveStop, rejectStop;
    const promise = new Promise((resolve, reject) => { resolveStop = resolve; rejectStop = reject; });
    const timer = setTimeout(() => this.fail(codedError('FRAME_TIMEOUT')), this.stopTimeoutMs);
    this.stopWaiter = { resolve: value => { clearTimeout(timer); resolveStop(value); },
      reject: error => { clearTimeout(timer); rejectStop(error); }, promise };
    try { this.port.postMessage({ protocol: 1, type: 'STOP', sessionId: this.sessionId, reason }); }
    catch { this.fail(codedError('HOST_UNAVAILABLE')); }
    return promise;
  }

  closePort() {
    if (this.closed) return;
    this.closed = true;
    const port = this.port;
    this.port = null;
    if (port) {
      port.onMessage.removeListener(this.onMessage);
      port.onDisconnect.removeListener(this.onDisconnect);
      try { port.disconnect(); } catch { /* Already closed by the service worker. */ }
    }
  }

  fail(error) {
    if (this.closed) return;
    this.startWaiter?.reject(error); this.startWaiter = null;
    this.configureWaiter?.reject(error); this.configureWaiter = null;
    this.stopWaiter?.reject(error); this.stopWaiter = null;
    if (this.inFlight) { clearTimeout(this.inFlight.timer); this.inFlight.reject(error); this.inFlight = null; }
    if (this.pending) { this.pending.reject(error); this.pending = null; }
    this.closePort();
  }
}
