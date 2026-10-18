import { NativeEngine } from '../engine/native-engine.js';
import { resolveBackendDescriptor } from '../core/backend-descriptor.js';
import { normalizeSettings } from '../core/settings.js';
import { isNativeMessageShape } from '../core/native-protocol.js';
import { sessionDiagnostic } from '../core/session-diagnostics.js';
import { inspectNativePng, nativePngDimensionsError, sanitizeNativeStats, MAX_NATIVE_FRAME_WAIT_MS } from '../content/native-host-client.js';

export const MAX_NATIVE_FRAME_BYTES = 16 * 1024 * 1024;
const PORT_NAME = 'animescaler-native-inline-v1';
const PNG_PREFIX = 'data:image/png;base64,';
const FRAME_TIMEOUT_MS = 40000;
const dimension = value => Number.isSafeInteger(value) && value > 0 && value <= 0x7fffffff;
function fail(code, diagnostic) {
  const error = new Error(code);
  error.code = code;
  if (diagnostic && Object.keys(diagnostic).length) error.diagnostic = diagnostic;
  throw error;
}

function staleSession(message, details = {}) {
  fail('STALE_SESSION', sessionDiagnostic('native_host', message?.sessionId, {
    messageType: typeof message?.type === 'string' ? message.type : 'UNKNOWN',
    sequence: Number.isSafeInteger(message?.sequence) ? message.sequence : undefined,
    revision: Number.isSafeInteger(message?.revision) ? message.revision : undefined,
    ...details,
  }));
}

function inspectPng(dataUrl, width, height, limit) {
  const png = inspectNativePng(dataUrl, width, height, limit);
  const encoded = dataUrl.slice(PNG_PREFIX.length);
  // Bound the allocation before decoding; the limit counts binary PNG bytes.
  const binary = atob(encoded);
  const bytes = Uint8Array.from(binary, character => character.charCodeAt(0));
  return { ...png, bytes };
}

function nativeSettings(value, backend = value?.backend, model = value?.model) {
  const requested = { ...value, backend, model };
  const descriptor = resolveBackendDescriptor(requested);
  if (!descriptor.requiresLocalService || descriptor.id !== backend || descriptor.modelId !== model) {
    fail('BACKEND_UNAVAILABLE');
  }
  return normalizeSettings({ ...requested, enabled: true });
}

async function decodePng(_dataUrl, png) {
  const bitmap = await createImageBitmap(new Blob([png.bytes], { type: 'image/png' }));
  return { bitmap, width: bitmap.width, height: bitmap.height, byteLength: png.byteLength };
}

export class NativeInlineHost {
  constructor({ send, createEngine = () => new NativeEngine(),
    createCanvas = () => document.createElement('canvas'), decodeInput = decodePng,
    maxFrameBytes = MAX_NATIVE_FRAME_BYTES, extensionId = globalThis.chrome?.runtime?.id } = {}) {
    if (typeof send !== 'function' || !Number.isSafeInteger(maxFrameBytes) || maxFrameBytes <= 0 ||
        maxFrameBytes > MAX_NATIVE_FRAME_BYTES) throw new TypeError('Invalid native host options');
    this.send = send;
    this.createEngine = createEngine;
    this.createCanvas = createCanvas;
    this.decodeInput = decodeInput;
    this.maxFrameBytes = maxFrameBytes;
    this.extensionId = extensionId;
    this.session = null;
    this.disconnected = false;
  }

  validateSender(sender) {
    if (!this.extensionId || sender?.id !== this.extensionId || sender.tab !== undefined ||
        sender.url !== `chrome-extension://${this.extensionId}/background.js`) {
      fail('BACKEND_UNAVAILABLE');
    }
  }

  attachPort(port) {
    if (port?.name !== PORT_NAME) fail('BACKEND_UNAVAILABLE');
    this.validateSender(port.sender);
    if (this.port || this.disconnected) fail('NATIVE_SESSION_BUSY');
    this.port = port;
    this.onMessage = message => { void this.handleMessage(message, port.sender); };
    this.onDisconnect = () => { void this.disconnect(); };
    port.onMessage.addListener(this.onMessage);
    port.onDisconnect.addListener(this.onDisconnect);
  }

  emit(type, message, fields = {}) {
    if (!this.disconnected) this.send({ protocol: 1, sessionId: message.sessionId, type, ...fields });
  }

  async handleMessage(message, sender, receivedAt = performance.now()) {
    try {
      // Direct calls are the trusted host/unit interface; ports always supply a sender.
      if (sender !== undefined) this.validateSender(sender);
      if (this.disconnected || !message || message.protocol !== 1 ||
          typeof message.sessionId !== 'string' || !message.sessionId || message.sessionId.length > 256) {
        fail('BACKEND_UNAVAILABLE');
      }
      if (!isNativeMessageShape(message)) fail('SOURCE_NOT_ISOLATED');
      if (message.type === 'START') return await this.start(message);
      const session = this.session;
      if (!session || session.id !== message.sessionId) staleSession(message);
      if (message.type === 'STOP') {
        session.explicitStop = true;
        await this.cleanup(session);
        this.emit('STOPPED', message);
      } else if (session.closed) staleSession(message, { closed: true });
      else if (message.type === 'FRAME_IN') await this.frame(session, message, receivedAt);
      else if (message.type === 'CONFIGURE') {
        if (!session.started || session.busy) fail('NATIVE_SESSION_BUSY');
        if (!Number.isSafeInteger(message.revision) || message.revision <= session.revision ||
            !Number.isSafeInteger(message.adaptiveLevel) || message.adaptiveLevel < 0 || message.adaptiveLevel > 7 ||
            !dimension(message.sourceWidth) || !dimension(message.sourceHeight)) staleSession(message, {
              expectedRevision: session.revision + 1,
            });
        const settings = nativeSettings({ ...session.settings, ...message.settings });
        const sizeError = nativePngDimensionsError(message.sourceWidth, message.sourceHeight);
        if (sizeError) fail(sizeError);
        const video = { videoWidth: message.sourceWidth, videoHeight: message.sourceHeight };
        session.busy = true;
        try {
          session.configuring = (async () => {
            if (settings.backend !== session.settings.backend || settings.model !== session.settings.model) {
              await session.engine.destroy();
              if (session.closed) return;
              const canvas = this.createCanvas();
              const engine = this.createEngine();
              session.engine = engine;
              session.canvas = canvas;
              await engine.initialize({ canvas, video, settings, adaptiveLevel: message.adaptiveLevel,
                sessionId: message.sessionId });
            } else {
              await session.engine.updateSettings(settings, video, message.adaptiveLevel);
            }
          })();
          await session.configuring;
          if (!session.closed) {
            session.settings = settings;
            session.video = video;
            session.revision = message.revision;
            session.adaptiveLevel = message.adaptiveLevel;
            this.emit('CONFIGURED', message, { revision: message.revision,
              backend: settings.backend, model: settings.model });
          }
        } catch (error) {
          if (session.closed || session.explicitStop) return;
          await this.cleanup(session);
          if (!session.explicitStop && !this.disconnected) throw error;
        } finally { session.configuring = null; session.busy = false; }
      }
    } catch (error) {
      this.reportError(message, error);
    }
  }

  reportError(message, error) {
    // Never relay engine messages: they may contain private connection details.
    const code = error.code || 'BACKEND_UNAVAILABLE';
    const stale = code === 'STALE_SESSION' ? {
      staleOrigin: error.diagnostic?.staleOrigin ?? 'native_host',
      messageType: error.diagnostic?.messageType ?? message?.type ?? 'UNKNOWN',
      ...(Number.isSafeInteger(message?.sequence) ? { sequence: message.sequence } : {}),
      ...(Number.isSafeInteger(message?.revision) ? { revision: message.revision } : {}),
    } : {};
    this.emit('ERROR', message ?? {}, { code, ...stale });
    if (message?.type === 'FRAME_IN') this.emit('FRAME_ACK', message, {
      sequence: message.sequence, accepted: false, code, ...stale,
    });
  }

  async processWithDeadline(session, bitmap, options) {
    let timer;
    const deadline = new Promise((_, reject) => {
      session.cancelFrame = () => reject(Object.assign(new Error('STALE_SESSION'), {
        code: 'STALE_SESSION', diagnostic: sessionDiagnostic('native_host', session.id, {
          messageType: 'FRAME_IN', sequence: options.sequence, closed: session.closed,
        }),
      }));
      timer = setTimeout(() => reject(Object.assign(new Error('FRAME_TIMEOUT'), { code: 'FRAME_TIMEOUT' })), FRAME_TIMEOUT_MS);
      session.frameTimer = timer;
    });
    try {
      const processing = Promise.resolve().then(() => {
        if (session.closed) staleSession({ sessionId: session.id, type: 'FRAME_IN', sequence: options.sequence }, { closed: true });
        return session.engine.processImage(bitmap, options);
      });
      return await Promise.race([processing, deadline]);
    } finally {
      clearTimeout(timer);
      session.frameTimer = null;
      session.cancelFrame = null;
    }
  }

  async start(message) {
    if (this.session) fail('NATIVE_SESSION_BUSY');
    if (!['direct-png', 'restricted-png'].includes(message.captureMode) ||
        !dimension(message.sourceWidth) || !dimension(message.sourceHeight)) fail('SOURCE_NOT_ISOLATED');
    const sizeError = nativePngDimensionsError(message.sourceWidth, message.sourceHeight);
    if (sizeError) fail(sizeError);
    if ((message.settings?.backend && message.settings.backend !== message.backend) ||
        (message.settings?.model && message.settings.model !== message.model)) fail('BACKEND_UNAVAILABLE');
    const settings = nativeSettings(message.settings, message.backend, message.model);
    const session = { id: message.sessionId, settings, sequence: -1, revision: 0, adaptiveLevel: 0,
      busy: false, closed: false,
      video: { videoWidth: message.sourceWidth, videoHeight: message.sourceHeight } };
    this.session = session;
    try {
      session.canvas = this.createCanvas();
      session.engine = this.createEngine();
      session.ready = Promise.resolve(session.engine.initialize({ canvas: session.canvas, video: session.video, settings,
        sessionId: message.sessionId }));
      await session.ready;
      if (!session.closed) {
        session.started = true;
        this.emit('STARTED', message, { backend: settings.backend });
      }
    } catch (error) {
      if (session.closed || session.explicitStop) return;
      await this.cleanup(session);
      if (!session.explicitStop && !this.disconnected) throw error;
    }
  }

  async frame(session, message, receivedAt) {
    if (!session.started || (session.busy && !session.processing)) fail('NATIVE_SESSION_BUSY');
    if (!Number.isSafeInteger(message.sequence) || message.sequence < 0 ||
        message.sequence <= (session.pending?.sequence ?? session.sequence) ||
        !Number.isFinite(message.mediaTime) || message.mediaTime < 0) fail('STALE_FRAME');
    if (message.mime !== 'image/png') fail('SOURCE_NOT_ISOLATED');
    const expire = () => {
      if (performance.now() - receivedAt < MAX_NATIVE_FRAME_WAIT_MS) return false;
      session.sequence = Math.max(session.sequence, message.sequence);
      this.emit('FRAME_ACK', message, { sequence: message.sequence, accepted: false, code: 'STALE_FRAME' });
      return true;
    };
    if (expire()) return;
    const png = inspectPng(message.dataUrl, message.width, message.height, this.maxFrameBytes);
    if (session.processing) {
      // Retain only the newest PNG; reject displaced frames immediately without a backlog.
      if (session.pending) this.emit('FRAME_ACK', session.pending, {
        sequence: session.pending.sequence, accepted: false, code: 'STALE_FRAME',
      });
      session.pending = message;
      session.pendingReceivedAt = receivedAt;
      return;
    }
    session.busy = true;
    session.processing = true;
    session.sequence = message.sequence;
    let decoded;
    try {
      if (expire()) return;
      decoded = await this.decodeInput(message.dataUrl, png);
      if (!decoded?.bitmap || decoded.width !== png.width || decoded.height !== png.height ||
          decoded.byteLength !== png.byteLength) fail('SOURCE_NOT_ISOLATED');
      if (session.closed) return;
      if (!expire()) {
        const output = await this.processWithDeadline(session, decoded.bitmap, {
          sequence: message.sequence, mediaTime: message.mediaTime,
          crop: { x: 0, y: 0, width: png.width, height: png.height },
        });
        if (session.closed) return;
        if (output?.sequence !== message.sequence || output.mediaTime !== message.mediaTime ||
            output.mime !== 'image/png') fail('INVALID_PNG_OUTPUT');
        inspectPng(output.dataUrl, output.width, output.height, this.maxFrameBytes);
        this.emit('FRAME_OUT', message, { sequence: output.sequence, mediaTime: output.mediaTime,
          width: output.width, height: output.height, mime: 'image/png', dataUrl: output.dataUrl,
          ...(output.stats === undefined ? {} : { stats: sanitizeNativeStats(output.stats) }) });
        this.emit('FRAME_ACK', message, { sequence: message.sequence, accepted: true });
      }
    } catch (error) {
      if (session.closed) return;
      const pending = session.pending;
      await this.cleanup(session);
      if (!session.explicitStop && !this.disconnected) {
        this.reportError(message, error);
        if (pending) this.emit('FRAME_ACK', pending, {
          sequence: pending.sequence, accepted: false, code: error.code || 'BACKEND_UNAVAILABLE',
        });
      }
      return;
    } finally {
      decoded?.bitmap?.close();
      session.busy = false;
      session.processing = false;
    }
    if (!session.closed) {
      const pending = session.pending;
      const pendingReceivedAt = session.pendingReceivedAt;
      session.pending = null;
      session.pendingReceivedAt = undefined;
      if (pending) await this.handleMessage(pending, undefined, pendingReceivedAt);
    }
  }

  cleanup(session) {
    if (session.cleanup) return session.cleanup;
    session.stopping = true;
    session.closed = true;
    session.pending = null;
    session.pendingReceivedAt = undefined;
    clearTimeout(session.frameTimer);
    session.cancelFrame?.();
    // Cancel owned work before waiting for initialization or configuration to settle.
    let destroying;
    try { destroying = Promise.resolve(session.engine?.destroy()); }
    catch (error) { destroying = Promise.reject(error); }
    session.cleanup = (async () => {
      try {
        const results = await Promise.allSettled([session.ready, session.configuring, destroying]);
        if (results[2].status === 'rejected') throw results[2].reason;
      }
      finally {
        if (session.canvas) { session.canvas.width = 1; session.canvas.height = 1; }
        if (this.session === session) this.session = null;
      }
    })();
    return session.cleanup;
  }

  async disconnect() {
    this.disconnected = true;
    if (this.port) {
      this.port.onMessage.removeListener(this.onMessage);
      this.port.onDisconnect.removeListener(this.onDisconnect);
      this.port = null;
    }
    if (this.session) await this.cleanup(this.session);
  }
}
