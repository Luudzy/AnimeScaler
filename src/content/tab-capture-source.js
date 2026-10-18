import { buildTabCaptureConstraints } from '../core/tab-capture.js';

export class TabCaptureSourceError extends Error {
  constructor(code, message, cause) {
    super(message, { cause });
    this.name = 'TabCaptureSourceError';
    this.code = code;
  }
}

export class TabCaptureSource {
  constructor({
    mediaDevices = globalThis.navigator?.mediaDevices,
    restrictionTargetApi = globalThis.RestrictionTarget,
    documentRef = globalThis.document,
  } = {}) {
    this.mediaDevices = mediaDevices;
    this.restrictionTargetApi = restrictionTargetApi;
    this.documentRef = documentRef;
    this.stream = null;
    this.video = null;
    this.targetElement = null;
    this.previousIsolation = '';
  }

  async start({ streamId, targetElement, timeoutMs = 8000 }) {
    if (!targetElement) {
      throw new TabCaptureSourceError('CAPTURE_TARGET_NOT_FOUND', 'Não foi possível localizar o player para a captura da aba.');
    }
    if (!this.restrictionTargetApi?.fromElement) {
      throw new TabCaptureSourceError(
        'ELEMENT_CAPTURE_UNAVAILABLE',
        'Este navegador não oferece Element Capture para isolar o player. O modo inline não pode capturar esta fonte.',
      );
    }
    if (!this.mediaDevices?.getUserMedia) {
      throw new TabCaptureSourceError('CAPTURE_UNAVAILABLE', 'getUserMedia não está disponível neste contexto.');
    }

    this.targetElement = targetElement;
    this.previousIsolation = targetElement.style?.isolation ?? '';
    if (targetElement.style) targetElement.style.isolation = 'isolate';

    try {
      this.stream = await this.mediaDevices.getUserMedia(buildTabCaptureConstraints(streamId));
      const [track] = this.stream.getVideoTracks?.() ?? [];
      if (!track) {
        throw new TabCaptureSourceError('TAB_CAPTURE_NO_VIDEO', 'A captura da aba não retornou uma faixa de vídeo.');
      }
      if (typeof track.restrictTo !== 'function') {
        throw new TabCaptureSourceError(
          'ELEMENT_CAPTURE_UNAVAILABLE',
          'A faixa de captura não suporta restrictTo(); o modo inline não pode isolar o player.',
        );
      }

      try {
        const restrictionTarget = await this.restrictionTargetApi.fromElement(targetElement);
        await track.restrictTo(restrictionTarget);
      } catch (error) {
        throw new TabCaptureSourceError(
          'SOURCE_NOT_ISOLATED',
          `Element Capture falhou (${error?.name || 'erro'}); nenhum frame foi consumido.`,
          error,
        );
      }

      this.video = this.documentRef.createElement('video');
      this.video.muted = true;
      this.video.autoplay = true;
      this.video.playsInline = true;
      this.video.srcObject = this.stream;
      await this.video.play?.();
      await this.waitForVideoReady(timeoutMs);
      return this.video;
    } catch (error) {
      this.destroy();
      if (error instanceof TabCaptureSourceError) throw error;
      throw new TabCaptureSourceError(
        'TAB_CAPTURE_START_FAILED',
        `Não foi possível iniciar a captura da aba: ${error?.message ?? String(error)}`,
        error,
      );
    }
  }

  async waitForVideoReady(timeoutMs) {
    if (this.video?.videoWidth > 0 && this.video?.videoHeight > 0) return;
    await new Promise((resolve, reject) => {
      let timer = null;
      const events = ['loadedmetadata', 'loadeddata', 'canplay', 'playing', 'resize'];
      const cleanup = () => {
        if (timer) clearTimeout(timer);
        for (const event of events) this.video?.removeEventListener?.(event, check);
      };
      const check = () => {
        if (this.video?.videoWidth > 0 && this.video?.videoHeight > 0) {
          cleanup();
          resolve();
        }
      };
      for (const event of events) this.video?.addEventListener?.(event, check, { passive: true });
      timer = setTimeout(() => {
        cleanup();
        reject(new TabCaptureSourceError(
          'TAB_CAPTURE_NOT_READY',
          'A captura da aba iniciou, mas não entregou frames de vídeo a tempo.',
        ));
      }, timeoutMs);
      check();
    });
  }

  destroy() {
    try { this.video?.pause?.(); } catch { /* no-op */ }
    if (this.video) this.video.srcObject = null;
    for (const track of this.stream?.getTracks?.() ?? []) {
      try { track.stop?.(); } catch { /* no-op */ }
    }
    if (this.targetElement?.style) this.targetElement.style.isolation = this.previousIsolation;
    this.stream = null;
    this.video = null;
    this.targetElement = null;
    this.previousIsolation = '';
  }
}
