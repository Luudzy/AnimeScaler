import { inspectNativePng } from './native-host-client.js';

export class VideoOverlay {
  constructor() {
    this.video = null;
    this.canvas = document.createElement('canvas');
    this.canvas.dataset.animescaler = 'canvas';
    Object.assign(this.canvas.style, {
      position: 'fixed',
      display: 'none',
      pointerEvents: 'none',
      margin: '0',
      padding: '0',
      border: '0',
      zIndex: '2147483000',
      background: 'black',
      objectFit: 'contain',
    });
    document.documentElement.appendChild(this.canvas);
  }

  attach(video) {
    this.video = video;
    this.sync();
  }

  sync() {
    if (!this.video?.isConnected) return;
    const rect = this.video.getBoundingClientRect();
    Object.assign(this.canvas.style, {
      left: `${rect.left}px`,
      top: `${rect.top}px`,
      width: `${rect.width}px`,
      height: `${rect.height}px`,
      borderRadius: getComputedStyle(this.video).borderRadius,
    });
  }

  show() {
    this.canvas.style.display = 'block';
  }

  async presentFrame(frame, isCurrent = () => true) {
    if (!this.canvas?.isConnected || !this.video?.isConnected || !isCurrent()) return false;
    if (this.presentationBusy) throw Object.assign(new Error('NATIVE_SESSION_BUSY'), { code: 'NATIVE_SESSION_BUSY' });
    if (!Number.isSafeInteger(frame.sequence) || frame.sequence <= (this.presentedSequence ?? -1) ||
        !Number.isFinite(frame.mediaTime) || frame.mediaTime < 0 || frame.mime !== 'image/png' ||
        !Number.isSafeInteger(frame.width) || frame.width <= 0 || !Number.isSafeInteger(frame.height) || frame.height <= 0 ||
        typeof frame.dataUrl !== 'string' || !frame.dataUrl.startsWith('data:image/png;base64,')) {
      throw Object.assign(new Error('INVALID_PNG_OUTPUT'), { code: 'INVALID_PNG_OUTPUT' });
    }
    const expired = () => Number.isFinite(frame.expiresAt) && performance.now() >= frame.expiresAt;
    if (expired()) return false;
    let bitmap;
    this.presentationBusy = true;
    try {
      try { inspectNativePng(frame.dataUrl, frame.width, frame.height); }
      catch (error) {
        const code = error.code === 'FRAME_TOO_LARGE' ? error.code : 'INVALID_PNG_OUTPUT';
        throw Object.assign(new Error(code, { cause: error }), { code });
      }
      const encoded = frame.dataUrl.slice('data:image/png;base64,'.length);
      const binary = atob(encoded);
      const bytes = new Uint8Array(binary.length);
      for (let index = 0; index < binary.length; index++) bytes[index] = binary.charCodeAt(index);
      bitmap = await createImageBitmap(new Blob([bytes], { type: 'image/png' }));
      if (bitmap.width !== frame.width || bitmap.height !== frame.height) {
        throw Object.assign(new Error('INVALID_PNG_OUTPUT'), { code: 'INVALID_PNG_OUTPUT' });
      }
      if (!isCurrent() || expired() || !this.canvas?.isConnected || !this.video?.isConnected ||
          frame.sequence <= (this.presentedSequence ?? -1)) return false;
      const context = this.canvas.getContext('2d');
      if (!context) throw Object.assign(new Error('PRESENTER_LOST'), { code: 'PRESENTER_LOST' });
      this.canvas.width = frame.width;
      this.canvas.height = frame.height;
      context.drawImage(bitmap, 0, 0, frame.width, frame.height);
      this.presentedSequence = frame.sequence;
      return true;
    } catch (error) {
      if (error?.code) throw error;
      throw Object.assign(new Error('PRESENTER_LOST', { cause: error }), { code: 'PRESENTER_LOST' });
    } finally {
      try { bitmap?.close(); }
      finally { this.presentationBusy = false; }
    }
  }

  hide() {
    this.canvas.style.display = 'none';
  }

  destroy() {
    this.video = null;
    this.canvas.remove();
  }
}
