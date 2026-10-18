// Keep an absolute cadence instead of accumulating callback lateness every frame.
export class FramePacer {
  constructor() { this.reset(); }
  reset() { this.nextAt = null; this.fps = null; }
  take(now, fps) {
    const interval = 1000 / fps;
    if (!Number.isFinite(now) || !Number.isFinite(interval) || interval <= 0) return false;
    if (this.fps !== fps || this.nextAt == null) {
      this.fps = fps;
      this.nextAt = now + interval;
      return true;
    }
    if (now + 0.5 < this.nextAt) return false;
    // A stalled/busy renderer drops missed frames rather than catching up in bursts.
    this.nextAt = now - this.nextAt >= interval ? now + interval : this.nextAt + interval;
    return true;
  }
}
