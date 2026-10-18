export class SustainedFpsGate {
  constructor({ targetFps, windowMs = 5000 }) {
    if (!Number.isFinite(targetFps) || targetFps <= 0 || !Number.isFinite(windowMs) || windowMs <= 0) {
      throw new RangeError('targetFps and windowMs must be positive finite numbers');
    }
    this.targetFps = targetFps;
    this.windowMs = windowMs;
    this.samples = [];
  }

  reset() {
    this.samples.length = 0;
  }

  record(timestamp, { warmup = false } = {}) {
    if (!Number.isFinite(timestamp) || timestamp < 0) throw new RangeError('timestamp must be a non-negative finite number');
    if (warmup) return { status: 'warming', fps: null };

    const previous = this.samples.at(-1);
    if (previous !== undefined && (timestamp < previous || timestamp - previous > this.windowMs)) this.reset();
    const latest = this.samples.at(-1);
    if (latest === undefined || timestamp > latest) this.samples.push(timestamp);

    const cutoff = timestamp - this.windowMs;
    while (this.samples.length > 1 && this.samples[1] < cutoff) this.samples.shift();
    const first = this.samples[0];
    const last = this.samples.at(-1);
    if (first === undefined || last - first < this.windowMs) return { status: 'warming', fps: null };

    const fps = (this.samples.length - 1) * 1000 / (last - first);
    return { status: fps >= this.targetFps ? 'within-target' : 'below-target', fps };
  }
}
