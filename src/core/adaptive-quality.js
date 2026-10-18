export class AdaptiveQualityController {
  constructor({
    initialLevel = 0,
    maxLevel = 7,
    overloadSamples = 8,
    recoverySamples = 90,
    cooldownSamples = 30,
    alpha = 0.2,
    overloadRatio = 0.92,
    recoveryRatio = 0.55,
  } = {}) {
    this.level = Math.max(0, Math.min(maxLevel, initialLevel));
    this.maxLevel = Math.max(0, maxLevel);
    this.overloadSamples = Math.max(1, overloadSamples);
    this.recoverySamples = Math.max(1, recoverySamples);
    this.cooldownSamples = Math.max(0, cooldownSamples);
    this.alpha = Math.max(0.01, Math.min(1, alpha));
    this.overloadRatio = overloadRatio;
    this.recoveryRatio = recoveryRatio;
    this.emaMs = null;
    this.overloadCount = 0;
    this.recoveryCount = 0;
    this.cooldown = 0;
  }

  reset(level = 0) {
    this.level = Math.max(0, Math.min(this.maxLevel, level));
    this.emaMs = null;
    this.overloadCount = 0;
    this.recoveryCount = 0;
    this.cooldown = 0;
  }

  record(processMs, frameIntervalMs) {
    if (!Number.isFinite(processMs) || processMs < 0 || !Number.isFinite(frameIntervalMs) || frameIntervalMs <= 0) {
      return { changed: false, level: this.level, emaMs: this.emaMs };
    }

    this.emaMs = this.emaMs == null
      ? processMs
      : this.alpha * processMs + (1 - this.alpha) * this.emaMs;

    // A cooldown measured in frames can last minutes at multi-second inference
    // times. Let severe overload step down without waiting for more samples.
    if (processMs > Math.max(250, frameIntervalMs * 4) && this.level < this.maxLevel) {
      this.level += 1;
      this.overloadCount = 0;
      this.recoveryCount = 0;
      this.emaMs = processMs;
      this.cooldown = this.cooldownSamples;
      return { changed: true, level: this.level, emaMs: this.emaMs };
    }

    if (this.cooldown > 0) {
      this.cooldown -= 1;
      return { changed: false, level: this.level, emaMs: this.emaMs };
    }

    if (this.emaMs > frameIntervalMs * this.overloadRatio) {
      this.overloadCount += 1;
      this.recoveryCount = 0;
      if (this.overloadCount >= this.overloadSamples && this.level < this.maxLevel) {
        this.level += 1;
        this.overloadCount = 0;
        this.cooldown = this.cooldownSamples;
        return { changed: true, level: this.level, emaMs: this.emaMs };
      }
    } else if (this.emaMs < frameIntervalMs * this.recoveryRatio) {
      this.recoveryCount += 1;
      this.overloadCount = 0;
      if (this.recoveryCount >= this.recoverySamples && this.level > 0) {
        this.level -= 1;
        this.recoveryCount = 0;
        this.cooldown = this.cooldownSamples;
        return { changed: true, level: this.level, emaMs: this.emaMs };
      }
    } else {
      this.overloadCount = 0;
      this.recoveryCount = 0;
    }

    return { changed: false, level: this.level, emaMs: this.emaMs };
  }
}
