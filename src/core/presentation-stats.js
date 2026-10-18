// Delivered image statistics: inference FPS alone does not describe AI coverage.
export class PresentationStats {
  constructor({ startedAt = null, maxSamples = 600 } = {}) {
    this.startedAt = startedAt;
    this.maxSamples = Math.max(1, Math.floor(maxSamples));
    this.frames = 0; this.coverage = 0; this.lowConfidence = 0; this.ages = [];
    this.components = {};
  }
  record(metrics, now) {
    const coverage = metrics?.aiCoverage, age = metrics?.historyAgeMs;
    if (!Number.isFinite(now) || !Number.isFinite(coverage) || coverage < 0 || coverage > 1 || !Number.isFinite(age) || age < 0) return false;
    if (this.startedAt === null) this.startedAt = now;
    this.frames++; this.coverage += coverage;
    if (coverage < .1) this.lowConfidence++;
    this.ages.push(age);
    if (this.ages.length > this.maxSamples) this.ages.shift();
    for(const key of ['motionCoverage','confidenceCoverage','temporalWeight']) {
      const value=metrics[key];
      if(Number.isFinite(value) && value>=0 && value<=1) {
        const sample=this.components[key]??={sum:0,count:0};sample.sum+=value;sample.count++;
      }
    }
    return true;
  }
  snapshot(now) {
    if (!this.frames) return null;
    const ages = [...this.ages].sort((a, b) => a - b);
    const elapsedMs = Math.max(1, now - this.startedAt);
    return { ...Object.fromEntries(Object.entries(this.components).map(([key,value])=>[`${key}Mean`,value.sum/value.count])),
      presentedFrames: this.frames, elapsedMs, presentationFps: this.frames * 1000 / elapsedMs,
      aiCoverageMean: this.coverage / this.frames, lowConfidenceRatio: this.lowConfidence / this.frames,
      historyAgeP95Ms: ages[Math.ceil(ages.length * .95) - 1] };
  }
  flush(now) {
    const result = this.snapshot(now);
    this.startedAt = null; this.frames = 0; this.coverage = 0; this.lowConfidence = 0; this.ages = [];
    this.components = {};
    return result;
  }
}
