// Keep source age bounded even when inference latency spikes.
export const TEMPORAL_MAX_SOURCE_AGE_MS = 300;

export function temporalStrength(ageMs, latencyMs = 50) {
  if (!Number.isFinite(ageMs) || ageMs < 0) return 0;
  const latency = Number.isFinite(latencyMs) ? Math.max(0,latencyMs) : 50;
  const budget = TEMPORAL_MAX_SOURCE_AGE_MS;
  // A valid result must retain constant weight between ordinary AI arrivals.
  // Fading at the inference period produced a repeating sharp/soft pulse.
  const fadeStart = Math.min(240,Math.max(180,latency*3+66));
  return Math.max(0,Math.min(1,(budget-ageMs)/(budget-fadeStart)));
}
