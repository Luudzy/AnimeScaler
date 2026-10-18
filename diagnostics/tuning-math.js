export function compareSamples(reference, samples) {
  if (!reference?.length || reference.length !== samples?.length) return {passed: false};
  let sum = 0, max = 0;
  for (let i = 0; i < reference.length; i++) {
    if (!Number.isFinite(reference[i]) || !Number.isFinite(samples[i])) return {passed: false};
    const error = Math.abs(reference[i] - samples[i]);
    sum += error; max = Math.max(max, error);
  }
  const meanAbsoluteError = sum / reference.length;
  return {passed: meanAbsoluteError <= .005 && max <= .05, meanAbsoluteError, maxAbsoluteError: max};
}
export function chooseCandidate(results) {
  return results.filter(r => !r.error && r.correctness?.passed && Number.isFinite(r.p95Ms))
    .sort((a, b) => a.p95Ms - b.p95Ms)[0] ?? null;
}
