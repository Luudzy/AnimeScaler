export function formatPresentationMetrics(metrics) {
  const coverage = metrics?.aiCoverage, age = metrics?.historyAgeMs;
  if (!Number.isFinite(coverage) || coverage < 0 || coverage > 1 || !Number.isFinite(age) || age < 0) return '';
  return `Contribuição IA ${Math.round(coverage * 100)}% · atraso ${Math.round(age)} ms`;
}
