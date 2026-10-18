function reachableVideoFrames(probes = []) {
  return probes.filter((probe) => probe?.ok && probe.hasVideo && Number.isFinite(probe.videoScore) && probe.videoScore >= 0);
}

export function chooseFrameForStart(probes = []) {
  return reachableVideoFrames(probes)
    .sort((a, b) => (b.videoScore ?? -1) - (a.videoScore ?? -1))[0] ?? null;
}

export function chooseRunningFrame(probes = []) {
  return probes.find((probe) => probe?.ok && (probe.state === 'running' || probe.state === 'starting')) ?? null;
}

export function chooseFrameForStatus(probes = []) {
  return chooseRunningFrame(probes)
    ?? probes.find((probe) => probe?.ok && probe.state === 'error' && probe.captureMode === 'tab')
    ?? probes.find((probe) => probe?.ok && probe.state === 'error' && probe.hasVideo)
    ?? chooseFrameForStart(probes)
    ?? probes.find((probe) => probe?.ok)
    ?? null;
}
