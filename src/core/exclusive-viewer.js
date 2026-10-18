export function exclusiveFrameChain(frames, targetFrameId) {
  const chain = [], seen = new Set();
  let frame = frames.find(f => f.frameId === targetFrameId);
  while (frame && !seen.has(frame.frameId)) {
    chain.unshift(frame); seen.add(frame.frameId);
    if (frame.frameId === 0) break;
    frame = frames.find(f => f.frameId === frame.parentFrameId);
  }
  if (chain[0]?.frameId !== 0) throw new Error('Não foi possível localizar toda a cadeia de iframes do vídeo.');
  return chain;
}

export async function restoreExclusiveSource(session, chrome = globalThis.chrome) {
  await Promise.allSettled((session.exclusiveFrames ?? []).map(frameId => chrome.tabs.sendMessage(
    session.sourceTabId, {type:'ANIMESCALER_EXCLUSIVE_EXIT', sessionId:session.sessionId}, {frameId},
  )));
}

export async function expandExclusiveSource(tabId, targetFrameId, frames, sessionId, chrome = globalThis.chrome) {
  const chain = exclusiveFrameChain(frames, targetFrameId), touched = [];
  try {
    for (let i = 0; i < chain.length; i++) {
      const frameId = chain[i].frameId;
      touched.push(frameId);
      const result = await chrome.tabs.sendMessage(tabId, {type:'ANIMESCALER_EXCLUSIVE_ENTER', sessionId,
        captureFrame:chain[i+1] ?? {frameId:0}}, {frameId});
      if (!result?.ok) throw new Error(result?.message || 'O player não respondeu ao modo exclusivo.');
    }
    return touched;
  } catch (error) {
    await restoreExclusiveSource({sourceTabId:tabId, exclusiveFrames:touched, sessionId}, chrome);
    throw error;
  }
}
