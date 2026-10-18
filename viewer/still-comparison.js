// The caller drains live inference/presentation before entering this function.
// Snapshot once: even a queued decoder update cannot change the comparison.
export async function renderStillComparison({capture, original, engine, crop, snapshot = createImageBitmap}) {
  capture.pause();
  const frozen = await snapshot(capture);
  try {
  const {outputWidth,outputHeight} = engine.currentSize;
  original.width = outputWidth;
  original.height = outputHeight;
  const context = original.getContext('2d');
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = 'high';
  context.drawImage(frozen,crop.x,crop.y,crop.width,crop.height,0,0,outputWidth,outputHeight);
  return await engine.processFrame(frozen,crop,{still:true});
  } finally {frozen.close();}
}
