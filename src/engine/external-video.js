// External textures must be imported/bound/submitted in the same JS task for
// HTMLVideoElement, and before closing a VideoFrame. Never cache across frames.
export function importVideoTexture(device,source) {
  const video=typeof HTMLVideoElement==='function' && source instanceof HTMLVideoElement;
  const frame=typeof VideoFrame==='function' && source instanceof VideoFrame;
  if (!device.importExternalTexture || (!video && !frame)) return null;
  try {return device.importExternalTexture({source,colorSpace:'srgb'});}
  catch(error){
    if(error.name==='SecurityError')throw error;
    return null; // Canvas/ImageBitmap and unsupported browsers use copy path.
  }
}

export function videoCropUniform(source,crop,width,height) {
  const w=source.videoWidth??source.displayWidth,h=source.videoHeight??source.displayHeight;
  const x=crop?.x??0,y=crop?.y??0,cw=crop?.width??width,ch=crop?.height??height;
  if (!(w>0 && h>0 && cw>0 && ch>0 && x>=0 && y>=0 && x+cw<=w && y+ch<=h)) {
    throw new RangeError('Video crop exceeds decoded frame dimensions');
  }
  // Clamp to the cropped region's pixel centers, preserving the old copied
  // texture's edge behavior when the input/model resolution exceeds the crop.
  return new Float32Array([x/w,y/h,cw/w,ch/h,(x+.5)/w,(y+.5)/h,(x+cw-.5)/w,(y+ch-.5)/h]);
}
