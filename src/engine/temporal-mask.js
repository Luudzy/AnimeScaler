// Conservative thumbnail comparison, not optical flow or frame interpolation.
export function stableMask(reference, current, width, height) {
  const length = width * height;
  if (!reference || !current || reference.length !== length*4 || current.length !== length*4) return null;
  const changed = new Uint8Array(length);
  let count = 0;
  for (let i=0; i<length; i++) {
    for (let c=0; c<3; c++) if (Math.abs(reference[i*4+c]-current[i*4+c]) > 3) { changed[i]=1; count++; break; }
  }
  if (count > length * .35) return null;
  const mask = new Uint8ClampedArray(length*4);
  for (let y=0; y<height; y++) for (let x=0; x<width; x++) {
    let stable = true;
    for (let dy=-1; dy<=1 && stable; dy++) for (let dx=-1; dx<=1; dx++) {
      const xx=x+dx, yy=y+dy;
      if (xx>=0 && xx<width && yy>=0 && yy<height && changed[yy*width+xx]) { stable=false; break; }
    }
    const offset = (y*width+x)*4;
    mask[offset]=mask[offset+1]=mask[offset+2]=255;
    mask[offset+3]=stable?255:0;
  }
  return mask;
}
