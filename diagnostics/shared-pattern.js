export function sharedPattern(width, height) {
  const rgba = new Uint8ClampedArray(width * height * 4);
  const nchw = new Float32Array(width * height * 3);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const index = y * width + x;
    for (let c = 0; c < 3; c++) {
      const value = (x * 7 + y * 3 + c * 47) % 256;
      rgba[index * 4 + c] = value;
      nchw[c * width * height + index] = value / 255;
    }
    rgba[index * 4 + 3] = 255;
  }
  return {rgba, nchw};
}
