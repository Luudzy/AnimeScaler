export function alignTo(value, alignment = 16) {
  if (!Number.isFinite(value) || value < 0) throw new RangeError('value must be non-negative');
  if (!Number.isInteger(alignment) || alignment <= 0) throw new RangeError('alignment must be a positive integer');
  return Math.ceil(value / alignment) * alignment;
}

export function nchwFloatBytes(width, height, alignment = 4) {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
    throw new RangeError('width and height must be positive integers');
  }
  return alignTo(1 * 3 * width * height * Float32Array.BYTES_PER_ELEMENT, alignment);
}

export function outputShape2x(inputWidth, inputHeight) {
  return [1, 3, inputHeight * 2, inputWidth * 2];
}
