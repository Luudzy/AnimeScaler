"""Bounded little-endian RGBA protocol shared with the browser worker."""
import struct

HEADER = struct.Struct('<6I')
MAGIC = 0x31435341  # ASC1
MAX_PIXELS = 1920 * 1080


def decode_frame(data):
    if len(data) < HEADER.size:
        raise ValueError('Missing frame header')
    magic, frame_id, width, height, out_width, out_height = HEADER.unpack_from(data)
    if magic != MAGIC:
        raise ValueError('Unsupported protocol')
    if (min(width, height) < 40 or width % 4 or height % 4
            or width * height > MAX_PIXELS or max(width, height) > 1920):
        raise ValueError('Invalid input dimensions')
    if not (2 <= out_width <= 3840 and 2 <= out_height <= 3840) or out_width % 2 or out_height % 2 or out_width * out_height > MAX_PIXELS * 4:
        raise ValueError('Invalid output dimensions')
    if len(data) != HEADER.size + width * height * 4:
        raise ValueError('Invalid RGBA payload length')
    return frame_id, width, height, out_width, out_height
