"""Binary framing shared by the Anime4KCPP worker and the native bridge."""
import struct

HEADER = struct.Struct('<6I')
MAGIC = 0x31435341
RESPONSE_MAGIC = 0x33435341
MAX_INPUT_PIXELS = 1920 * 1080
MAX_OUTPUT_PIXELS = MAX_INPUT_PIXELS * 4


def pack_request(frame_id, width, height, output_width, output_height):
    if min(width, height) < 40 or width % 4 or height % 4:
        raise ValueError('invalid input dimensions')
    if width * height > MAX_INPUT_PIXELS:
        raise ValueError('input exceeds worker limit')
    if output_width % 2 or output_height % 2 or output_width * output_height > MAX_OUTPUT_PIXELS:
        raise ValueError('invalid output dimensions')
    return HEADER.pack(MAGIC, frame_id, width, height, output_width, output_height)


def unpack_response_header(data, expected_frame_id=None, expected_input_size=None, expected_output_size=None):
    if len(data)!=HEADER.size:
        raise ValueError('invalid worker response header length')
    magic,frame_id,width,height,output_width,output_height=HEADER.unpack(data)
    if magic!=RESPONSE_MAGIC:
        raise ValueError('invalid worker response magic')
    if expected_frame_id is not None and frame_id!=expected_frame_id:
        raise ValueError('invalid worker response frame id')
    if min(width,height)<1 or width*height>MAX_INPUT_PIXELS:
        raise ValueError('invalid worker response input dimensions')
    if expected_input_size is not None and (width,height)!=tuple(expected_input_size):
        raise ValueError('invalid worker response input dimensions')
    if min(output_width,output_height)<2 or output_width%2 or output_height%2 or output_width*output_height>MAX_OUTPUT_PIXELS:
        raise ValueError('invalid worker response output dimensions')
    if expected_output_size is not None and (output_width,output_height)!=tuple(expected_output_size):
        raise ValueError('invalid worker response dimensions')
    return frame_id,output_width,output_height


def unpack_response(data):
    if len(data) < HEADER.size:
        raise ValueError('short worker response')
    frame_id,output_width,output_height=unpack_response_header(data[:HEADER.size])
    expected = HEADER.size + output_width * output_height * 4
    if len(data) != expected:
        raise ValueError('invalid worker response payload')
    return frame_id, output_width, output_height, data[HEADER.size:]
