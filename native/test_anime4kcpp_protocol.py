import struct
import unittest

from anime4kcpp_protocol import HEADER, MAGIC, RESPONSE_MAGIC, pack_request, unpack_response, unpack_response_header


class Anime4KCPPProtocolTests(unittest.TestCase):
    def test_request_header_matches_native_layout(self):
        header = pack_request(7, 1920, 1080, 3840, 2160)
        self.assertEqual(len(header), HEADER.size)
        self.assertEqual(struct.unpack('<6I', header), (MAGIC, 7, 1920, 1080, 3840, 2160))

    def test_response_round_trip_validates_rgba_payload(self):
        payload = bytes([1, 2, 3, 255]) * 8
        response = HEADER.pack(RESPONSE_MAGIC, 11, 4, 2, 4, 2) + payload
        frame_id, width, height, output = unpack_response(response)
        self.assertEqual((frame_id, width, height), (11, 4, 2))
        self.assertEqual(output, payload)

    def test_response_header_rejects_wrong_frame_before_payload_read(self):
        header=HEADER.pack(RESPONSE_MAGIC,12,40,40,80,80)
        with self.assertRaisesRegex(ValueError,'frame id'):
            unpack_response_header(header,expected_frame_id=11,expected_output_size=(80,80))

    def test_response_header_rejects_dimensions_outside_request(self):
        header=HEADER.pack(RESPONSE_MAGIC,11,40,40,0,80)
        with self.assertRaisesRegex(ValueError,'dimensions'):
            unpack_response_header(header,expected_frame_id=11,expected_output_size=(80,80))


if __name__ == '__main__':
    unittest.main()
