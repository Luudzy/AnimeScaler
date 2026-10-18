import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from PIL import Image

from prepare_video_degradation import FFMPEG, ROOT, frame_timing, prepare_video, validate_options


class VideoDegradationTests(unittest.TestCase):
    def test_invalid_settings_and_directories_rejected(self):
        for codec, crf, gop, count in [
            ('other', 24, 48, 6), ('h264', 52, 48, 6),
            ('av1', 64, 48, 6), ('h265', 24, 0, 6), ('h264', 24, 48, 1),
        ]:
            with self.subTest(codec=codec, crf=crf, gop=gop, count=count):
                with self.assertRaises(ValueError):
                    validate_options(codec, crf, gop, count)
        with self.assertRaises(ValueError):
            prepare_video(Path(__file__).parent, Path('unused'))

    def test_missing_reordered_or_duplicate_pts_cannot_establish_alignment(self):
        header = 'config in time_base: 1/1000, frame_rate: 25/1\n'
        for frames in [
            [(0, 0)],
            [(0, 0), (2, 40)],
            [(0, 0), (1, 0)],
            [(0, 40), (1, 0)],
        ]:
            log = header + '\n'.join(
                f'n: {n} pts: {pts} pts_time:0 iskey:1 type:I' for n, pts in frames)
            with self.subTest(frames=frames):
                with self.assertRaises(ValueError):
                    frame_timing(log, 2)

    def test_cumulative_pts_drift_is_rejected(self):
        log = 'config in time_base: 1/1000, frame_rate: 25/1\n' + '\n'.join(
            f'n: {n} pts: {pts} pts_time:0 iskey:1 type:I'
            for n, pts in enumerate([0, 41, 82, 123]))
        with self.assertRaisesRegex(ValueError, 'Variable or discontinuous'):
            frame_timing(log, 4)

    def test_changed_codec_fps_is_rejected(self):
        log = 'config in time_base: 1/1000, frame_rate: 25/1\n'
        with self.assertRaisesRegex(ValueError, 'changed frame rate'):
            frame_timing(log, 2, '24/1')


@unittest.skipUnless(FFMPEG.is_file(), 'Bundled CPU ffmpeg unavailable')
class VideoDegradationIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='video-test-', dir=ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)

    def make_clip(self, *, rate='12', count=12, filters=None):
        source = self.folder / 'synthetic.mkv'
        command = [str(FFMPEG), '-nostdin', '-v', 'error', '-f', 'lavfi',
                   '-i', f'testsrc2=size=102x70:rate={rate}']
        if filters:
            command += ['-vf', filters]
        command += ['-frames:v', str(count), '-fps_mode', 'passthrough',
                    '-c:v', 'ffv1', '-threads', '2', str(source)]
        subprocess.run(command, check=True, capture_output=True, timeout=30)
        return source

    def test_all_codecs_preserve_alignment_and_record_real_gop(self):
        source = self.make_clip()
        for codec in ('h264', 'h265', 'av1'):
            with self.subTest(codec=codec):
                output = self.folder / codec
                report = prepare_video(source, output, codec=codec, gop=4,
                                       max_frames=10, synthetic=True)
                self.assertEqual(report['frameCount'], 10)
                self.assertEqual(report['fps'], '12/1')
                self.assertFalse(report['realVideoValidation'])
                self.assertTrue(report['syntheticSource'])
                self.assertEqual(report['sourceSize'], [102, 70])
                self.assertEqual(report['crop'], [0, 0, 100, 68])
                self.assertEqual(report['codec']['gop'], 4)
                self.assertTrue(report['codec'].get('interFrameObserved'), report['codec'])
                self.assertEqual(report['codec']['keyframeIndices'], [0, 4, 8])
                self.assertLessEqual(report['codec']['maxObservedGopFrames'], 4)
                self.assertEqual(len(list((output / 'hr').glob('*.png'))), 10)
                self.assertEqual(len(list((output / 'lr').glob('*.png'))), 10)
                for index, row in enumerate(report['frames']):
                    self.assertEqual(row['index'], index)
                    self.assertAlmostEqual(row['sourcePtsSeconds'], index / 12, delta=.0011)
                    self.assertAlmostEqual(row['encodedPtsSeconds'], index / 12, delta=.0011)
                    with Image.open(output / row['hr']) as hr, Image.open(output / row['lr']) as lr:
                        self.assertEqual(hr.size, (100, 68))
                        self.assertEqual(lr.size, (50, 34))
                    for kind in ('hr', 'lr'):
                        self.assertEqual(row[kind + 'Sha256'], hashlib.sha256(
                            (output / row[kind]).read_bytes()).hexdigest())
                self.assertEqual(json.loads((output / 'complete.json').read_text()), report)

    def test_nonzero_large_source_timestamps_are_preserved(self):
        source = self.make_clip(count=4, filters='setpts=PTS+100000/TB')
        report = prepare_video(source, self.folder / 'pairs', max_frames=4, synthetic=True)
        self.assertAlmostEqual(report['frames'][0]['sourcePtsSeconds'], 100000, places=6)
        self.assertAlmostEqual(report['frames'][3]['sourcePtsSeconds'], 100000.25, places=6)
        self.assertEqual(report['frames'][0]['sourcePts'], 100000000)
        self.assertAlmostEqual(report['frames'][0]['encodedPtsSeconds'], 0, places=6)

    def test_image_disguised_as_video_is_rejected(self):
        source = self.folder / 'image.mkv'
        frames = [Image.new('RGB', (32, 32), (i * 30, 0, 0)) for i in range(6)]
        frames[0].save(source, format='GIF', save_all=True, append_images=frames[1:], duration=100)
        output = self.folder / 'pairs'
        with self.assertRaises((RuntimeError, ValueError)):
            prepare_video(source, output, max_frames=6)
        self.assertFalse((output / 'complete.json').exists())

    def test_vfr_sequence_is_rejected_without_completion_marker(self):
        source = self.make_clip(count=8, filters="select='not(eq(n,3))'")
        output = self.folder / 'pairs'
        with self.assertRaisesRegex(ValueError, 'Variable or discontinuous'):
            prepare_video(source, output, max_frames=8, synthetic=True)
        self.assertFalse((output / 'complete.json').exists())

    def test_fractional_fps_is_preserved(self):
        source = self.make_clip(count=8, rate='24000/1001')
        report = prepare_video(source, self.folder / 'pairs', max_frames=8, synthetic=True)
        self.assertEqual(report['fps'], '24000/1001')
        self.assertAlmostEqual(report['frames'][7]['sourcePtsSeconds'], 7 * 1001 / 24000, delta=.0011)

    def test_intra_only_request_is_reported_as_intra_only(self):
        source = self.make_clip(count=4)
        report = prepare_video(source, self.folder / 'pairs', gop=1, max_frames=4, synthetic=True)
        self.assertTrue(report['codec']['intraOnlyRequested'])
        self.assertEqual(report['codec'].get('keyframeIndices'), [0, 1, 2, 3])
        self.assertFalse(report['codec']['interFrameObserved'])

    def test_existing_output_is_untouched(self):
        source = self.make_clip(count=4)
        output = self.folder / 'pairs'
        output.mkdir()
        sentinel = output / 'keep.txt'
        sentinel.write_text('keep')
        with self.assertRaisesRegex(ValueError, 'already exists'):
            prepare_video(source, output, max_frames=4, synthetic=True)
        self.assertEqual(sentinel.read_text(), 'keep')
        self.assertEqual(list(output.iterdir()), [sentinel])


if __name__ == '__main__':
    unittest.main()
