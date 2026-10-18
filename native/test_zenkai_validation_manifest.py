"""CPU tests for replayable validation and protected source/split boundaries."""
import hashlib
import importlib.util
import json
import platform
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image


class ValidationManifestTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('zenkai_validation_manifest'),
                             'The fixed validation manifest builder is missing')
        import zenkai_validation_manifest
        self.api = zenkai_validation_manifest
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'approved'
        self.source.mkdir()
        self.dataset = self.root / 'dataset'
        self.dataset.mkdir()
        self.rows = []
        for index in range(2):
            path = self.source / f'{index}.png'
            pixels = np.random.default_rng(index).integers(0, 256, (296, 304, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(path)
            self.rows.append({'file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                              'domain': 'anime', 'split': 'validation', 'cluster': f'group-{index}'})
        self.save_rows(self.rows)

    def save_rows(self, rows):
        (self.dataset / 'manifest.jsonl').write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')

    def build(self, **kwargs):
        return self.api.build_validation_manifest(self.dataset, allowed_roots=[self.source], **kwargs)

    def test_replay_persists_crop_and_seed_and_rejects_changed_pixels(self):
        report = self.build()
        sample = next(s for s in report['samples'] if s['case'] == 'heavy')
        self.assertEqual(report['patchLR'], 128)
        self.assertEqual(sample['cropHR']['width'], 256)
        self.assertEqual(sample['cropHR']['height'], 256)
        self.assertIsInstance(sample['degradationSeed'], int)
        first_lr, first_hr = self.api.render_validation_sample(sample, allowed_roots=[self.source])
        second_lr, second_hr = self.api.render_validation_sample(sample, allowed_roots=[self.source])
        np.testing.assert_array_equal(first_lr, second_lr)
        np.testing.assert_array_equal(first_hr, second_hr)
        self.assertEqual(first_lr.size, (128, 128))
        with Image.open(sample['sourceFile']) as source:
            crop = sample['cropHR']
            expected = source.convert('RGB').crop((crop['left'], crop['top'],
                                                   crop['left'] + 256, crop['top'] + 256))
        np.testing.assert_array_equal(first_hr, expected)
        self.assertEqual(hashlib.sha256(first_lr.tobytes()).hexdigest(), sample['lrPixelsSha256'])
        Image.new('RGB', (304, 296), 'white').save(sample['sourceFile'])
        with self.assertRaisesRegex(ValueError, 'Source checksum'):
            self.api.render_validation_sample(sample, allowed_roots=[self.source])

    def test_reordering_or_subsetting_sources_preserves_each_sample(self):
        original = self.build(per_image=2)
        self.save_rows(list(reversed(self.rows)))
        reordered = self.build(per_image=2)
        self.assertEqual(original['samples'], reordered['samples'])
        self.assertNotEqual(original['sourceManifestSha256'], reordered['sourceManifestSha256'])
        self.save_rows(self.rows[:1])
        subset = self.build(per_image=2)
        retained = [s for s in original['samples'] if s['sourceSha256'] == self.rows[0]['sha256']]
        self.assertEqual(retained, subset['samples'])

    def test_never_opens_training_or_final_hidden_sources(self):
        blocked = [{'file': str(self.root / f'missing-{split}.png'), 'split': split,
                    'domain': 'anime', 'sha256': '0' * 64} for split in ('train', 'test', 'final-hidden')]
        self.save_rows(self.rows + blocked)
        report = self.build()
        self.assertEqual(len(report['samples']), 4)
        self.assertTrue(all(s['split'] == 'validation' for s in report['samples']))

    def test_rejects_source_outside_approved_root_before_opening(self):
        self.rows[0]['file'] = str(self.root / 'approved-other' / 'missing.png')
        self.save_rows(self.rows)
        with self.assertRaisesRegex(ValueError, 'outside approved'):
            self.build()

    def test_record_rejects_hidden_and_invalid_crop(self):
        sample = self.build()['samples'][0]
        hidden = {**sample, 'split': 'final-hidden'}
        with self.assertRaisesRegex(ValueError, 'validation only'):
            self.api.render_validation_sample(hidden, allowed_roots=[self.source])
        invalid = {**sample, 'cropHR': {**sample['cropHR'], 'left': -1}}
        with self.assertRaisesRegex(ValueError, 'crop'):
            self.api.render_validation_sample(invalid, allowed_roots=[self.source])

    def test_record_rejects_invalid_crop_geometry_scale_and_case(self):
        sample = self.build()['samples'][0]
        invalid_crops = (
            {'left': True},
            {'left': '0'},
            {'top': -1},
            {'width': 255},
            {'height': 0},
            {'left': 49},
        )
        for updates in invalid_crops:
            with self.subTest(crop_updates=updates):
                invalid = {**sample, 'cropHR': {**sample['cropHR'], **updates}}
                with self.assertRaisesRegex(ValueError, 'crop'):
                    self.api.render_validation_sample(invalid, allowed_roots=[self.source])

        with self.assertRaisesRegex(ValueError, 'native x2'):
            self.api.render_validation_sample({**sample, 'scale': 4}, allowed_roots=[self.source])
        with self.assertRaisesRegex(ValueError, 'native x2'):
            self.api.render_validation_sample({**sample, 'case': 'codec'}, allowed_roots=[self.source])

    def test_record_rejects_changed_hr_crop_hash(self):
        sample = self.build()['samples'][0]
        changed = {**sample, 'hrPixelsSha256': '0' * 64}
        with self.assertRaisesRegex(ValueError, 'HR crop pixels'):
            self.api.render_validation_sample(changed, allowed_roots=[self.source])

    def test_rejects_animated_validation_source(self):
        path = self.source / 'animated.gif'
        first = Image.new('RGB', (304, 296), 'black')
        second = Image.new('RGB', (304, 296), 'white')
        first.save(path, save_all=True, append_images=[second], duration=100, loop=0)
        row = {'file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
               'domain': 'anime', 'split': 'validation', 'cluster': 'animated'}
        self.save_rows([row])

        with self.assertRaisesRegex(ValueError, 'Only static validation images'):
            self.build()

    def test_manifest_records_hash_bound_provenance(self):
        report = self.build()
        provenance = report['provenance']
        self.assertEqual(report['schemaVersion'], 1)
        self.assertEqual(report['sourceManifestSha256'],
                         hashlib.sha256((self.dataset / 'manifest.jsonl').read_bytes()).hexdigest())
        self.assertEqual(provenance['builderSha256'],
                         hashlib.sha256(Path(self.api.__file__).read_bytes()).hexdigest())
        import lite_degradations
        self.assertEqual(provenance['degradationSha256'],
                         hashlib.sha256(Path(lite_degradations.__file__).read_bytes()).hexdigest())
        self.assertEqual(provenance['python'], platform.python_version())
        self.assertEqual(provenance['pillow'], self.api.PIL.__version__)
        self.assertEqual(provenance['numpy'], np.__version__)

    def test_writer_refuses_overwrite_and_preserves_source_manifest(self):
        output = self.root / 'validation.json'
        original = (self.dataset / 'manifest.jsonl').read_bytes()
        report = self.api.write_validation_manifest(self.dataset, output, allowed_roots=[self.source])
        self.assertEqual(json.loads(output.read_text()), report)
        self.assertEqual((self.dataset / 'manifest.jsonl').read_bytes(), original)
        saved = output.read_bytes()
        with self.assertRaises(FileExistsError):
            self.api.write_validation_manifest(self.dataset, output, allowed_roots=[self.source])
        self.assertEqual(output.read_bytes(), saved)

    def test_writer_leaves_no_partial_manifest_when_atomic_publish_fails(self):
        output = self.root / 'validation.json'
        with patch('artifact_io.os.link', side_effect=OSError('publish failed')):
            with self.assertRaisesRegex(OSError, 'publish failed'):
                self.api.write_validation_manifest(self.dataset, output, allowed_roots=[self.source])
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob('.validation.json.*.pending')), [])

    def test_rejects_empty_validation_and_invalid_dimensions(self):
        with self.assertRaisesRegex(ValueError, 'patch_lr'):
            self.build(patch_lr=0)
        with self.assertRaisesRegex(ValueError, 'per_image'):
            self.build(per_image=0)
        with self.assertRaisesRegex(ValueError, 'too small'):
            self.build(patch_lr=160)
        self.save_rows([{**self.rows[0], 'split': 'train'}])
        with self.assertRaisesRegex(ValueError, 'No validation'):
            self.build()

    def test_replay_detects_degradation_output_drift(self):
        sample = self.build()['samples'][0]
        changed = {**sample, 'lrPixelsSha256': '0' * 64}
        with self.assertRaisesRegex(ValueError, 'Degradation output'):
            self.api.render_validation_sample(changed, allowed_roots=[self.source])


if __name__ == '__main__':
    unittest.main()
