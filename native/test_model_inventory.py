from pathlib import Path
import hashlib
import runpy
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
inventory_module = runpy.run_path(str(ROOT / 'scripts' / 'inventory-models.py'))
build_storage_inventory = inventory_module['build_storage_inventory']
sha256_file = inventory_module['sha256_file']


class ModelInventoryTests(unittest.TestCase):
    def test_inventory_totals_protects_models_and_runtime_caches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = {
                'models/example.onnx': b'model',
                'diagnostics/native-cache/tensorrt/example.engine': b'engine bytes',
                'diagnostics/native-cache/torch/weights.bin': b'torch cache',
                'native/training-data/private.bin': b'private data',
                '.env': b'secret=hidden',
            }
            for relative, content in files.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)

            report = build_storage_inventory(root)

        self.assertEqual(report['totals'], {'files': 3, 'bytes': 5 + 12 + 11})
        self.assertEqual([item['path'] for item in report['files']], [
            'diagnostics/native-cache/tensorrt/example.engine',
            'diagnostics/native-cache/torch/weights.bin',
            'models/example.onnx',
        ])
        self.assertTrue(all(item['protected'] for item in report['files']))
        self.assertTrue(all(item['retention'] == 'user_authorization_required' for item in report['files']))

    def test_missing_inventory_roots_return_empty_totals(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = build_storage_inventory(Path(temporary))
        self.assertEqual(report['totals'], {'files': 0, 'bytes': 0})
        self.assertEqual(report['files'], [])

    def test_model_hash_matches_across_bounded_chunks(self):
        contents = b'anime-model-artifact' * 9000
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'model.onnx'
            path.write_bytes(contents)
            self.assertEqual(sha256_file(path, chunk_size=127), hashlib.sha256(contents).hexdigest())


if __name__ == '__main__':
    unittest.main()
