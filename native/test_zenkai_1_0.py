import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from model_registry import ModelRegistry, build_registry

ROOT = Path(__file__).resolve().parents[1]


class ZenkaiOneTests(unittest.TestCase):
    def test_renderer_smoke_uses_actual_bridge_input_limits(self):
        spec = importlib.util.spec_from_file_location('zenkai_one_validation', ROOT / 'diagnostics/zenkai-overnight/validate_zenkai_1_0.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(hasattr(module, 'renderer_cases'), 'Renderer samples must be distinct from odd-shape model parity cases')
        from protocol import HEADER, MAGIC, decode_frame
        cases = module.renderer_cases()
        self.assertEqual(len(cases), 3)
        for w, h, ow, oh in cases:
            data = HEADER.pack(MAGIC, 1, w, h, ow, oh) + bytes(w * h * 4)
            self.assertEqual(decode_frame(data), (1, w, h, ow, oh))
        self.assertIn((96, 64, 144, 96), cases)

    def modules(self):
        for name in ('zenkai_1_0', 'build_zenkai_1_0'):
            self.assertIsNotNone(importlib.util.find_spec(name), 'Missing Zenkai 1.0 implementation: ' + name)
        return importlib.import_module('zenkai_1_0'), importlib.import_module('build_zenkai_1_0')

    @unittest.skipUnless((ROOT/'models/zenkai/experiments/zenkai-lite-v1-c-codec50-5k/step-5000-master-fp32.pt').is_file(), 'download the pinned Zenkai source checkpoint first')
    def test_package_preserves_source_and_refuses_overwrite(self):
        module, builder = self.modules()
        source = ROOT / 'models/zenkai/experiments/zenkai-lite-v1-c-codec50-5k/step-5000-master-fp32.pt'
        pin = hashlib.sha256(source.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            folder = builder.build_package(Path(tmp), source_root=ROOT)
            manifest = module.validate_package(folder)
            self.assertEqual(manifest['sourceCheckpointSha256'], pin)
            self.assertFalse(manifest['newTrainingExecuted'])
            self.assertEqual(manifest['qualityAdvancement'], 'FAIL')
            self.assertFalse(manifest['releaseApproved'])
            self.assertEqual(manifest['recipe']['codecFraction'], .5)
            self.assertEqual(manifest['recipe']['patchLR'], 128)
            self.assertEqual(manifest['recipe']['spabIdentitySkipGain'], 1.)
            with self.assertRaises(FileExistsError):
                builder.build_package(Path(tmp), source_root=ROOT)
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), pin)

    def test_missing_package_is_optional_and_incomplete_package_fails_closed(self):
        module, _ = self.modules()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = ModelRegistry()
            self.assertIsNone(module.register_zenkai_1_0(registry, root))
            self.assertEqual(registry.list_models(), [])
            folder = root / module.PACKAGE_RELATIVE
            folder.mkdir(parents=True)
            (folder / 'manifest.json').write_text(json.dumps({'id': module.MODEL_ID}), encoding='utf-8')
            with self.assertRaises(ValueError):
                module.register_zenkai_1_0(registry, root)
            self.assertEqual(registry.list_models(), [])

    @unittest.skipUnless((ROOT/'models/zenkai/experiments/zenkai-lite-v1-c-codec50-5k/step-5000-master-fp32.pt').is_file(), 'download the pinned Zenkai source checkpoint first')
    def test_registered_family_preserves_trained_identity_skip_on_odd_sizes(self):
        module, builder = self.modules()
        from export_zenkai_skip import load_candidate
        source, _ = load_candidate(ROOT / 'models/zenkai/experiments/zenkai-lite-v1-c-codec50-5k/step-5000-master-fp32.pt')
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            builder.build_package(root, source_root=ROOT)
            registry = build_registry(root)
            meta = registry.get(module.MODEL_ID)
            self.assertEqual(meta.name, 'Zenkai 1.0')
            self.assertEqual(meta.status, 'experimental')
            self.assertFalse(meta.capture_graph)
            model = registry.load(module.MODEL_ID, device='cpu', precision='fp32')
            self.assertEqual(model.backend.model.config['family'], 'ZenkaiIdentitySkipProbe')
            for h, w in ((1, 1), (17, 19), (23, 31)):
                x = torch.rand(1, 3, h, w)
                with torch.inference_mode():
                    expected, actual = source(x), model.infer(x)
                self.assertEqual(tuple(actual.shape), (1, 3, h * 2, w * 2))
                self.assertLessEqual(float((expected - actual).abs().max()), 2e-5)
            registry.unload(module.MODEL_ID)

    @unittest.skipUnless((ROOT/'models/zenkai/experiments/zenkai-lite-v1-c-codec50-5k/step-5000-master-fp32.pt').is_file(), 'download the pinned Zenkai source checkpoint first')
    def test_changed_artifact_rejected_before_any_backend_initializes(self):
        module, builder = self.modules()
        for filename, backend, cls in (
            ('deploy-fp32.pt', 'torch', 'TorchBackend'),
            ('dynamic-fp32.onnx', 'onnx', 'OnnxBackend'),
            ('dynamic-fp32.onnx', 'tensorrt', 'TensorRTBackend'),
        ):
            with self.subTest(filename=filename, backend=backend), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                folder = builder.build_package(root, source_root=ROOT)
                registry = build_registry(root)
                (folder / filename).write_bytes(b'corrupted after registration')
                with patch('model_registry.' + cls + '.load') as loader:
                    with self.assertRaises(ValueError):
                        registry.load(module.MODEL_ID, backend=backend, root=root)
                    loader.assert_not_called()

    @unittest.skipUnless((ROOT/'models/zenkai/experiments/zenkai-lite-v1-c-codec50-5k/step-5000-master-fp32.pt').is_file(), 'download the pinned Zenkai source checkpoint first')
    def test_wrong_version_recipe_or_quality_promotion_rejected(self):
        module, builder = self.modules()
        with tempfile.TemporaryDirectory() as tmp:
            folder = builder.build_package(Path(tmp), source_root=ROOT)
            path = folder / 'manifest.json'
            original = json.loads(path.read_text(encoding='utf-8'))
            changes = [{'version': '2.0.0'}, {'releaseApproved': True}, {'qualityAdvancement': 'PASS'},
                       {'family': 'Zenkai'}, {'recipe': {**original['recipe'], 'codecFraction': .25}}]
            for fields in changes:
                path.write_text(json.dumps({**original, **fields}), encoding='utf-8')
                with self.subTest(fields=fields), self.assertRaises(ValueError):
                    module.validate_package(folder)


if __name__ == '__main__':
    unittest.main()
