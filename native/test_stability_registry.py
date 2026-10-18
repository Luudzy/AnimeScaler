import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from model_registry import ModelRegistry
from zenkai_realesr_deploy import ANIMEVIDEO_PACKAGE, ANIMEVIDEO_MODEL_ID, register_zenkai_realesr


class IndependentZenkaiRegistrationTests(unittest.TestCase):
    def test_animevideo_graphs_stay_disabled_after_live_regression(self):
        from types import SimpleNamespace
        from server import Renderer
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); weights=root / ANIMEVIDEO_PACKAGE / 'deploy-fp16.pt'
            weights.parent.mkdir(parents=True); weights.write_bytes(b'fixture')
            for enabled in ('1','0'):
                with patch.dict(os.environ,{'ANIMESCALER_ANIMEVIDEO_GRAPH':enabled}):
                    registry=ModelRegistry();register_zenkai_realesr(registry,root)
                    metadata=registry.get(ANIMEVIDEO_MODEL_ID)
                    self.assertFalse(metadata.capture_graph,'never allow unbounded graphs')
                    self.assertEqual(metadata.capture_graph_max_pixels,0)
                    renderer=Renderer.__new__(Renderer);renderer.use_graph=True
                    renderer.model=SimpleNamespace(metadata=metadata)
                    self.assertFalse(renderer.can_capture(360,640))
                    self.assertFalse(renderer.can_capture(720,1280))
                    renderer.use_graph=False
                    self.assertFalse(renderer.can_capture(360,640),'global HIP graph override remains authoritative')

    def test_animevideo_registers_and_checks_hash_without_clean5k(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / ANIMEVIDEO_PACKAGE / 'deploy-fp16.pt'
            weights.parent.mkdir(parents=True)
            weights.write_bytes(b'independent-animevideo-artifact')
            registry = ModelRegistry()
            register_zenkai_realesr(registry, root)
            self.assertEqual(registry.get(ANIMEVIDEO_MODEL_ID).weights_path, str(weights))
            with patch('model_registry.TorchBackend.load') as load:
                with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                    registry.load(ANIMEVIDEO_MODEL_ID)
                load.assert_not_called()
