import unittest
from pathlib import Path
import importlib.util
import os

MODEL_DIR = Path(os.environ.get('ANIMESCALER_NCNN_MODELS', Path(__file__).resolve().parents[1] / 'models'))
NCNN_AVAILABLE = importlib.util.find_spec('ncnn') is not None
MODEL_AVAILABLE = all((MODEL_DIR / ('realesr-animevideov3-x2' + suffix)).is_file() for suffix in ('.param', '.bin'))


class NcnnVulkanBackendTests(unittest.TestCase):
    @unittest.skipUnless(NCNN_AVAILABLE and MODEL_AVAILABLE, 'Optional NCNN/Vulkan integration requires ncnn and AnimeVideo x2 .param/.bin; set ANIMESCALER_NCNN_MODELS')
    def test_loads_animevideo_v3_and_returns_requested_x2_shape(self):
        from ncnn_vulkan_backend import NcnnVulkanBackend
        backend = NcnnVulkanBackend(MODEL_DIR, 'realesr-animevideov3-x2')
        try:
            self.assertTrue(backend.vulkan_available)
            output = backend.infer_bytes(bytes(640 * 360 * 3), 640, 360, 2)
            self.assertEqual(output.shape, (720, 1280, 3))
        finally:
            backend.close()


if __name__ == '__main__':
    unittest.main()
