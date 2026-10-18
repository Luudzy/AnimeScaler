import unittest
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


class CompactReference(nn.Module):
    """Independent original SRVGG equation; can load the actual supplied weights."""
    def __init__(self, features=8, depth=3):
        super().__init__()
        self.upscale = 4
        self.body = nn.ModuleList([nn.Conv2d(3, features, 3, padding=1), nn.PReLU(features)])
        for _ in range(depth):
            self.body.extend([nn.Conv2d(features, features, 3, padding=1), nn.PReLU(features)])
        self.body.append(nn.Conv2d(features, 48, 3, padding=1))
        self.upsampler = nn.PixelShuffle(4)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        return self.upsampler(out) + F.interpolate(x, scale_factor=4, mode='nearest')


class RealESROptimizedTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(929)

    def optimized(self, model, **kwargs):
        import realesr_optimized
        return realesr_optimized.RealESRAnimeVideo(model.eval(), **kwargs).eval()

    def test_reduced_heads_match_original_resize_including_edges(self):
        model = CompactReference().eval()
        optimized = self.optimized(model)
        for h, w in ((8, 10), (30, 46), (40, 48)):
            x = torch.rand(2, 3, h, w)
            with torch.inference_mode():
                for size in ((h*2,w*2), (h*3//2,w*3//2), (h*4,w*4), (h+3,w+5)):
                    expected = F.interpolate(model(x), size=size, mode='bilinear', align_corners=False)
                    actual = optimized.forward_to(x, size)
                    self.assertEqual(actual.shape, expected.shape)
                    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)

    def test_exact_tiling_has_no_seams_or_blending(self):
        model = CompactReference().eval()
        optimized = self.optimized(model, max_pixels=1, tile=16)
        for h,w in ((30,46), (31,47), (8,10)):
            x = torch.rand(1,3,h,w)
            with torch.inference_mode():
                for size in ((h*2,w*2), (h*4,w*4), (h*3//2,w*3//2)):
                    expected = F.interpolate(model(x),size=size,mode='bilinear',align_corners=False)
                    torch.testing.assert_close(optimized.forward_to(x,size),expected,atol=2e-6,rtol=2e-6)

    def test_actual_checkpoint_preserved_and_reduced_heads_equivalent(self):
        path = Path(__file__).resolve().parents[1] / 'models/realesr-animevideov3.pth'
        if not path.exists():
            self.skipTest('User supplied checkpoint absent')
        model = CompactReference(64,16).eval()
        state = torch.load(path,map_location='cpu',weights_only=True)['params']
        model.load_state_dict(state)
        original = {k:v.clone() for k,v in model.state_dict().items()}
        optimized = self.optimized(model,max_pixels=1,tile=24)
        x = torch.rand(1,3,40,48)
        with torch.inference_mode():
            for size in ((80,96),(60,72),(160,192)):
                expected = F.interpolate(model(x),size=size,mode='bilinear',align_corners=False)
                torch.testing.assert_close(optimized.forward_to(x,size),expected,atol=3e-5,rtol=3e-5)
        for k,v in model.state_dict().items():
            self.assertTrue(torch.equal(v,original[k]),k)

    def test_other_architectures_and_invalid_tile_fail_closed(self):
        with self.assertRaises(ValueError):
            self.optimized(nn.Identity())
        for tile in (0,-2,15):
            with self.assertRaises(ValueError):
                self.optimized(CompactReference(),tile=tile)

    def test_eval_only_and_forward_retains_native_four_x(self):
        optimized = self.optimized(CompactReference())
        x = torch.rand(1,3,9,11)
        self.assertEqual(optimized(x).shape,(1,3,36,44))
        self.assertFalse(any(p.requires_grad for p in optimized.parameters()))

    def test_registry_uses_optimized_path_with_explicit_original_fallback(self):
        import tempfile,os
        from unittest.mock import patch
        from model_registry import build_registry
        from tiled_model import TiledModel
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'models').mkdir()
            (root/'models/realesr-animevideov3.pth').write_bytes(b'fixture weights')
            for enabled in ('1','0'):
                with patch.dict(os.environ,{'ANIMESCALER_REALESR_OPTIMIZE':enabled}),patch(
                        'user_models.load_user_model',side_effect=lambda *args:(CompactReference().eval(),{'scale':4,'supportsHalf':True})):
                    registry=build_registry(root)
                    model=registry.load('realesr-anime')
                    self.assertEqual(model.metadata.native_scale,4)
                    self.assertEqual(model.metadata.capture_graph_max_pixels,640*360 if enabled=='1' else 0)
                    self.assertEqual(isinstance(model.backend.model,TiledModel),enabled=='0')
                    self.assertEqual(model.infer_to(torch.rand(1,3,10,12),(20,24)).shape,(1,3,20,24))


if __name__ == '__main__':
    unittest.main()
