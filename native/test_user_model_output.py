import unittest
import torch
from types import SimpleNamespace
from server import Renderer

class OutputScaleTests(unittest.TestCase):
    def test_renderer_uses_requested_size_before_allocating_native_output(self):
        from test_realesr_optimized import CompactReference
        from inference_backends import TorchBackend
        from model_registry import SRModel, ModelMetadata
        from realesr_optimized import RealESRAnimeVideo
        raw=CompactReference().eval()
        optimized=RealESRAnimeVideo(raw)
        backend=TorchBackend(precision='fp16').load(lambda:optimized)
        r=Renderer.__new__(Renderer)
        r.torch=torch;r.shape=(40,48,60,72);r.memory_format=torch.contiguous_format
        r.gpu_input=torch.full((40,48,4),128,dtype=torch.uint8)
        r.gpu_output=torch.zeros((60,72,3),dtype=torch.uint8)
        r.model=SRModel(ModelMetadata('test','Test','External','SRVGG',4,'test'),backend,None)
        # Native x4 inference must not be called for an optimized output head.
        raw.forward=lambda x:(_ for _ in ()).throw(AssertionError('Native x4 allocation'))
        optimized.forward=lambda x:(_ for _ in ()).throw(AssertionError('Native x4 allocation'))
        with torch.inference_mode():r.compute()
        self.assertEqual(r.gpu_output.shape,(60,72,3))

    def test_capture_is_bounded_to_small_realesr_inputs(self):
        r=Renderer.__new__(Renderer);r.use_graph=True
        r.model=SimpleNamespace(metadata=SimpleNamespace(capture_graph=False,capture_graph_max_pixels=640*360))
        self.assertTrue(r.can_capture(360,640))
        self.assertFalse(r.can_capture(720,1280))
        r.use_graph=False
        self.assertFalse(r.can_capture(360,640))

    def test_four_x_model_obeys_requested_two_x_and_four_x_output(self):
        for factor in (2,4):
            renderer=Renderer.__new__(Renderer)
            renderer.torch=torch;renderer.shape=(40,48,40*factor,48*factor)
            renderer.current_model='realesr-anime';renderer.backend='rocm'
            renderer.memory_format=torch.contiguous_format
            renderer.gpu_input=torch.full((40,48,4),128,dtype=torch.uint8)
            renderer.gpu_output=torch.zeros((40*factor,48*factor,3),dtype=torch.uint8)
            renderer.model=lambda x:torch.nn.functional.interpolate(x,scale_factor=4,mode='nearest')
            renderer.compute()
            self.assertTrue((renderer.gpu_output==128).all())

if __name__=='__main__':unittest.main()
