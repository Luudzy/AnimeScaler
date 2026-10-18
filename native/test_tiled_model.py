import unittest
import torch
from torch import nn
from tiled_model import TiledModel

class Upscale(nn.Module):
    def forward(self,x):return torch.nn.functional.interpolate(x,scale_factor=2,mode='nearest')

class TiledTests(unittest.TestCase):
    def test_overlap_has_no_gaps_or_brightness_change(self):
        x=torch.rand(1,3,76,108)
        result=TiledModel(Upscale(),2,max_pixels=1,tile=40,overlap=8,context=4)(x)
        torch.testing.assert_close(result,Upscale()(x),atol=1e-6,rtol=1e-6)
    def test_small_images_use_full_inference(self):
        x=torch.rand(1,3,40,48)
        self.assertTrue(torch.equal(TiledModel(Upscale(),2)(x),Upscale()(x)))
