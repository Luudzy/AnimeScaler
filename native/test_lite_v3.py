import unittest,random
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from lite_v3 import AdoreLiteV3
from lite_losses import loss_terms,ssim
from lite_degradations import degrade
from prepare_lite_v3 import fingerprints

class LiteV3Tests(unittest.TestCase):
    def test_initial_residual_is_bilinear_and_shapes_dynamic(self):
        model=AdoreLiteV3()
        self.assertEqual(sum(p.numel() for p in model.parameters()),50148)
        for h,w in [(16,20),(24,32)]:
            x=torch.rand(1,3,h,w)
            self.assertTrue(torch.equal(model(x),F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False)))
    def test_loss_finite_and_gradients_reach_model(self):
        x=torch.rand(1,3,24,24,requires_grad=True);terms=loss_terms(x,torch.zeros_like(x))
        sum(terms.values()).backward();self.assertTrue(torch.isfinite(x.grad).all())
        self.assertAlmostEqual(float(ssim(x.detach(),x.detach())),1.,places=5)
    def test_degradation_reproducible(self):
        image=Image.fromarray(np.random.default_rng(1).integers(0,256,(192,192,3),dtype=np.uint8))
        a=degrade(image,random.Random(2));b=degrade(image,random.Random(2))
        self.assertEqual(a.size,(96,96));self.assertEqual(a.tobytes(),b.tobytes())
        self.assertEqual(fingerprints(image),fingerprints(image.copy()))
if __name__=='__main__':unittest.main()
