import unittest
import torch
from zenkai_losses import ZenkaiLoss, block_dct, high_frequency


class FrequencyLossTests(unittest.TestCase):
    def test_constant_checkerboard_and_edge_responses(self):
        constant=torch.ones(1,3,24,24)*.5
        grid=torch.arange(24)
        checker=((grid[:,None]+grid[None,:])%2).float()[None,None].repeat(1,3,1,1)
        edge=torch.zeros_like(constant);edge[...,12:]=1
        flat=float(high_frequency(constant).abs().mean())
        sharp=float(high_frequency(edge).abs().mean())
        alternating=float(high_frequency(checker).abs().mean())
        self.assertLess(flat,1e-6)
        self.assertGreater(sharp,flat+.001)
        self.assertGreater(alternating,sharp)

    def test_orthonormal_dct_preserves_energy_and_dc(self):
        torch.manual_seed(17);image=torch.rand(2,3,16,24)
        self.assertTrue(torch.allclose(block_dct(image).square().sum(),image.square().sum(),rtol=1e-5))
        coeff=block_dct(torch.ones(1,1,8,8))
        self.assertAlmostEqual(float(coeff.flatten()[0]),8,places=5)
        self.assertLess(float(coeff.flatten()[1:].abs().max()),1e-5)

    def test_half_gradients_and_odd_shapes_are_finite(self):
        for preset in ['zenkai_baseline','zenkai_frequency','zenkai_frequency_edge']:
            pred=torch.rand(1,3,17,19,dtype=torch.float16,requires_grad=True)
            loss,terms=ZenkaiLoss(preset)(pred,torch.zeros_like(pred))
            self.assertEqual(loss.dtype,torch.float32)
            loss.backward()
            self.assertTrue(torch.isfinite(pred.grad).all())
            self.assertTrue(all(torch.isfinite(value) for value in terms.values()))

    def test_matching_images_have_zero_frequency_error(self):
        image=torch.rand(1,3,16,16)
        _,terms=ZenkaiLoss('zenkai_frequency')(image,image)
        self.assertEqual(float(terms['hf']),0)
        self.assertEqual(float(terms['dct']),0)


if __name__=='__main__':unittest.main()
