import unittest
import torch


class ZenkaiRealESRLiteTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(929)

    def test_shape_and_config(self):
        from zenkai_realesr_lite import ZenkaiRealESRLite
        model = ZenkaiRealESRLite(channels=32, blocks=6)
        self.assertEqual(model.config['channels'], 32)
        self.assertEqual(model.config['blocks'], 6)
        self.assertEqual(model.config['nativeScale'], 2)
        self.assertEqual(model.config['architecture'], 'SRVGG-inspired PReLU')
        self.assertGreater(sum(p.numel() for p in model.parameters()), 100_000)
        self.assertLess(sum(p.numel() for p in model.parameters()), 500_000)

    def test_zero_head_is_exact_bilinear_base(self):
        from zenkai_realesr_lite import ZenkaiRealESRLite
        model = ZenkaiRealESRLite()
        value = torch.rand(2, 3, 12, 16)
        with torch.inference_mode():
            actual = model(value)
            expected = torch.nn.functional.interpolate(
                value, scale_factor=2, mode='bilinear', align_corners=False)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_native_and_requested_sizes_are_finite(self):
        from zenkai_realesr_lite import ZenkaiRealESRLite
        model = ZenkaiRealESRLite().eval()
        value = torch.rand(2, 3, 12, 16)
        with torch.inference_mode():
            native = model(value)
            resized = model.forward_to(value, (30, 42))
        self.assertEqual(tuple(native.shape), (2, 3, 24, 32))
        self.assertEqual(tuple(resized.shape), (2, 3, 30, 42))
        self.assertTrue(torch.isfinite(native).all())
        self.assertTrue(torch.isfinite(resized).all())
        self.assertFalse(any(p.requires_grad is False for p in []))

    def test_invalid_configuration_and_small_shape_rejected(self):
        from zenkai_realesr_lite import ZenkaiRealESRLite
        with self.assertRaises(ValueError):
            ZenkaiRealESRLite(channels=30)
        with self.assertRaises(ValueError):
            ZenkaiRealESRLite(blocks=5)
        with self.assertRaises(ValueError):
            ZenkaiRealESRLite(scale=4)
        model = ZenkaiRealESRLite()
        with self.assertRaises(ValueError):
            model(torch.rand(1, 3, 3, 4))

    def test_multiscale_variant_is_bilinear_at_initialization(self):
        from zenkai_realesr_lite import ZenkaiAnimeSRLite
        model = ZenkaiAnimeSRLite()
        value = torch.rand(1, 3, 16, 20)
        expected = torch.nn.functional.interpolate(value, scale_factor=2, mode='bilinear', align_corners=False)
        torch.testing.assert_close(model(value), expected, rtol=0, atol=0)
        self.assertEqual(model.config['architecture'], 'AnimeSR-inspired multi-scale PReLU')


if __name__ == '__main__':
    unittest.main()
