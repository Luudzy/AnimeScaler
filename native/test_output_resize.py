import unittest
import torch
from output_resize import resize_output

class ResizeTests(unittest.TestCase):
    def test_identity_and_aa_fp16(self):
        value=torch.ones(1,3,32,48,dtype=torch.float16)*.4
        self.assertIs(resize_output(value,(32,48)),value)
        out=resize_output(value,(24,36),'bicubic-aa')
        self.assertEqual(tuple(out.shape),(1,3,24,36))
        self.assertTrue(torch.isfinite(out).all());self.assertLess(float((out-.4).abs().max()),.001)
    def test_invalid_filter(self):
        with self.assertRaises(ValueError):resize_output(torch.zeros(1,3,16,16),(24,24),'unknown')
if __name__=='__main__':unittest.main()
