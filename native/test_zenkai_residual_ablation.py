import unittest
import torch
from zenkai_model import ZenkaiLite
from zenkai_ablation_residual import ResidualScaleProbe


class ResidualAblationTests(unittest.TestCase):
    def test_only_output_residual_gain_changes(self):
        torch.set_num_threads(1)
        torch.manual_seed(929);baseline=ZenkaiLite(32,5).eval()
        torch.manual_seed(929);probe=ResidualScaleProbe().eval()
        for key,value in baseline.state_dict().items():self.assertTrue(torch.equal(value,probe.state_dict()[key]))
        with torch.no_grad():baseline.upsample.direct3.bias.fill_(.2)
        probe.load_state_dict(baseline.state_dict())
        x=torch.rand(1,3,8,8);base=torch.nn.functional.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False)
        with torch.no_grad():expected=base+.2*(baseline(x)-base);actual=probe(x);deploy=probe.deploy_copy()(x)
        self.assertTrue(torch.allclose(actual,expected,atol=1e-7))
        self.assertTrue(torch.allclose(actual,deploy,atol=1e-6))


if __name__=='__main__':unittest.main()
