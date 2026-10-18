import unittest
import torch
from zenkai_model import ZenkaiLite
from zenkai_ablation_skip import IdentitySkipProbe


class SkipProbeTests(unittest.TestCase):
    def test_original_weights_are_preserved_and_skip_is_explicit(self):
        torch.set_num_threads(1)
        torch.manual_seed(929);baseline=ZenkaiLite(32,5).eval()
        torch.manual_seed(929);probe=IdentitySkipProbe().eval()
        for key,value in probe.state_dict().items():
            self.assertTrue(torch.equal(value,baseline.state_dict()[key.replace('.core.','.')]))
        x=torch.rand(1,32,8,8)
        with torch.no_grad():
            self.assertTrue(torch.allclose(probe.blocks[1](x),baseline.blocks[1](x)+x))
            probe.upsample.direct3.bias.fill_(.1)
            value=torch.rand(1,3,8,8)
            self.assertTrue(torch.allclose(probe(value),probe.deploy_copy()(value),atol=1e-6))


if __name__=='__main__':unittest.main()
