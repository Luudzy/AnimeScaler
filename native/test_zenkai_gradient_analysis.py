import unittest
import torch
from zenkai_gradient_analysis import simulated_update

class GradientAnalysisTests(unittest.TestCase):
    def test_probe_does_not_mutate_original_weights_and_reports_update_ratio(self):
        model=torch.nn.Conv2d(3,3,1)
        original={key:value.detach().clone() for key,value in model.state_dict().items()}
        x=torch.rand(1,3,4,4);target=torch.zeros_like(x)
        report=simulated_update(model,x,target,lambda a,b:(a-b).square().mean(),1e-4)
        self.assertIn('fresh-state',report['scope'])
        self.assertGreater(report['groups']['weight']['updateL2'],0)
        self.assertGreater(report['groups']['weight']['weightL2'],0)
        for key,value in original.items():self.assertTrue(torch.equal(value,model.state_dict()[key]))

if __name__=='__main__':unittest.main()
