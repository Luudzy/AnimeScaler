import unittest
import numpy as np
import torch
from evaluate_sr_temporal import evaluate_synthetic_temporal, temporal_delta_mae


class TemporalTests(unittest.TestCase):
    def test_reference_change_is_not_counted_as_flicker(self):
        old = torch.zeros(1,3,4,4)
        new = torch.ones_like(old)*.25
        self.assertEqual(temporal_delta_mae(new,old,new,old),0)
        self.assertAlmostEqual(temporal_delta_mae(new+.1,old,new,old),.1,places=6)

    def test_model_registry_callable_and_no_input_mutation(self):
        image = np.full((224,224,3),128,dtype=np.uint8)
        class IdentityScale:
            def infer(self, value):
                return torch.nn.functional.interpolate(value,scale_factor=2,mode='nearest')
        model = IdentityScale()
        report = evaluate_synthetic_temporal({'model':model.infer},[image],device='cpu',dtype=torch.float32,frames=3)
        self.assertEqual(report['sequences'],1)
        self.assertEqual(report['realVideoValidation'],'NOT EXECUTED')
        self.assertEqual(report['temporalDeltaMae']['model'],0)
        self.assertTrue(np.all(image==128))

    def test_wrong_resolution_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'shape'):
            evaluate_synthetic_temporal({'bad':lambda x:x},[np.zeros((224,224,3),dtype=np.uint8)],device='cpu',frames=2)

    def test_empty_input_is_rejected(self):
        with self.assertRaises(ValueError):evaluate_synthetic_temporal({},[],device='cpu')


if __name__=='__main__':unittest.main()
