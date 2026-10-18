import unittest
import torch
from zenkai_precision_probe import summarize_gradients


class PrecisionProbeTests(unittest.TestCase):
    def test_normalizes_scaled_gradients_and_preserves_zero_fraction(self):
        model=torch.nn.Linear(2,1,bias=False)
        model.weight.grad=torch.tensor([[0.,8.]])
        row=summarize_gradients(model,4.)['weight']
        self.assertEqual(row['gradientL2'],2.)
        self.assertEqual(row['zeroElementsPct'],50.)
        self.assertTrue(torch.equal(model.weight.grad,torch.tensor([[0.,8.]])))

    def test_nonfinite_gradients_are_rejected(self):
        model=torch.nn.Linear(1,1)
        model.weight.grad=torch.tensor([[float('nan')]])
        with self.assertRaises(ValueError):summarize_gradients(model,1.)


if __name__=='__main__':unittest.main()
