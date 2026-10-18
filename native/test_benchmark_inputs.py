import unittest
import torch
from benchmark_timing import fill_distinct_input

class BenchmarkInputsTests(unittest.TestCase):
    def test_entire_frames_are_distinct_reproducible_and_in_range(self):
        x = torch.empty(1, 3, 8, 9)
        fill_distinct_input(x, 0); first = x.clone()
        fill_distinct_input(x, 1); second = x.clone()
        self.assertGreater(int((first != second).sum()), first.numel() * .9)
        fill_distinct_input(x, 0)
        self.assertTrue(torch.equal(first, x))
        self.assertGreaterEqual(float(second.min()), 0)
        self.assertLess(float(second.max()), 1)

if __name__ == '__main__': unittest.main()
