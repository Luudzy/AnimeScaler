import unittest
from benchmark_timing import timed_iterations

class TimingTests(unittest.TestCase):
    def test_completion_barrier_and_unique_inputs(self):
        events=[]
        result=timed_iterations(lambda:events.append('infer'),lambda:events.append('sync'),1,2,lambda i:events.append(i))
        self.assertEqual(events,[0,'infer','sync',1,'sync','infer','sync',2,'sync','infer','sync'])
        self.assertEqual(result['iterations'],2)
        self.assertGreaterEqual(result['meanMs'],result['cpuSubmitMeanMs'])
