import unittest
import numpy as np
from native_benchmark import compare, execute


class NativeBenchmarkTest(unittest.TestCase):
    def test_quality_rejects_invalid_and_wrong_output(self):
        reference = np.ones((1, 3, 4, 4), dtype=np.float32)
        self.assertTrue(compare(reference, reference)['passed'])
        self.assertFalse(compare(reference, reference * np.nan)['passed'])
        self.assertFalse(compare(reference, reference + .1)['passed'])

    def test_missing_provider_is_not_measured_as_cpu(self):
        result, output = execute('nonexistent.onnx', None, 'MissingExecutionProvider', 1, 1)
        self.assertEqual(result['status'], 'unavailable')
        self.assertIsNone(output)


if __name__ == '__main__':
    unittest.main()
