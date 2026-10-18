import copy
import unittest
from baseline_reports import merge_baseline_report


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.performance = {'backend': 'torch', 'provider': 'ROCm', 'precision': 'fp16',
                            'cases': [{'model': 'a', 'sha256': 'abc', 'nativeScale': 2,
                                       'width': 20, 'height': 20, 'outputWidth': 40, 'outputHeight': 40}]}
        self.previous = {'performance': copy.deepcopy(self.performance),
                         'quality': {'backend': 'ROCm', 'precision': 'fp16', 'models': {'a': {'sha256': 'abc'}}},
                         'syntheticTemporal': {'sequences': 20}}

    def test_compatible_rerun_preserves_quality_and_temporal(self):
        result = merge_baseline_report(self.previous, self.performance)
        self.assertEqual(result['quality'], self.previous['quality'])
        self.assertEqual(result['syntheticTemporal'], self.previous['syntheticTemporal'])

    def test_changed_artifact_or_case_invalidates_quality(self):
        for field, value in [('sha256', 'changed'), ('outputWidth', 80)]:
            report = copy.deepcopy(self.performance)
            report['cases'][0][field] = value
            result = merge_baseline_report(self.previous, report)
            self.assertNotIn('syntheticTemporal', result)
            self.assertIn('qualityInvalidation', result)

    def test_quality_from_other_checkpoint_is_not_preserved(self):
        self.previous['quality']['models']['a']['sha256'] = 'other'
        self.assertIn('qualityInvalidation', merge_baseline_report(self.previous, self.performance))

    def test_onnx_export_change_invalidates_even_with_same_source(self):
        for report in (self.previous['performance'], self.performance):
            report['backend'] = 'onnx'
            report['cases'][0]['artifactSha256'] = 'old'
        self.performance['cases'][0]['artifactSha256'] = 'new'
        self.assertIn('qualityInvalidation', merge_baseline_report(self.previous, self.performance))
