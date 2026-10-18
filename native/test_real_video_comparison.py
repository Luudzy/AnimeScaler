import copy
import unittest
from compare_real_video import compare_reports,merge_completed_sequences


def fixture(candidate, value):
    return {'state': 'complete', 'backend': 'torch-cpu', 'precision': 'fp32', 'torch': 'same',
            'cutPolicy': {'threshold': .15}, 'artifacts': {'adore-lite-v3': {'sha256': 'v3'},
            candidate: {'sha256': candidate, 'step': 5000}}, 'sequences': [{
            'manifest': 'clip/complete.json', 'manifestSha256': 'clip', 'frameCount': 48,
            'suspectedCutTransitions': [], 'models': {
            'adore-lite-v3': {'mae': .1, 'referenceCompensatedTemporalMae': .02, 'includedTransitions': 47},
            candidate: {'mae': value, 'referenceCompensatedTemporalMae': .03, 'includedTransitions': 47}}}]}


class ComparisonTests(unittest.TestCase):
    def test_interrupted_report_can_merge_only_completed_distinct_sequences(self):
        first=fixture('zenkai-b',.08);first['state']='running'
        rest=fixture('zenkai-b',.08);rest['sequences'][0]['manifestSha256']='other'
        merged=merge_completed_sequences(first,rest)
        self.assertEqual(merged['state'],'complete');self.assertEqual(len(merged['sequences']),2)
        rest['sequences'][0]['manifestSha256']='clip'
        with self.assertRaises(ValueError):merge_completed_sequences(first,rest)
    def test_gains_and_regressions_are_both_retained(self):
        result = compare_reports(fixture('zenkai-a', .12), fixture('zenkai-b', .08))
        row = result['sequences'][0]
        self.assertAlmostEqual(row['bVersusV3']['mae']['changePercent'], -20)
        self.assertAlmostEqual(row['bVersusV3']['referenceCompensatedTemporalMae']['changePercent'], 50)
        self.assertFalse(result['releaseApproved'])

    def test_unmatched_protocol_or_artifacts_rejected(self):
        a, b = fixture('zenkai-a', .12), fixture('zenkai-b', .08)
        for field in ('precision', 'torch', 'backend', 'cutPolicy'):
            bad = copy.deepcopy(b); bad[field] = 'changed'
            with self.assertRaises(ValueError): compare_reports(a, bad)
        for field in ('manifestSha256', 'frameCount', 'suspectedCutTransitions'):
            bad = copy.deepcopy(b); bad['sequences'][0][field] = 'changed'
            with self.assertRaises(ValueError): compare_reports(a, bad)
        bad = copy.deepcopy(b); bad['artifacts']['adore-lite-v3']['sha256'] = 'changed'
        with self.assertRaises(ValueError): compare_reports(a, bad)

    def test_failed_or_nonfinite_results_rejected(self):
        a, b = fixture('zenkai-a', .12), fixture('zenkai-b', .08)
        b['state'] = 'failed'
        with self.assertRaises(ValueError): compare_reports(a, b)
        b['state'] = 'complete'; b['sequences'][0]['models']['zenkai-b']['mae'] = float('nan')
        with self.assertRaises(ValueError): compare_reports(a, b)


if __name__ == '__main__': unittest.main()
