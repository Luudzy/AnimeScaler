import importlib.util
from pathlib import Path
import sys
import unittest


class DomainDecisionUnitsTests(unittest.TestCase):
    def test_psnr_delta_is_db_separate_from_relative_percent(self):
        folder = Path(__file__).resolve().parents[1] / 'diagnostics/zenkai-overnight'
        sys.path.insert(0, str(folder))
        try:
            spec = importlib.util.spec_from_file_location('domain_decision_units', folder / 'build_domain10_decision.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            sys.path.remove(str(folder))
        cases = {'clean': {
            'v2-domain10-1k': {'psnr': 31., 'mae': .02},
            'v2-dataset-1k': {'psnr': 30., 'mae': .025},
            'adore-lite-v3': {'psnr': 32., 'mae': .01},
        }}
        percent, db = module.quality_deltas(cases)
        self.assertNotIn('psnr', percent['clean'])
        self.assertEqual(db['clean'], {'vsUniform1k': 1., 'vsHistoricalV3': -1.})
        self.assertAlmostEqual(percent['clean']['mae']['vsUniform1k'], -20.)
        self.assertAlmostEqual(percent['clean']['mae']['vsHistoricalV3'], 100.)


if __name__ == '__main__':
    unittest.main()
