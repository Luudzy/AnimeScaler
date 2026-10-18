import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


class HistoricalReportPinTests(unittest.TestCase):
    def test_metrics_edit_with_same_model_identity_is_rejected(self):
        folder=Path(__file__).resolve().parents[1]/'diagnostics/zenkai-overnight'
        sys.path.insert(0,str(folder))
        try:
            spec=importlib.util.spec_from_file_location('domain10_report_test',folder/'build_domain10_report.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        finally:
            sys.path.remove(str(folder))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'comparison.json'
            original={'state':'COMPLETE','modelSha256':'same-model','mae':.01}
            path.write_text(json.dumps(original),encoding='utf-8')
            meta={'evaluationReportSha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            self.assertEqual(module.load_pinned_comparison(path,meta),original)
            path.write_text(json.dumps({**original,'mae':.001}),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'hash'):
                module.load_pinned_comparison(path,meta)


if __name__=='__main__':unittest.main()
