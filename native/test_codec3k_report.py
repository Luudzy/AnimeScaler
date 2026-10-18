import importlib.util
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('codec_report',Path(__file__).resolve().parents[1]/'diagnostics/zenkai-overnight/build_codec3k_report.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class CompleteAttemptTests(unittest.TestCase):
    def test_partial_failed_attempt_cannot_supply_metrics(self):
        self.assertIsNone(module.selected({'state':'NOT_MEASURED','result':{'path':'partial.json'}}))
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);path=out/'report.json';path.write_text('{}')
            op={'state':'PASS','selectedAttempt':2,'attempts':[
                {'attempt':1,'state':'HIP_INFRA_FAILURE','exitCode':1,'output':str(out)},
                {'attempt':2,'state':'PASS','exitCode':0,'output':str(out)}],
                'result':{'path':str(path),'sha256':module.sha(path)}}
            self.assertEqual(module.selected(op),op['result'])
            op['selectedAttempt']=1
            with self.assertRaises(ValueError):module.selected(op)
            op['selectedAttempt']=2;path.write_text('{"partial":true}')
            with self.assertRaises(ValueError):module.selected(op)


if __name__=='__main__':unittest.main()
