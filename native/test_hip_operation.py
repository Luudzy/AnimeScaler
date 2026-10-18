import json
import sys
import tempfile
import unittest
from pathlib import Path


class CleanHipOperationTests(unittest.TestCase):
    def run_case(self,failures,start=1,kind='hip'):
        from hip_operation import run_operation
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder)
            code=("import sys,json,os;from pathlib import Path;"
                  "n=int(sys.argv[1]);p=Path(sys.argv[2]);"
                  f"bad=n<={failures};"
                  "p.write_text(json.dumps({'complete':not bad,'attempt':n,'pid':os.getpid()}));"
                  "sys.stderr.write('HIP error: unspecified launch failure' if bad and sys.argv[3]=='hip' else ('unrelated error' if bad else ''));"
                  "sys.exit(1 if bad else 0)")
            def command(n,out):return [sys.executable,'-c',code,str(n),str(out/'result.json'),kind]
            def validate(out):
                result=json.loads((out/'result.json').read_text())
                if not result['complete']:raise ValueError('Partial result')
                return result
            result=run_operation(base,command,validate,start_attempt=start)
            return result

    def test_clean_retry_selects_only_complete_attempt(self):
        result=self.run_case(1)
        self.assertEqual(result['state'],'PASS')
        self.assertEqual(result['selectedAttempt'],2)
        self.assertEqual(result['result']['attempt'],2)
        self.assertEqual([a['state'] for a in result['attempts']],['HIP_INFRA_FAILURE','PASS'])
        self.assertEqual(len(set(a['pid'] for a in result['attempts'])),2)

    def test_exhausted_hip_is_not_model_failure(self):
        result=self.run_case(3)
        self.assertEqual(result['state'],'NOT_MEASURED')
        self.assertIsNone(result['selectedAttempt'])
        self.assertEqual(len(result['attempts']),3)
        self.assertEqual(result['reason'],'recurrent HIP infrastructure failure')

    def test_original_failure_leaves_only_two_new_attempts(self):
        result=self.run_case(3,start=2)
        self.assertEqual([a['attempt'] for a in result['attempts']],[2,3])

    def test_non_hip_failure_does_not_retry(self):
        result=self.run_case(3,kind='other')
        self.assertEqual(result['state'],'NOT_MEASURED')
        self.assertEqual(result['attempts'][0]['state'],'INCONCLUSIVE')
        self.assertEqual(len(result['attempts']),1)
        self.assertEqual(result['scope'],'operation only; no model invalidation')

    def test_integrity_preflight_failure_leaves_terminal_record_without_gpu_spawn(self):
        from hip_operation import run_operation
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder)
            def command(attempt,out):raise ValueError('Checkpoint SHA mismatch')
            result=run_operation(base,command,lambda out:None)
            self.assertEqual(result['state'],'FAIL')
            self.assertEqual(result['attempts'][0]['state'],'FAIL')
            self.assertIsNone(result['attempts'][0]['pid'])
            self.assertEqual(json.loads((base/'operation.json').read_text()),result)


if __name__=='__main__':unittest.main()
