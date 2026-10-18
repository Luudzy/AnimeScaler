import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import torch
from zenkai_ablation_skip import IdentitySkipProbe
from zenkai_probe_evaluation import load_probe


class Codec3kEvaluationTests(unittest.TestCase):
    def test_new_scheduler_identity_does_not_relax_other_factor_runs(self):
        reference={'run':'zenkai-ablation-spab-identity-skip-1k','family':'ZenkaiIdentitySkipProbe',
                'channels':32,'depth':5,'nativeScale':2,'spabIdentitySkipGain':1.,'loss':'zenkai_frequency',
                'plannedSteps':1000,'patchLR':128,'seed':929,'codecFraction':.25,'learningRate':[.0002,.00001]}
        config={**reference,'run':'zenkai-ablation-skip-factor-scheduler-constant-v2-1k',
                'factorName':'scheduler-constant','learningRate':[.0002,.0002],
                'schedulerExperiment':'same warmup300 then constant2e-4','comparisonRun':reference['run']}
        with tempfile.TemporaryDirectory() as folder,mock.patch(
                'zenkai_probe_evaluation.load_factor_reference_config',return_value=reference):
            path=Path(folder)/'master.pt';model=IdentitySkipProbe()
            torch.save({'model':model.state_dict(),'config':config,'step':1000},path)
            self.assertEqual(load_probe(path,1000)[1]['step'],1000)
            config['codecFraction']=.5
            torch.save({'model':model.state_dict(),'config':config,'step':1000},path)
            with self.assertRaises(ValueError):load_probe(path,1000)

    def test_exact_codec50_continuation_only(self):
        source={'run':'zenkai-ablation-skip-factor-codec50-1k','family':'ZenkaiIdentitySkipProbe',
                'channels':32,'depth':5,'nativeScale':2,'spabIdentitySkipGain':1.,'factorName':'codec50',
                'codecFraction':.5,'loss':'zenkai_frequency','plannedSteps':1000,'patchLR':128,'seed':929}
        cfg={**source,'run':'zenkai-ablation-skip-factor-codec50-3k-continuation','sourceRun':source['run'],
             'plannedSteps':3000,'continuedFromSha256':'pinned-source-state'}
        with tempfile.TemporaryDirectory() as folder,mock.patch(
                'zenkai_probe_evaluation.load_codec_resume_source',return_value=(source,'pinned-source-state'),create=True):
            path=Path(folder)/'master.pt';model=IdentitySkipProbe()
            torch.save({'model':model.state_dict(),'config':cfg,'step':3000},path)
            self.assertEqual(load_probe(path,3000)[1]['step'],3000)
            for key,value in (('codecFraction',.25),('seed',1),('plannedSteps',15000),('continuedFromSha256','wrong')):
                bad=copy.deepcopy(cfg);bad[key]=value
                torch.save({'model':model.state_dict(),'config':bad,'step':3000},path)
                with self.assertRaises(ValueError,msg=key):load_probe(path,3000)


if __name__=='__main__':unittest.main()
