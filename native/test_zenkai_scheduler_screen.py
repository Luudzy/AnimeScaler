import copy
import random
import unittest
import torch
from zenkai_scheduler_screen import validate_snapshot, schedule_rate, RUN
from train_zenkai import learning_rate


class SchedulerScreenTests(unittest.TestCase):
    def fixture(self):
        return {'step':0,'config':{'run':'zenkai-ablation-spab-identity-skip-1k',
                'family':'ZenkaiIdentitySkipProbe','channels':32,'depth':5,'nativeScale':2,
                'spabIdentitySkipGain':1.,'loss':'zenkai_frequency','codecFraction':.25,
                'patchLR':128,'seed':929,'batch':4,'accumulation':2,'effectiveBatch':8,'scheduleTotalSteps':5000},
                'model':{'weight':torch.ones(1)},'optimizer':{'state':{},'param_groups':[{}]},
                'scaler':{'scale':65536.,'growth_factor':2.,'backoff_factor':.5,'growth_interval':2000,'_growth_tracker':0},
                'rng':{'python':random.Random(929).getstate(),'torch':torch.get_rng_state(),
                       'cuda':[torch.ones(20,dtype=torch.uint8)],'numpy':{'kind':'MT19937'}},
                'datasetRng':random.Random(929).getstate(),'samplerIndices':torch.arange(40000),
                'nextSampleOffset':0,'scheduler':{'kind':'explicit cosine with 300-step warmup','totalSteps':5000,'completedStep':0}}

    def test_new_run_uses_only_real_initial_or_complete_indexed_recovery(self):
        state=self.fixture();validate_snapshot(state)
        for key in ('optimizer','scaler','rng','model','datasetRng','samplerIndices','scheduler'):
            bad=copy.deepcopy(state);bad.pop(key)
            with self.assertRaises(ValueError,msg=key):validate_snapshot(bad)
        for change in ({'step':200},{'config':{**state['config'],'codecFraction':.5}},
                       {'config':{**state['config'],'run':'zenkai-ablation-skip-factor-scheduler-constant-1k'}}):
            bad=copy.deepcopy(state);bad.update(change)
            with self.assertRaises(ValueError):validate_snapshot(bad)

    def test_constant_schedule_changes_only_after_shared_warmup(self):
        self.assertTrue(RUN.endswith('-v2-1k'))
        for step in range(1,301):self.assertEqual(schedule_rate(step),learning_rate(step,5000))
        for step in (301,500,1000):self.assertEqual(schedule_rate(step),2e-4)


if __name__=='__main__':unittest.main()
