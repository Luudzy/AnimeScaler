import copy
import random
import unittest
import torch
from zenkai_experiments import state_snapshot


class CodecResumeTests(unittest.TestCase):
    def snapshot(self):
        model=torch.nn.Linear(3,3)
        optimizer=torch.optim.AdamW(model.parameters())
        model(torch.ones(2,3)).sum().backward();optimizer.step()
        for state in optimizer.state.values():state['step'].fill_(1000)
        config={'run':'zenkai-ablation-skip-factor-codec50-1k','family':'ZenkaiIdentitySkipProbe',
            'factorName':'codec50','codecFraction':.5,'loss':'zenkai_frequency','channels':32,'depth':5,
            'nativeScale':2,'spabIdentitySkipGain':1.,'scheduleTotalSteps':5000,'effectiveBatch':8,
            'batch':4,'accumulation':2,'patchLR':128,'seed':929,'plannedSteps':1000}
        scaler=type('Scaler',(),{'state_dict':lambda self:{'scale':65536.,'growth_factor':2.,'backoff_factor':.5,
                'growth_interval':2000,'_growth_tracker':1000}})()
        saved=state_snapshot(model,optimizer,scaler,1000,config,random.Random(929).getstate(),range(40000))
        saved['rng']['cuda']=[torch.zeros(16,dtype=torch.uint8)]
        return saved

    def test_only_complete_codec50_state_can_continue_to_three_k(self):
        from zenkai_codec_resume import validate_codec_resume,RUN
        state=self.snapshot();validate_codec_resume(state,RUN,3000)
        for field in ('optimizer','scaler','datasetRng','scheduler','samplerIndices','model'):
            bad=copy.deepcopy(state);bad.pop(field)
            with self.assertRaises(ValueError,msg=field):validate_codec_resume(bad,RUN,3000)
        for field in ('cuda','python','torch','numpy'):
            bad=copy.deepcopy(state);bad['rng'].pop(field)
            with self.assertRaises(ValueError,msg=field):validate_codec_resume(bad,RUN,3000)

    def test_wrong_budget_sampler_or_factor_rejected(self):
        from zenkai_codec_resume import validate_codec_resume,RUN
        state=self.snapshot()
        for key,value in (('codecFraction',.25),('patchLR',96),('loss','zenkai_legacy_v3'),('plannedSteps',15000)):
            bad=copy.deepcopy(state);bad['config'][key]=value
            with self.assertRaises(ValueError,msg=key):validate_codec_resume(bad,RUN,3000)
        bad=copy.deepcopy(state);bad['nextSampleOffset']=0
        with self.assertRaises(ValueError):validate_codec_resume(bad,RUN,3000)
        with self.assertRaises(ValueError):validate_codec_resume(state,RUN,15000)

    def test_optimizer_steps_and_finite_master_are_required(self):
        from zenkai_codec_resume import validate_codec_resume,RUN
        state=self.snapshot()
        bad=copy.deepcopy(state);next(iter(bad['optimizer']['state'].values()))['step'].fill_(999)
        with self.assertRaises(ValueError):validate_codec_resume(bad,RUN,3000)
        bad=copy.deepcopy(state);next(iter(bad['model'].values())).fill_(float('nan'))
        with self.assertRaises(ValueError):validate_codec_resume(bad,RUN,3000)


if __name__=='__main__':unittest.main()
