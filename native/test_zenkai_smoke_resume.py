import random
import unittest
import torch
from types import SimpleNamespace
from zenkai_experiments import state_snapshot
from zenkai_smoke_resume import restore_training_state,validate_resume


class SmokeResumeTests(unittest.TestCase):
    def test_next_stochastic_optimizer_update_matches_uninterrupted_run(self):
        torch.set_num_threads(1);torch.manual_seed(929)
        model=torch.nn.Sequential(torch.nn.Linear(3,3),torch.nn.Dropout(.3))
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        scaler=torch.amp.GradScaler('cuda',enabled=False)
        data=SimpleNamespace(rng=random.Random(929))
        x=torch.rand(4,3);model(x).square().mean().backward();optimizer.step();optimizer.zero_grad()
        snapshot=state_snapshot(model,optimizer,scaler,1,{'effectiveBatch':4},data.rng.getstate(),list(range(16)))
        expected_random=data.rng.random();x=torch.rand(4,3)
        model(x).square().mean().backward();optimizer.step()
        expected={k:v.clone() for k,v in model.state_dict().items()}
        resumed=torch.nn.Sequential(torch.nn.Linear(3,3),torch.nn.Dropout(.3))
        opt=torch.optim.AdamW(resumed.parameters(),lr=.8)
        dataset=SimpleNamespace(rng=random.Random(1))
        restore_training_state(resumed,opt,scaler,snapshot,dataset)
        self.assertEqual(dataset.rng.random(),expected_random)
        resumed(torch.rand(4,3)).square().mean().backward();opt.step()
        for k,v in resumed.state_dict().items():self.assertTrue(torch.equal(v,expected[k]),k)

    def test_only_reviewed_one_to_three_k_pair_can_resume(self):
        snapshot={'step':1000,'nextSampleOffset':8000,'samplerIndices':torch.arange(40000),
            'scheduler':{'totalSteps':5000,'completedStep':1000},'optimizer':{},'scaler':{},'rng':{},'datasetRng':(),
            'config':{'run':'zenkai-ablation-frequency-1k','family':'Zenkai','effectiveBatch':8,'scheduleTotalSteps':5000}}
        validate_resume(snapshot,'zenkai-ablation-frequency-3k-continuation',3000)
        with self.assertRaises(ValueError):validate_resume(snapshot,'unsafe',15000)
        snapshot['nextSampleOffset']=0
        with self.assertRaises(ValueError):validate_resume(snapshot,'zenkai-ablation-frequency-3k-continuation',3000)


if __name__=='__main__':unittest.main()
