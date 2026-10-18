import tempfile
from pathlib import Path
import unittest
import torch
from zenkai_ablation_residual import ResidualScaleProbe
from zenkai_ablation_skip import IdentitySkipProbe
from zenkai_probe_evaluation import load_probe,load_reference_checkpoint
from zenkai_model import ZenkaiLite


class ProbeEvaluationTests(unittest.TestCase):
    def test_skip_legacy_loader_requires_exact_loss_identity_and_budget(self):
        model=IdentitySkipProbe()
        config={'family':'ZenkaiIdentitySkipProbe','channels':32,'depth':5,'nativeScale':2,
            'spabIdentitySkipGain':1.,'run':'zenkai-ablation-spab-identity-skip-legacy-loss-1k',
            'loss':'zenkai_legacy_v3','comparisonRun':'zenkai-ablation-spab-identity-skip-1k'}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'probe.pt'
            torch.save({'config':config,'step':1000,'model':model.state_dict()},path)
            self.assertEqual(load_probe(path)[1]['step'],1000)
            config['loss']='zenkai_frequency'
            torch.save({'config':config,'step':1000,'model':model.state_dict()},path)
            with self.assertRaises(ValueError):load_probe(path)

    def test_three_k_skip_requires_explicit_stage_and_exact_continuation_identity(self):
        model=IdentitySkipProbe()
        config={'family':'ZenkaiIdentitySkipProbe','channels':32,'depth':5,'nativeScale':2,
            'spabIdentitySkipGain':1.,'run':'zenkai-ablation-spab-identity-skip-3k-continuation'}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'probe.pt'
            torch.save({'config':config,'step':3000,'model':model.state_dict()},path)
            with self.assertRaises(ValueError):load_probe(path)
            self.assertEqual(load_probe(path,expected_step=3000)[1]['step'],3000)

    def test_reference_requires_the_matched_run_and_exact_smoke_step(self):
        config={'family':'Zenkai','channels':32,'depth':5,'nativeScale':2,'run':'zenkai-ablation-frequency-1k'}
        model=ZenkaiLite(32,5)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'reference.pt'
            torch.save({'model':model.state_dict(),'config':config,'step':5000},path)
            with self.assertRaises(ValueError):load_reference_checkpoint(path)
            torch.save({'model':model.state_dict(),'config':config,'step':1000},path)
            self.assertEqual(load_reference_checkpoint(path)[1]['step'],1000)

    def test_loader_retains_probe_gain_and_rejects_unreviewed_config(self):
        torch.set_num_threads(1);model=ResidualScaleProbe().eval()
        with torch.no_grad():model.upsample.direct3.bias.fill_(.2)
        config={'family':'ZenkaiResidualScaleProbe','channels':32,'depth':5,'nativeScale':2,
            'outputResidualScale':.2,'run':'zenkai-ablation-output-residual-02-1k'}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'probe.pt'
            torch.save({'config':config,'step':1000,'model':model.state_dict()},path)
            loaded,_=load_probe(path);x=torch.rand(1,3,8,8)
            with torch.no_grad():self.assertTrue(torch.allclose(model(x),loaded(x),atol=1e-7))
            config['outputResidualScale']=1.
            torch.save({'config':config,'step':1000,'model':model.state_dict()},path)
            with self.assertRaises(ValueError):load_probe(path)


if __name__=='__main__':unittest.main()
