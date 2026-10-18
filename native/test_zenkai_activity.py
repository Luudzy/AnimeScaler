import unittest
import torch
from zenkai_activity import correction_stats, weight_change_summary, gradient_flow_probe, load_initial_reference


class ZenkaiActivityTests(unittest.TestCase):
    def test_same_resolution_correction_separates_model_from_resize(self):
        baseline=torch.full((1,3,16,16),.25)
        zero=correction_stats(baseline,baseline)
        self.assertEqual(zero['visibleCorrectionMeanAbs'],0)
        self.assertEqual(zero['quantizedPixelChangePct'],0)
        changed=correction_stats(baseline+.01,baseline)
        self.assertAlmostEqual(changed['visibleCorrectionMeanAbs'],.01,places=6)
        self.assertEqual(changed['quantizedPixelChangePct'],100)
        white=torch.ones_like(baseline)
        self.assertEqual(correction_stats(white+.1,white)['quantizedPixelChangePct'],0)

    def test_invalid_or_unaligned_input_cannot_claim_correction(self):
        baseline=torch.zeros(1,3,4,4)
        for value in [baseline+float('nan'),torch.zeros(1,3,2,2)]:
            with self.assertRaises(ValueError):correction_stats(value,baseline)

    def test_weight_report_detects_unchanged_body_and_nonzero_head(self):
        initial={'stem.weight':torch.ones(2,2),'blocks.0.weight':torch.ones(2,2),'upsample.weight':torch.zeros(2,2)}
        trained={name:value.clone() for name,value in initial.items()}
        trained['upsample.weight']+=.1
        report=weight_change_summary(trained,initial)
        self.assertEqual(report['stem']['deltaL2'],0)
        self.assertEqual(report['blocks.0']['changedElementsPct'],0)
        self.assertGreater(report['upsample']['deltaL2'],0)
        self.assertIsNone(report['upsample']['deltaToInitialRatio'])

    def test_gradient_probe_reports_unused_parameters_without_updating_weights(self):
        class ProbeModel(torch.nn.Module):
            def __init__(self):
                super().__init__();self.stem=torch.nn.Conv2d(3,3,1)
                self.unused=torch.nn.Parameter(torch.ones(2))
            def forward(self,value):return self.stem(value)
        model=ProbeModel();before={name:value.clone() for name,value in model.state_dict().items()}
        result=gradient_flow_probe(model,torch.ones(1,3,4,4),torch.zeros(1,3,4,4),lambda a,b:(a-b).square().mean())
        self.assertGreater(result['groups']['stem']['gradientL2'],0)
        self.assertEqual(result['groups']['unused']['gradientL2'],0)
        self.assertEqual(result['groups']['unused']['missingGradientElementsPct'],100)
        self.assertTrue(all(torch.equal(value,before[name]) for name,value in model.state_dict().items()))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_attention_probe_measures_each_block_and_removes_hooks(self):
        from zenkai_model import ZenkaiLite
        model=ZenkaiLite(channels=4,depth=5)
        result=gradient_flow_probe(model,torch.ones(1,3,8,8),torch.zeros(1,3,16,16),lambda a,b:(a-b).square().mean())
        self.assertEqual(set(result['attention']),{'blocks.0','blocks.1','blocks.2','blocks.3','tail'})
        for row in result['attention'].values():
            self.assertGreaterEqual(row['inputRms'],0)
            self.assertLessEqual(row['maxAbsGate'],.5)
        self.assertTrue(all(not module._forward_hooks for module in model.modules()))

    def test_initial_reference_cannot_mix_torch_versions_or_training_seeds(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'initial.pt'
            config={'seed':929,'channels':32,'depth':5,'nativeScale':2,'torch':str(torch.__version__)}
            import hashlib
            code_hash=hashlib.sha256(Path(__file__).with_name('zenkai_model.py').read_bytes()).hexdigest()
            reference={'config':config,'torch':str(torch.__version__),'modelCodeSha256':code_hash,
                       'model':{'stem.weight':torch.ones(2,2)}}
            torch.save(reference,path)
            self.assertEqual(load_initial_reference(path,config)['stem.weight'].sum(),4)
            for changed in [{**config,'seed':1},{**config,'torch':'different-version'}]:
                with self.assertRaises(ValueError):load_initial_reference(path,changed)


if __name__=='__main__':unittest.main()
