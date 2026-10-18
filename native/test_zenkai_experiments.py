import unittest
import torch
from zenkai_experiments import build_experiment_model, state_snapshot, protocol_config
from lite_v3 import AdoreLiteV3
from zenkai_experiments import load_control_checkpoint, require_codec_cache,validate_experiment_identity
from pathlib import Path
import tempfile

class ExperimentTests(unittest.TestCase):
    def test_control_run_identity_cannot_be_consumed_by_another_architecture(self):
        validate_experiment_identity('control-v3','adore-v3-retrained-control-5k',5000)
        validate_experiment_identity('zenkai-a','zenkai-ablation-charbonnier-1k',1000)
        with self.assertRaises(ValueError):validate_experiment_identity('zenkai-a','adore-v3-retrained-control-5k',5000)
    def test_configured_codec_cache_cannot_silently_disappear(self):
        with tempfile.TemporaryDirectory() as folder:
            config={'codecFraction':.25,'codecCache':folder}
            with self.assertRaises(ValueError):require_codec_cache(config)
            (Path(folder)/'complete.json').write_text('{}')
            self.assertEqual(require_codec_cache(config),Path(folder))
    def test_control_is_exact_historical_architecture(self):
        torch.manual_seed(929); expected=AdoreLiteV3(24,4,.2)
        torch.manual_seed(929); actual=build_experiment_model('control-v3')
        self.assertEqual(set(expected.state_dict()),set(actual.state_dict()))
        for key,value in expected.state_dict().items():self.assertTrue(torch.equal(value,actual.state_dict()[key]))

    def test_snapshot_preserves_optimizer_rng_and_master_weights(self):
        model=build_experiment_model('control-v3')
        optimizer=torch.optim.AdamW(model.parameters())
        scaler=torch.amp.GradScaler('cuda',enabled=False)
        snapshot=state_snapshot(model,optimizer,scaler,3,{'run':'control'},'dataset-state',[])
        self.assertIn('optimizer',snapshot);self.assertIn('rng',snapshot)
        self.assertIn('datasetRng',snapshot);self.assertEqual(snapshot['step'],3)
        for value in snapshot['model'].values():self.assertEqual(value.dtype,torch.float32)

    def test_protocol_is_copied_and_only_declared_loss_change_is_allowed(self):
        source={'seed':929,'batch':4,'accumulation':2,'patchLR':128,'optimizer':'AdamW',
                'loss':'zenkai_frequency','channels':32,'depth':5,'run':'a'}
        control=protocol_config(source,'control-v3','control',5000,'zenkai_frequency')
        for key in ('seed','batch','accumulation','patchLR','optimizer','loss'):self.assertEqual(source[key],control[key])
        self.assertEqual(control['channels'],24);self.assertEqual(control['blocks'],4)
        with self.assertRaises(ValueError):protocol_config(source,'control-v3','unsafe',5000,'zenkai_baseline')

    def test_control_loader_checks_exact_architecture_and_finite_weights(self):
        model=build_experiment_model('control-v3')
        config={'family':'AdoreControl','channels':24,'blocks':4,'residualScale':.2,'nativeScale':2}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'control.pt'
            torch.save({'model':model.state_dict(),'config':config,'step':5000},path)
            loaded,metadata=load_control_checkpoint(path)
            self.assertEqual(metadata['step'],5000)
            self.assertEqual(set(loaded.state_dict()),set(model.state_dict()))
            config['blocks']=5
            torch.save({'model':model.state_dict(),'config':config,'step':5000},path)
            with self.assertRaises(ValueError):load_control_checkpoint(path)

if __name__=='__main__':unittest.main()
