import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zenkai_artifacts import register_zenkai_export,register_probe_checkpoint
from model_registry import ModelRegistry


class ZenkaiArtifactTests(unittest.TestCase):
    def test_probe_registry_retains_wrapper_and_pins_artifact(self):
        import torch
        from zenkai_ablation_skip import IdentitySkipProbe
        torch.set_num_threads(1);model=IdentitySkipProbe().eval()
        config={'family':'ZenkaiIdentitySkipProbe','channels':32,'depth':5,'nativeScale':2,
            'spabIdentitySkipGain':1.,'run':'zenkai-ablation-spab-identity-skip-3k-continuation'}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'probe.pt';torch.save({'model':model.state_dict(),'config':config,'step':3000},path)
            registry=ModelRegistry();meta=register_probe_checkpoint(registry,path,expected_step=3000)
            self.assertEqual(meta.status,'experimental')
            loaded=registry.load(meta.id,device='cpu',precision='fp32')
            x=torch.rand(1,3,8,8)
            with torch.no_grad():self.assertTrue(torch.allclose(loaded.infer(x),model.deploy_copy()(x),atol=1e-6))
            registry.unload(meta.id);path.write_bytes(b'changed')
            with self.assertRaises(ValueError):registry.load(meta.id,device='cpu',precision='fp32')

    def test_failed_or_incomplete_export_never_registers(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            for report in [{'state':'failed'}, {'state':'complete','parityApproved':False}]:
                (folder/'export-report.json').write_text(json.dumps(report))
                registry=ModelRegistry()
                with self.assertRaises(ValueError):register_zenkai_export(registry,folder,'zenkai-test')
                self.assertEqual(registry.list_models(),[])

    def test_valid_export_is_opt_in_and_detects_artifact_change(self):
        root=Path(__file__).resolve().parents[1]
        folder=root/'models/zenkai/exports/smoke-04-cpu'
        if not folder.is_dir():self.skipTest('Run the explicit smoke export before artifact integration tests')
        registry=ModelRegistry()
        register_zenkai_export(registry,folder,'zenkai-smoke')
        self.assertEqual(registry.get('zenkai-smoke').status,'experimental')
        self.assertEqual(registry.get('zenkai-smoke').family,'Zenkai')
        with patch('model_registry.hashlib.sha256') as digest:
            digest.return_value.hexdigest.return_value='changed'
            with self.assertRaises(ValueError):registry.load('zenkai-smoke','torch')
        self.assertEqual(registry.loaded,{})
        import torch
        model=registry.load('zenkai-smoke','torch',device='cpu',precision='fp32')
        self.assertEqual(tuple(model.infer(torch.zeros(1,3,7,9)).shape),(1,3,14,18))
        registry.unload('zenkai-smoke')


if __name__=='__main__':unittest.main()
