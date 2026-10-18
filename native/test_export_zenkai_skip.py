import tempfile
import unittest
from pathlib import Path
from unittest import mock
import torch
from zenkai_ablation_skip import IdentitySkipProbe
from export_zenkai import parity_metrics, require_parity, TOLERANCES, load_checkpoint
from export_zenkai_skip import load_candidate, load_deploy


class SkipExportTests(unittest.TestCase):
    def test_family_and_fused_reload_preserve_identity_forward(self):
        torch.manual_seed(5)
        model=IdentitySkipProbe().eval()
        # A nonzero residual is needed to expose a lost internal identity bypass.
        torch.nn.init.normal_(model.upsample.direct3.weight,std=.01)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/'master.pt'
            saved={'step':3000,'config':{'run':'zenkai-ablation-spab-identity-skip-3k-continuation',
                   'family':'ZenkaiIdentitySkipProbe'},'model':model.state_dict()}
            torch.save(saved,path)
            with mock.patch('export_zenkai_skip.load_probe',return_value=(model,saved)),mock.patch(
                    'export_zenkai_skip.ROOT',root):
                candidate,metadata=load_candidate(path)
                deploy=candidate.deploy_copy()
                torch.save({'model':deploy.state_dict(),'config':deploy.config,
                            'sourceSha256':'source'},root/'deploy-fp32.pt')
                report={'family':'ZenkaiIdentitySkipProbe','checkpointSha256':'source',
                        'artifacts':{'deploy-fp32.pt':{'sha256':'artifact'}}}
                with mock.patch('export_zenkai_skip.sha256',return_value='artifact'):
                    restored=load_deploy(root,report)
                x=torch.rand(1,3,17,19)
                with torch.inference_mode():
                    require_parity(parity_metrics(model(x),restored(x)),**TOLERANCES['fp32'])
                self.assertEqual(restored.config['family'],'ZenkaiIdentitySkipProbe')
                self.assertEqual(metadata['step'],3000)
            with self.assertRaises(ValueError):load_checkpoint(path)

    def test_unreviewed_run_and_artifact_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/'bad.pt'
            model=IdentitySkipProbe().eval()
            saved={'step':3000,'config':{'run':'unknown'},'model':model.state_dict()}
            torch.save(saved,path)
            with mock.patch('export_zenkai_skip.load_probe',return_value=(model,saved)),mock.patch(
                    'export_zenkai_skip.ROOT',root):
                with self.assertRaises(ValueError):load_candidate(path)
            report={'family':'ZenkaiIdentitySkipProbe','artifacts':{'deploy-fp32.pt':{'sha256':'pinned'}}}
            with mock.patch('export_zenkai_skip.sha256',return_value='changed'):
                with self.assertRaises(ValueError):load_deploy(root,report)


if __name__=='__main__':unittest.main()
