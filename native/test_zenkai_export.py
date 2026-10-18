import tempfile
import unittest
from pathlib import Path
import torch
from zenkai_model import ZenkaiLite, GRepConv
from export_zenkai import load_checkpoint, parity_metrics, require_parity,configure_cache
import os
from unittest.mock import patch


class ZenkaiExportTests(unittest.TestCase):
    def test_standalone_runtime_cache_stays_in_workspace(self):
        with tempfile.TemporaryDirectory() as temporary,patch.dict(os.environ,{},clear=True):
            root=Path(temporary)
            configure_cache(root)
            for key in ('MIOPEN_USER_DB_PATH','MIOPEN_CUSTOM_CACHE_DIR'):
                self.assertTrue(Path(os.environ[key]).is_relative_to(root))
                self.assertTrue(Path(os.environ[key]).is_dir())
    def test_parity_fails_on_shape_nonfinite_and_excess_error(self):
        reference=torch.zeros(1,3,2,2)
        for wrong in [torch.zeros(1,3,1,2),reference+float('nan')]:
            with self.assertRaises(ValueError):parity_metrics(reference,wrong)
        scores=parity_metrics(reference,reference+.01)
        with self.assertRaises(ValueError):require_parity(scores,max_error=.004,mean_error=.0005)
        self.assertTrue(require_parity(parity_metrics(reference,reference),max_error=2e-5,mean_error=2e-6))

    def test_full_nonzero_model_fusion_matches_including_image_borders(self):
        torch.manual_seed(193);torch.set_num_threads(2)
        source=ZenkaiLite(channels=32,depth=5).eval()
        # The default zero residual head would conceal errors in inner blocks.
        torch.nn.init.normal_(source.upsample.direct3.weight,std=.02)
        before={name:value.clone() for name,value in source.state_dict().items()}
        deploy=source.deploy_copy().eval()
        self.assertFalse(any(isinstance(module,GRepConv) for module in deploy.modules()))
        with torch.inference_mode():
            for h,w in [(1,1),(17,19),(24,32)]:
                value=torch.rand(1,3,h,w)
                score=parity_metrics(source(value),deploy(value))
                require_parity(score,max_error=2e-5,mean_error=2e-6)
        self.assertTrue(all(torch.equal(before[name],value) for name,value in source.state_dict().items()))

    def test_load_rejects_nonmaster_or_wrong_family(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'checkpoint.pt'
            model=ZenkaiLite(channels=32,depth=5)
            config={'family':'Zenkai','channels':32,'depth':5,'nativeScale':2}
            torch.save({'model':model.state_dict(),'config':config,'step':20},path)
            loaded,metadata=load_checkpoint(path)
            self.assertEqual(metadata['step'],20)
            self.assertEqual(loaded.config['depth'],5)
            torch.save({'model':model.half().state_dict(),'config':config,'step':20},path)
            with self.assertRaises(ValueError):load_checkpoint(path)
            torch.save({'model':model.float().state_dict(),'config':{**config,'family':'Adore'},'step':20},path)
            with self.assertRaises(ValueError):load_checkpoint(path)


if __name__=='__main__':unittest.main()
