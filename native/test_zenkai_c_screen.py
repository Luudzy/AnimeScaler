import copy
import unittest
import tempfile
from pathlib import Path
from unittest import mock
import json
import torch
from zenkai_ablation_skip import IdentitySkipProbe
from zenkai_probe_evaluation import load_probe
import test_zenkai_codec_resume as codec_test
from zenkai_c_screen import validate_c_resume,RUN,SOURCE


class CResumeTests(unittest.TestCase):
    def test_five_k_candidate_load_is_bound_to_real_three_k_source_config(self):
        model=IdentitySkipProbe().eval();cfg={'run':SOURCE,'plannedSteps':3000,'family':'ZenkaiIdentitySkipProbe',
            'channels':32,'depth':5,'nativeScale':2,'codecFraction':.5,'seed':929,'spabIdentitySkipGain':1.}
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'models/zenkai/experiments'/SOURCE;source.mkdir(parents=True)
            (source/'config.json').write_text(json.dumps(cfg));p=source/'step-3000-training-state.pt';p.write_bytes(b'actual source bytes')
            from export_zenkai import sha256
            digest=sha256(p);(source/'metadata.json').write_text(json.dumps({'checkpointsSha256':{p.name:digest}}))
            candidate=root/'c.pt';ccfg={**cfg,'run':RUN,'plannedSteps':5000,'cSourceRun':SOURCE,'cSourceSha256':digest}
            with mock.patch('zenkai_c_screen.ROOT',root):
                torch.save({'model':model.state_dict(),'config':ccfg,'step':5000},candidate)
                self.assertEqual(load_probe(candidate,5000)[1]['step'],5000)
                ccfg['seed']=1;torch.save({'model':model.state_dict(),'config':ccfg,'step':5000},candidate)
                with self.assertRaises(ValueError):load_probe(candidate,5000)

    def test_c_is_only_faithful_codec50_three_to_five_k(self):
        original=codec_test.CodecResumeTests().snapshot()
        original['step']=3000;original['nextSampleOffset']=24000
        original['scheduler']['completedStep']=3000
        original['config'].update(run=SOURCE,sourceRun='zenkai-ablation-skip-factor-codec50-1k',plannedSteps=3000)
        for v in original['optimizer']['state'].values():v['step'].fill_(3000)
        validate_c_resume(original,5000)
        for key in ('optimizer','scaler','rng','datasetRng','scheduler','samplerIndices','model'):
            bad=copy.deepcopy(original);bad.pop(key)
            with self.assertRaises(ValueError,msg=key):validate_c_resume(bad,5000)
        for target in (3000,15000,30000):
            with self.assertRaises(ValueError):validate_c_resume(original,target)
        bad=copy.deepcopy(original);bad['config']['codecFraction']=.25
        with self.assertRaises(ValueError):validate_c_resume(bad,5000)
        bad=copy.deepcopy(original);bad['config'].update(run=RUN,plannedSteps=5000,cSourceRun=SOURCE)
        bad['step']=3100;bad['nextSampleOffset']=24800;bad['scheduler']['completedStep']=3100
        for v in bad['optimizer']['state'].values():v['step'].fill_(3100)
        validate_c_resume(bad,5000)
        bad['rng']['cuda']=[]
        with self.assertRaises(ValueError):validate_c_resume(bad,5000)


if __name__=='__main__':unittest.main()
