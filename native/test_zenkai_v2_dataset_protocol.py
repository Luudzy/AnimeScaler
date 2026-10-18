import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
import torch
from zenkai_experiments import state_snapshot


class V2DatasetProtocolTests(unittest.TestCase):
    def test_foreign_absolute_checkpoint_path_maps_only_matching_run_attempt_and_step(self):
        from zenkai_v2_dataset_protocol import checkpoint_index_key
        foreign = r'C:\Users\other\AnimeScaler\models\zenkai\experiments\zenkai-ablation-skip-v2-dataset-1k\attempts\attempt-1\step-1000-training-state.pt'
        self.assertEqual(checkpoint_index_key(foreign, 1000, 1), r'attempts\attempt-1\step-1000-training-state.pt')
        for wrong in (
            foreign.replace('zenkai-ablation-skip-v2-dataset-1k', 'another-run'),
            foreign.replace('attempt-1', 'attempt-2'),
            foreign.replace('step-1000-training-state.pt', 'step-900-training-state.pt'),
        ):
            with self.subTest(path=wrong), self.assertRaises(ValueError):
                checkpoint_index_key(wrong, 1000, 1)

    @unittest.skipUnless((Path(__file__).resolve().parents[1]/'models/zenkai/experiments/zenkai-ablation-skip-v2-dataset-1k/step-1000-master-fp32.pt').is_file(), 'download the pinned Zenkai checkpoint first')
    def test_same_config_replacement_cannot_claim_completed_training(self):
        from zenkai_probe_evaluation import load_probe
        root=Path(__file__).resolve().parents[1]
        actual=root/'models/zenkai/experiments/zenkai-ablation-skip-v2-dataset-1k/step-1000-master-fp32.pt'
        saved=torch.load(actual,map_location='cpu',weights_only=True)
        with tempfile.TemporaryDirectory(dir=root/'diagnostics') as folder:
            path=Path(folder)/'replaced.pt'
            next(iter(saved['model'].values())).add_(.001)
            torch.save(saved,path)
            with self.assertRaisesRegex(ValueError,'training.*hash|hash.*training'):
                load_probe(path,1000)

    @unittest.skipUnless((Path(__file__).resolve().parents[1]/'models/zenkai/experiments/zenkai-ablation-skip-v2-dataset-1k/step-1000-master-fp32.pt').is_file(), 'download the pinned Zenkai checkpoint first')
    def test_reserved_loader_rejects_combined_factor_and_wrong_budget(self):
        from zenkai_probe_evaluation import load_probe
        from zenkai_v2_dataset_protocol import SOURCE,build_config
        from zenkai_ablation_skip import IdentitySkipProbe
        root=Path(__file__).resolve().parents[1]
        meta=json.loads((root/'models/zenkai/experiments'/SOURCE/'metadata.json').read_text())
        cfg=build_config(self.fixture()['config'],meta['checkpointsSha256']['initial-training-state.pt'],{})
        actual=root/'models/zenkai/experiments/zenkai-ablation-skip-v2-dataset-1k/step-1000-master-fp32.pt'
        # The positive case must originate from completed hash-pinned training;
        # an invented same-config finite checkpoint is now correctly rejected.
        model,_=load_probe(actual,1000)
        self.assertEqual(model.config['family'],'ZenkaiIdentitySkipProbe')
        with tempfile.TemporaryDirectory(dir=root/'diagnostics') as folder:
            path=Path(folder)/'candidate.pt';state=IdentitySkipProbe().state_dict()
            torch.save({'model':state,'config':cfg,'step':1000},path)
            with self.assertRaises(ValueError):load_probe(path,1000)
            for bad in ({**cfg,'codecFraction':.5},{**cfg,'seed':930},{**cfg,'datasetManifestSha256':'bad'}):
                torch.save({'model':state,'config':bad,'step':1000},path)
                with self.assertRaises(ValueError):load_probe(path,1000)
            torch.save({'model':state,'config':cfg,'step':1000},path)
            with self.assertRaises(ValueError):load_probe(path,3000)

    def fixture(self):
        root=Path(__file__).resolve().parents[1]
        cfg=json.loads((root/'models/zenkai/experiments/zenkai-ablation-spab-identity-skip-1k/config.json').read_text())
        model=torch.nn.Linear(3,3)
        optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,betas=(.9,.99),weight_decay=1e-5)
        scaler=type('Scaler',(),{'state_dict':lambda self:{'scale':65536.,'growth_factor':2.,'backoff_factor':.5,'growth_interval':2000,'_growth_tracker':0}})()
        indices=torch.randint(1792,(40000,),generator=torch.Generator().manual_seed(929))
        state=state_snapshot(model,optimizer,scaler,0,cfg,random.Random(929).getstate(),indices)
        state['rng']['cuda']=[torch.ones(20,dtype=torch.uint8)]
        return state

    @unittest.skipUnless((Path(__file__).resolve().parents[1]/'models/zenkai/experiments/zenkai-ablation-spab-identity-skip-1k/config.json').is_file(), 'download the Zenkai dataset protocol fixture first')
    def test_dataset_pair_exempts_only_membership_and_sampler(self):
        from zenkai_v2_dataset_protocol import build_config,validate_initial_pair
        ref=self.fixture();actual=copy.deepcopy(ref)
        actual['config']=build_config(ref['config'],'a'*64,{})
        actual['samplerIndices']=torch.randint(2394,(40000,),generator=torch.Generator().manual_seed(929))
        self.assertEqual(validate_initial_pair(actual,ref)['state'],'matched')
        for key,value in [('codecFraction',.5),('seed',930),('patchLR',96),('loss','zenkai_legacy_v3')]:
            wrong=copy.deepcopy(actual);wrong['config'][key]=value
            with self.assertRaises(ValueError,msg=key):validate_initial_pair(wrong,ref)
        for change in ('weights','datasetRng','scheduler','sampler'):
            wrong=copy.deepcopy(actual)
            if change=='weights':next(iter(wrong['model'].values())).add_(1)
            elif change=='datasetRng':wrong['datasetRng']=random.Random(930).getstate()
            elif change=='scheduler':wrong['scheduler']['kind']='constant'
            else:wrong['samplerIndices'][0]=2394
            with self.assertRaises(ValueError,msg=change):validate_initial_pair(wrong,ref)

    @unittest.skipUnless((Path(__file__).resolve().parents[1]/'models/zenkai/experiments/zenkai-ablation-spab-identity-skip-1k/config.json').is_file(), 'download the Zenkai dataset protocol fixture first')
    def test_full_initial_and_recovery_state_cannot_be_fabricated(self):
        from zenkai_v2_dataset_protocol import validate_snapshot,build_config
        state=self.fixture();validate_snapshot(state)
        for key in ('optimizer','scaler','rng','datasetRng','samplerIndices','scheduler'):
            bad=copy.deepcopy(state);bad.pop(key)
            with self.assertRaises(ValueError,msg=key):validate_snapshot(bad)
        bad=copy.deepcopy(state);bad['rng']['numpy'].pop('keys')
        with self.assertRaises(ValueError):validate_snapshot(bad)
        bad=copy.deepcopy(state);bad['step']=200
        with self.assertRaises(ValueError):validate_snapshot(bad)
        bad=copy.deepcopy(state);bad['config']=build_config(state['config'],'a'*64,{})
        bad['step']=100;bad['nextSampleOffset']=800;bad['scheduler']['completedStep']=100
        # Empty Adam state cannot be resumed as if100 updates occurred.
        with self.assertRaises(ValueError):validate_snapshot(bad)


if __name__=='__main__':unittest.main()
