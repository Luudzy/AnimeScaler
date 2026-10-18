import copy,json,tempfile,unittest
from pathlib import Path
import torch
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]

class DomainSamplerTests(unittest.TestCase):
 @unittest.skipUnless((ROOT/'models/zenkai/experiments/zenkai-ablation-skip-v2-dataset-1k/initial-training-state.pt').is_file(), 'download the pinned Zenkai training state first')
 def test_probe_routes_only_reserved1000_budget(self):
  from zenkai_probe_evaluation import load_probe
  from zenkai_domain_sampler import build_config,INITIAL_SHA
  saved=self.source();saved['config']=build_config(saved['config'],INITIAL_SHA,{});saved['step']=1000
  with tempfile.TemporaryDirectory(dir=ROOT/'diagnostics') as folder:
   p=Path(folder)/'candidate.pt';torch.save(saved,p)
   with patch('zenkai_domain_sampler.load_candidate',return_value=('candidate',{'step':1000})) as loader:
    self.assertEqual(load_probe(p,1000)[0],'candidate');loader.assert_called_once_with(p)
   with self.assertRaises(ValueError):load_probe(p,3000)
 def source(self):
  return torch.load(ROOT/'models/zenkai/experiments/zenkai-ablation-skip-v2-dataset-1k/initial-training-state.pt',map_location='cpu',weights_only=True)
 def test_weighted_stream_is_deterministic_correct_domain_mass(self):
  from zenkai_domain_sampler import sample_indices,domain_weights
  domains=['anime']*1792+['illustration']*602
  w=domain_weights(domains)
  self.assertAlmostEqual(float(w[:1792].sum()),.9)
  self.assertAlmostEqual(float(w[1792:].sum()),.1)
  a=sample_indices(domains);b=sample_indices(domains)
  self.assertTrue(torch.equal(a,b));self.assertEqual(len(a),40000)
  self.assertTrue(torch.all((a>=0)&(a<2394)))
  self.assertTrue(.08<float((a[:8000]>=1792).float().mean())<.12)
  with self.assertRaises(ValueError):domain_weights(['anime']*2394)
 @unittest.skipUnless((ROOT/'models/zenkai/experiments/zenkai-ablation-skip-v2-dataset-1k/initial-training-state.pt').is_file(), 'download the pinned Zenkai training state first')
 def test_pair_and_config_exempt_only_sampler(self):
  from zenkai_domain_sampler import build_config,validate_initial_pair,sample_indices
  ref=self.source();a=copy.deepcopy(ref)
  a['config']=build_config(ref['config'],'35af820a66d155e0a52410a995bbc3212be2895e4c6bc475ee22a4f3c37341f1',{})
  a['samplerIndices']=sample_indices(['anime']*1792+['illustration']*602)
  self.assertEqual(validate_initial_pair(a,ref)['state'],'matched')
  for key,value in [('codecFraction',.5),('seed',930),('patchLR',96),('loss','zenkai_legacy_v3'),('illustrationProbability',.25),('datasetManifestSha256','bad')]:
   bad=copy.deepcopy(a);bad['config'][key]=value
   with self.assertRaises(ValueError):validate_initial_pair(bad,ref)
  bad=copy.deepcopy(a);next(iter(bad['model'].values())).add_(1)
  with self.assertRaises(ValueError):validate_initial_pair(bad,ref)
  bad=copy.deepcopy(a);bad['datasetRng']=copy.deepcopy(ref['rng']['python']);bad['datasetRng']=(2,*bad['datasetRng'][1:])
  with self.assertRaises(ValueError):validate_initial_pair(bad,ref)
 @unittest.skipUnless((ROOT/'models/zenkai/experiments/zenkai-ablation-skip-v2-dataset-1k/initial-training-state.pt').is_file(), 'download the pinned Zenkai training state first')
 def test_recovery_refuses_missing_state_wrong_position_and_nonfinite(self):
  from zenkai_domain_sampler import validate_snapshot,build_config,sample_indices
  ref=self.source();validate_snapshot(ref)
  a=copy.deepcopy(ref);a['config']=build_config(ref['config'],'35af820a66d155e0a52410a995bbc3212be2895e4c6bc475ee22a4f3c37341f1',{});a['samplerIndices']=sample_indices(['anime']*1792+['illustration']*602)
  validate_snapshot(a)
  for key in ('optimizer','scaler','rng','datasetRng','scheduler','samplerIndices'):
   bad=copy.deepcopy(a);bad.pop(key)
   with self.assertRaises(ValueError):validate_snapshot(bad)
  bad=copy.deepcopy(a);bad['nextSampleOffset']=8
  with self.assertRaises(ValueError):validate_snapshot(bad)
  bad=copy.deepcopy(a);next(iter(bad['model'].values())).fill_(float('nan'))
  with self.assertRaises(ValueError):validate_snapshot(bad)
if __name__=='__main__':unittest.main()
