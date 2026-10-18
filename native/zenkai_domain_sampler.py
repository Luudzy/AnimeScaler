"""Reserved10% illustration sampler probe; original union/math/full states retained."""
import copy,json
from pathlib import Path
import torch
from torch.utils.data import WeightedRandomSampler
from train_zenkai import ROOT,_sha
from zenkai_v2_dataset_protocol import MIX_ROOT,MIX_SHA,validate_snapshot as validate_union_snapshot
from zenkai_recipe_loss import _equal,validate_paired_initial_state
SOURCE='zenkai-ablation-skip-v2-dataset-1k'
RUN='zenkai-ablation-skip-v2-domain10-1k'
INITIAL_SHA='35af820a66d155e0a52410a995bbc3212be2895e4c6bc475ee22a4f3c37341f1'
IGNORED={'run','comparisonRun','singleChangedVariable','wrapperSha256','exportStatus','sourceCodeSha256','sourceFilesSha256','gitWorkingTree','initialSourceSha256','samplingPolicy'}

def domain_weights(domains):
 if len(domains)!=2394 or domains.count('anime')!=1792 or domains.count('illustration')!=602:raise ValueError('Frozen TRAIN domain counts changed')
 return torch.tensor([.9/1792 if d=='anime' else .1/602 for d in domains],dtype=torch.float64)

def sample_indices(domains):
 return torch.tensor(list(WeightedRandomSampler(domain_weights(domains),40000,replacement=True,generator=torch.Generator().manual_seed(929))),dtype=torch.long)

def build_config(reference,initial_hash,source_hashes):
 return {**reference,'run':RUN,'comparisonRun':SOURCE,'initialSourceSha256':initial_hash,
  'singleChangedVariable':'Domain sampling:uniform union versus weighted replacement anime.9/illustration.1 on identical2394 TRAIN',
  'illustrationProbability':.1,'domainProbabilities':{'anime':.9,'illustration':.1},
  'samplingPolicy':'WeightedRandomSampler replacement, per-image domain mass, independent generatorseed929,40000 indices',
  'sourceFilesSha256':source_hashes,'sourceCodeSha256':source_hashes.get('zenkai_domain_screen.py'),
  'wrapperSha256':source_hashes.get('zenkai_domain_sampler.py'),'exportStatus':'Reserved sampler probe; no ordinary registration','releaseApproved':False}

def validate_config(config,reference):
 for key,value in reference.items():
  if key not in IGNORED:_equal(config.get(key),value,'config.'+key)
 expected={'run':RUN,'comparisonRun':SOURCE,'initialSourceSha256':INITIAL_SHA,'datasetManifestSha256':MIX_SHA,
  'illustrationProbability':.1,'domainProbabilities':{'anime':.9,'illustration':.1},
  'samplingPolicy':'WeightedRandomSampler replacement, per-image domain mass, independent generatorseed929,40000 indices'}
 if any(config.get(k)!=v for k,v in expected.items()) or config.get('factorName') or config.get('schedulerExperiment'):raise ValueError('Undeclared domain sampler factor/config change')

def validate_snapshot(saved):
 cfg=saved.get('config',{});step=saved.get('step',-1)
 if cfg.get('run')==SOURCE and step==0:
  validate_union_snapshot(saved);return
 if cfg.get('run')!=RUN or not (step==0 or 100<=step<=1000 and step%100==0):raise ValueError('Only original uniform initialization or indexed sampler states')
 reference=json.loads((ROOT/'models/zenkai/experiments'/SOURCE/'config.json').read_text(encoding='utf-8'))
 validate_config(cfg,reference)
 adjusted=copy.copy(saved);adjusted['config']={**cfg,'run':SOURCE}
 validate_union_snapshot(adjusted)

def validate_initial_pair(actual,reference):
 validate_snapshot(actual);validate_snapshot(reference)
 if actual['config']['run']!=RUN or reference['config']['run']!=SOURCE:raise ValueError('Wrong sampler pair')
 validate_config(actual['config'],reference['config'])
 adjusted=copy.copy(actual);adjusted['samplerIndices']=reference['samplerIndices']
 result=validate_paired_initial_state(adjusted,reference)
 return {**result,'stateExemptions':['samplerIndices'],'scope':'Only sampling distribution differs; union/model/optimizer/scaler/all RNG identical'}

def load_candidate(path):
 path=Path(path).resolve()
 if not path.is_relative_to(ROOT):raise ValueError('Candidate outside workspace')
 operation=json.loads((ROOT/'diagnostics/zenkai-training/domain10-1k-training/operation.json').read_text(encoding='utf-8'));result=operation.get('result') or {}
 folder=ROOT/'models/zenkai/experiments'/RUN;index=json.loads((folder/'checkpoint-index.json').read_text(encoding='utf-8'))
 if operation.get('state')!='PASS' or result.get('run')!=RUN or result.get('step')!=1000 or _sha(path)!=result.get('sha256') or _sha(folder/'step-1000-training-state.pt')!=result.get('stateSha256') or index.get('lastCompleteStep')!=1000 or index['checkpoints'].get(str(Path(index['lastCheckpoint']).relative_to(folder)))!=result.get('stateSha256'):
  raise ValueError('Completed training hash provenance mismatch')
 source=ROOT/'models/zenkai/experiments'/SOURCE
 if _sha(source/'initial-training-state.pt')!=INITIAL_SHA:raise ValueError('Original uniform source state changed')
 saved=torch.load(path,map_location='cpu',weights_only=True);cfg=saved.get('config',{})
 validate_config(cfg,json.loads((source/'config.json').read_text(encoding='utf-8')))
 if saved.get('step')!=1000 or not saved.get('model') or any(v.dtype!=torch.float32 or not torch.isfinite(v).all() for v in saved['model'].values()):raise ValueError('Wrong/nonfinite final master')
 from zenkai_ablation_skip import IdentitySkipProbe
 model=IdentitySkipProbe().eval();model.load_state_dict(saved['model'],strict=True)
 return model,{'config':cfg,'step':1000}
