"""Reserved dataset-only1k protocol and complete-state guards; no product registration."""
import copy
import json
from pathlib import Path, PureWindowsPath
import random
import numpy as np
import torch
from train_zenkai import ROOT,_sha
from zenkai_recipe_loss import _equal,validate_paired_initial_state

SOURCE='zenkai-ablation-spab-identity-skip-1k'
RUN='zenkai-ablation-skip-v2-dataset-1k'
BASE_SHA='61238cecf347280205d51d4de72ba13c1473adbcebc6b4ea588a1bfdf83a2124'
MIX_SHA='2d48aabf8b03023117a037539f22653ac81cb5631c087842757681a9737c4d62'
MIX_ROOT=ROOT/'native/training-data/zenkai-v2-mix'
IGNORED={'run','comparisonRun','singleChangedVariable','wrapperSha256','exportStatus',
         'sourceCodeSha256','sourceFilesSha256','gitWorkingTree','datasetManifestSha256'}


def build_config(reference,initial_hash,source_hashes):
    return {**reference,'run':RUN,'comparisonRun':SOURCE,'datasetManifestSha256':MIX_SHA,
        'sourceDatasetManifestSha256':BASE_SHA,'initialSourceSha256':initial_hash,
        'singleChangedVariable':'TRAIN union adds602 illustrations; matching uniform replacement sampler over2394 identities',
        'trainDomainCounts':{'anime':1792,'illustration':602},'trainImages':2394,
        'samplingPolicy':'uniform image with replacement, independent generator seed929,40000 indices',
        'sourceFilesSha256':source_hashes,'sourceCodeSha256':source_hashes.get('zenkai_v2_dataset_screen.py'),
        'wrapperSha256':source_hashes.get('zenkai_v2_dataset_protocol.py'),
        'exportStatus':'Research probe only; no ordinary model registration','releaseApproved':False}


def validate_config(config,reference):
    for key,value in reference.items():
        if key not in IGNORED:_equal(config.get(key),value,'config.'+key)
    expected={'run':RUN,'comparisonRun':SOURCE,'datasetManifestSha256':MIX_SHA,
        'sourceDatasetManifestSha256':BASE_SHA,'trainImages':2394,
        'trainDomainCounts':{'anime':1792,'illustration':602},
        'samplingPolicy':'uniform image with replacement, independent generator seed929,40000 indices',
        'codecFraction':.25,'loss':'zenkai_frequency','plannedSteps':1000,'scheduleTotalSteps':5000}
    if any(config.get(k)!=v for k,v in expected.items()):raise ValueError('Undeclared V2 dataset protocol change')
    if config.get('factorName') or config.get('schedulerExperiment'):raise ValueError('Combined factor forbidden')


def validate_snapshot(saved):
    cfg=saved.get('config',{});step=saved.get('step',-1)
    expected={'family':'ZenkaiIdentitySkipProbe','channels':32,'depth':5,'nativeScale':2,'spabIdentitySkipGain':1.,
        'loss':'zenkai_frequency','codecFraction':.25,'patchLR':128,'patchHR':256,'seed':929,
        'batch':4,'accumulation':2,'effectiveBatch':8,'scheduleTotalSteps':5000,'plannedSteps':1000}
    if any(cfg.get(k)!=v for k,v in expected.items()):raise ValueError('Wrong dataset-screen math')
    initial=cfg.get('run')==SOURCE and step==0 and cfg.get('datasetManifestSha256')==BASE_SHA
    new=cfg.get('run')==RUN and cfg.get('datasetManifestSha256')==MIX_SHA
    if not initial and not (new and (step==0 or 100<=step<=1000 and step%100==0)):
        raise ValueError('Only real source initialization or indexed V2 state permitted')
    required=('model','optimizer','scaler','rng','datasetRng','samplerIndices','nextSampleOffset','scheduler')
    if any(k not in saved for k in required):raise ValueError('Incomplete training state')
    if not saved['model'] or any(not isinstance(v,torch.Tensor) or v.dtype!=torch.float32 or not torch.isfinite(v).all() for v in saved['model'].values()):
        raise ValueError('Finite FP32 master required')
    rng=saved['rng']
    if any(k not in rng for k in ('python','torch','cuda','numpy')) or not rng['cuda']:raise ValueError('All RNG states required')
    for value in [rng['torch'],*rng['cuda']]:
        if not isinstance(value,torch.Tensor) or value.dtype!=torch.uint8 or value.ndim!=1 or not value.numel():raise ValueError('Malformed Torch/GPU RNG')
    try:
        random.Random().setstate(rng['python']);random.Random().setstate(saved['datasetRng'])
        s=rng['numpy'];np.random.RandomState().set_state((s['kind'],s['keys'].numpy().astype(np.uint32),s['position'],s['hasGauss'],s['cachedGauss']))
    except (KeyError,TypeError,ValueError,AttributeError) as error:raise ValueError('Malformed Python/dataset/NumPy RNG') from error
    indices=saved['samplerIndices'];count=2394 if new else 1792
    if not isinstance(indices,torch.Tensor) or indices.dtype!=torch.long or indices.ndim!=1 or len(indices)!=40000 or saved['nextSampleOffset']!=step*8:
        raise ValueError('Wrong sampler count/position')
    if torch.any(indices<0) or torch.any(indices>=count):raise ValueError('Sampler exceeds actual TRAIN range')
    if saved['scheduler']!={'kind':'explicit cosine with 300-step warmup','totalSteps':5000,'completedStep':step}:raise ValueError('Wrong scheduler state')
    opt=saved['optimizer'];scaler=saved['scaler']
    if not opt.get('param_groups') or any(k not in scaler for k in ('scale','growth_factor','backoff_factor','growth_interval','_growth_tracker')) or scaler['scale']<=0:
        raise ValueError('Incomplete optimizer/scaler')
    if step==0:
        if opt.get('state')!={}:raise ValueError('Initialization optimizer must be empty')
    else:
        if not opt.get('state') or 'resumeProgress' not in saved:raise ValueError('Missing optimizer/progress')
        ids=[i for group in opt['param_groups'] for i in group['params']]
        if set(ids)!=set(opt['state']):raise ValueError('Optimizer state does not cover every parameter')
        for state in opt['state'].values():
            if any(k not in state for k in ('step','exp_avg','exp_avg_sq')) or float(state['step'])!=step:raise ValueError('Wrong optimizer update count')
            if any(not torch.isfinite(v).all() for v in state.values() if isinstance(v,torch.Tensor)):raise ValueError('Nonfinite optimizer')


def validate_initial_pair(actual,reference):
    validate_snapshot(reference);validate_snapshot(actual)
    if reference['config']['run']!=SOURCE or reference['step']!=0 or actual['config']['run']!=RUN or actual['step']!=0:raise ValueError('Step-zero dataset pair required')
    validate_config(actual['config'],reference['config'])
    adjusted=copy.deepcopy(actual);adjusted['config']['datasetManifestSha256']=BASE_SHA
    adjusted['samplerIndices']=reference['samplerIndices']
    result=validate_paired_initial_state(adjusted,reference)
    return {**result,'stateExemptions':['samplerIndices'],'configExemptions':['datasetManifestSha256'],
        'scope':'Dataset membership and corresponding uniform sampler only; all math/RNG/optimizer/scaler/weights match'}


def checkpoint_index_key(last_checkpoint, step, attempt):
    if not isinstance(step, int) or step <= 0 or not isinstance(attempt, int) or attempt <= 0:
        raise ValueError('Invalid checkpoint step or attempt')
    expected = PureWindowsPath('attempts') / f'attempt-{attempt}' / f'step-{step}-training-state.pt'
    candidate = PureWindowsPath(str(last_checkpoint))
    expected_suffix = (RUN, *expected.parts)
    if candidate.is_absolute():
        actual = tuple(part.casefold() for part in candidate.parts[-len(expected_suffix):])
        if actual != tuple(part.casefold() for part in expected_suffix):
            raise ValueError('Checkpoint path does not match indexed run, attempt, and step')
    elif candidate.parts != expected.parts:
        raise ValueError('Relative checkpoint path does not match indexed attempt and step')
    return str(Path(*expected.parts))


def load_candidate(path):
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('Candidate outside workspace')
    operation=json.loads((ROOT/'diagnostics/zenkai-training/v2-dataset-1k-training/operation.json').read_text(encoding='utf-8'))
    folder=ROOT/'models/zenkai/experiments'/RUN
    index=json.loads((folder/'checkpoint-index.json').read_text(encoding='utf-8'))
    result=operation.get('result') or {}
    checkpoint_key=checkpoint_index_key(index.get('lastCheckpoint'),index.get('lastCompleteStep'),operation.get('selectedAttempt'))
    if (operation.get('state')!='PASS' or result.get('run')!=RUN or result.get('step')!=1000
            or _sha(path)!=result.get('sha256')
            or _sha(folder/'step-1000-training-state.pt')!=result.get('stateSha256')
            or index.get('lastCompleteStep')!=1000
            or index.get('checkpoints',{}).get(checkpoint_key)!=result.get('stateSha256')):
        raise ValueError('Completed training hash provenance does not match candidate/full state')
    saved=torch.load(path,map_location='cpu',weights_only=True)
    origin=ROOT/'models/zenkai/experiments'/SOURCE
    reference=json.loads((origin/'config.json').read_text(encoding='utf-8'))
    pins=json.loads((origin/'metadata.json').read_text(encoding='utf-8'))['checkpointsSha256']
    cfg=saved.get('config',{});validate_config(cfg,reference)
    if saved.get('step')!=1000 or cfg.get('initialSourceSha256')!=pins['initial-training-state.pt']:
        raise ValueError('Wrong V2 checkpoint budget/source')
    if _sha(origin/'initial-training-state.pt')!=pins['initial-training-state.pt']:raise ValueError('Original initialization changed')
    state=saved.get('model',{})
    if not state or any(v.dtype!=torch.float32 or not torch.isfinite(v).all() for v in state.values()):raise ValueError('Finite FP32 candidate required')
    from zenkai_ablation_skip import IdentitySkipProbe
    model=IdentitySkipProbe().eval();model.load_state_dict(state,strict=True)
    return model,{'config':cfg,'step':1000}
