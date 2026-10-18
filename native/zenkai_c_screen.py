"""Conditional C screening, faithful Codec50 3000→5000 TOTAL, no Stage2."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent));sys.path.insert(0,str(Path(__file__).resolve().parent/'.tools'))
import numpy as np
import torch
from torch.utils.data import DataLoader
from train_zenkai import ROOT,_sha,learning_rate,update_composite,_LegacyAdapter
from hip_operation import save_json
from artifact_io import atomic_copy_file,atomic_copy_file_new,atomic_torch_save
from zenkai_experiments import state_snapshot,require_codec_cache
from zenkai_smoke_resume import restore_training_state
from zenkai_probe_evaluation import load_probe
from zenkai_ablation_skip import IdentitySkipProbe
from zenkai_recipe_loss import make_experiment_loss
from lite_online_dataset import OnlineAnimeDataset
from zenkai_evaluation import ZenkaiFixedEvaluator
SOURCE='zenkai-ablation-skip-factor-codec50-3k-continuation'
RUN='zenkai-lite-v1-c-codec50-5k'


def validate_c_resume(saved,target):
    cfg=saved.get('config',{});step=saved.get('step',-1)
    expected={'family':'ZenkaiIdentitySkipProbe','factorName':'codec50','codecFraction':.5,
              'loss':'zenkai_frequency','channels':32,'depth':5,'nativeScale':2,'spabIdentitySkipGain':1.,
              'scheduleTotalSteps':5000,'seed':929,'patchLR':128,'batch':4,'accumulation':2,'effectiveBatch':8}
    if target!=5000 or any(cfg.get(k)!=v for k,v in expected.items()):raise ValueError('Exact C5k codec50 protocol only')
    source=cfg.get('run')==SOURCE and step==3000 and cfg.get('plannedSteps')==3000
    recovery=(cfg.get('run')==RUN and 3100<=step<=5000 and step%100==0 and cfg.get('plannedSteps')==5000 and cfg.get('cSourceRun')==SOURCE)
    if not source and not recovery:raise ValueError('Reserved actual3k source or indexed C recovery only')
    required=('model','optimizer','scaler','rng','datasetRng','samplerIndices','nextSampleOffset','scheduler')
    if any(k not in saved for k in required):raise ValueError('Incomplete full state')
    if not saved['model'] or any(v.dtype!=torch.float32 or not torch.isfinite(v).all() for v in saved['model'].values()):raise ValueError('Invalid FP32 master')
    rng=saved['rng']
    if any(k not in rng for k in ('python','torch','cuda','numpy')) or not rng['cuda']:raise ValueError('Missing RNG state')
    for state in [rng['torch'],*rng['cuda']]:
        if state.dtype!=torch.uint8 or state.ndim!=1 or not state.numel():raise ValueError('Malformed RNG')
    import random
    try:
        random.Random().setstate(rng['python']);random.Random().setstate(saved['datasetRng'])
        n=rng['numpy'];np.random.RandomState().set_state((n['kind'],n['keys'].numpy().astype(np.uint32),n['position'],n['hasGauss'],n['cachedGauss']))
    except (ValueError,TypeError,KeyError) as error:raise ValueError('Malformed Python/NumPy/dataset RNG') from error
    if saved['scheduler']!={'kind':'explicit cosine with 300-step warmup','totalSteps':5000,'completedStep':step}:raise ValueError('Scheduler timeline changed')
    indices=saved['samplerIndices']
    if indices.dtype!=torch.long or indices.ndim!=1 or len(indices)<40000 or saved['nextSampleOffset']!=step*8:raise ValueError('Sampler position changed')
    optimizer=saved['optimizer'];scaler=saved['scaler']
    if not optimizer.get('state') or not optimizer.get('param_groups') or any(k not in scaler for k in ('scale','growth_factor','backoff_factor','growth_interval','_growth_tracker')) or scaler['scale']<=0:raise ValueError('Incomplete optimizer/scaler')
    for state in optimizer['state'].values():
        if any(k not in state for k in ('step','exp_avg','exp_avg_sq')) or float(state['step'])!=step:raise ValueError('Optimizer state step differs')
        if any(not torch.isfinite(v).all() for v in state.values() if isinstance(v,torch.Tensor)):raise ValueError('Nonfinite optimizer state')


def load_c_candidate(path):
    saved=torch.load(path,map_location='cpu',weights_only=True);cfg=saved.get('config',{})
    origin=ROOT/'models/zenkai/experiments'/SOURCE
    source=json.loads((origin/'config.json').read_text());pins=json.loads((origin/'metadata.json').read_text())['checkpointsSha256']
    if _sha(origin/'step-3000-training-state.pt')!=pins['step-3000-training-state.pt']:raise ValueError('C source state changed')
    if saved.get('step')!=5000 or cfg.get('run')!=RUN or cfg.get('plannedSteps')!=5000 or cfg.get('cSourceRun')!=SOURCE or cfg.get('cSourceSha256')!=pins['step-3000-training-state.pt']:raise ValueError('Unreviewed C candidate')
    if any(cfg.get(k)!=v for k,v in source.items() if k not in ('run','plannedSteps')):raise ValueError('C source protocol changed')
    if not saved.get('model') or any(v.dtype!=torch.float32 or not torch.isfinite(v).all() for v in saved['model'].values()):raise ValueError('Finite C FP32 master required')
    model=IdentitySkipProbe().eval();model.load_state_dict(saved['model'],strict=True)
    return model,{'config':cfg,'step':5000}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--attempt',type=int,choices=(1,2,3),required=True);p.add_argument('--target',type=int,default=5000);args=p.parse_args()
    gate=json.loads((ROOT/'docs/zenkai-scheduler-decision.json').read_text())
    if gate.get('CRecipe')!='Skip+Frequency+Codec50+cosine' or gate.get('C5kAuthorizedByEvidence') is not True:raise ValueError('C recipe not frozen')
    if _sha(ROOT/'diagnostics/zenkai-codec-pair-3k/comparison.json')!=gate['codecComparisonSha256']:raise ValueError('C decision inputs changed')
    origin=ROOT/'models/zenkai/experiments'/SOURCE;folder=ROOT/'models/zenkai/experiments'/RUN
    source=json.loads((origin/'config.json').read_text());pins=json.loads((origin/'metadata.json').read_text())['checkpointsSha256']
    path=args.checkpoint.resolve()
    if path==origin/'step-3000-training-state.pt':digest=pins[path.name]
    elif path.is_relative_to(folder/'attempts'):digest=json.loads((folder/'checkpoint-index.json').read_text())['checkpoints'].get(str(path.relative_to(folder)))
    else:raise ValueError('Unexpected C source path')
    if not digest or _sha(path)!=digest:raise ValueError('C snapshot SHA changed')
    saved=torch.load(path,map_location='cpu',weights_only=True);validate_c_resume(saved,args.target)
    if not torch.version.hip or not torch.cuda.is_available() or str(torch.__version__)!=source['torch'] or torch.version.hip!=source['hip']:raise RuntimeError('Training stack changed')
    for name,pin in source['sourceFilesSha256'].items():
        if _sha(origin/'source'/name)!=pin:raise ValueError('Archived source changed')
        if name!='zenkai_factor_screens.py' and _sha(Path(__file__).with_name(name))!=pin:raise ValueError('Mathematical dependency changed: '+name)
    if _sha(Path(__file__).with_name('zenkai_smoke_resume.py'))!=source['restoreHelperSha256']:raise ValueError('Restore helper changed')
    config={**source,'run':RUN,'plannedSteps':5000,'cSourceRun':SOURCE,'cSourceSha256':pins['step-3000-training-state.pt'],
            'cResumeSourceSha256':_sha(Path(__file__)),'releaseApproved':False}
    if saved['step']==3000 and saved['config']!=source:raise ValueError('Original C source configuration differs')
    if saved['step']>3000 and saved['config']!=config:raise ValueError('C recovery config/source changed')
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    model=IdentitySkipProbe().cuda().float();root=ROOT/'native/training-data/zenkai';cache=require_codec_cache(config)
    if _sha(root/'manifest.jsonl')!=source['datasetManifestSha256'] or _sha(cache/'complete.json')!=source['codecCacheCompleteSha256']:raise ValueError('Training data changed')
    data=OnlineAnimeDataset(root,patch_lr=128,seed=929,codec_fraction=.5,codec_cache=cache);indices=saved['samplerIndices'].tolist()
    if any(i<0 or i>=len(data) for i in indices):raise ValueError('Sampler corrupt')
    batches=iter(DataLoader(data,batch_size=4,sampler=indices[saved['nextSampleOffset']:],num_workers=0))
    evaluator=ZenkaiFixedEvaluator(root/'validation-fixed.json',per_case=32)
    if evaluator.sha256!=source['fixedValidationSha256']:raise ValueError('Fixed validation changed')
    os.environ.setdefault('TORCH_HOME',str(ROOT/'diagnostics/native-cache/torch'))
    import lpips
    feature=lpips.LPIPS(net='squeeze').cuda().eval().requires_grad_(False)
    from model_registry import build_registry
    registry=build_registry(ROOT);legacy=registry.load('adore-lite-v3',device='cuda',precision='fp16')
    if legacy.artifact['sha256']!=source['legacyV3Sha256']:raise ValueError('V3 changed')
    criterion=make_experiment_loss('zenkai_frequency').cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,betas=(.9,.99),weight_decay=1e-5)
    scaler=torch.amp.GradScaler('cuda');restore_training_state(model,optimizer,scaler,saved,data)
    progress=saved['resumeProgress'];history=copy.deepcopy(progress['history']);best=copy.deepcopy(progress['best'])
    if not folder.exists():
        folder.mkdir();save_json(folder/'config.json',config);shutil.copytree(origin/'source',folder/'source')
        shutil.copyfile(Path(__file__),folder/'source/zenkai_c_screen.py')
        atomic_copy_file_new(origin/'best-composite-master-fp32.pt',folder/'best-composite-master-fp32.pt')
        save_json(folder/'checkpoint-index.json',{'checkpoints':{},'lastCompleteStep':3000})
    elif json.loads((folder/'config.json').read_text())!=config:raise ValueError('C immutable config changed')
    out=folder/'attempts'/f'attempt-{args.attempt}';out.mkdir(parents=True)
    save_json(out/'resume.json',{'step':saved['step'],'sha256':digest,'source':str(path),'stateRestored':True})
    started=time.perf_counter()
    for step in range(saved['step']+1,5001):
        model.train();optimizer.zero_grad(set_to_none=True);losses=[];lr=learning_rate(step,5000)
        for group in optimizer.param_groups:group['lr']=lr
        for _ in range(2):
            low,target=next(batches);low=low.cuda();target=target.cuda()
            with torch.autocast('cuda',dtype=torch.float16):prediction=model(low)
            loss,_=criterion(prediction,target)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            losses.append(float(loss.detach()));scaler.scale(loss/2).backward()
        scaler.unscale_(optimizer);grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(grad):raise RuntimeError('Nonfinite gradient')
        old=scaler.get_scale();scaler.step(optimizer);scaler.update()
        if scaler.get_scale()<old:raise RuntimeError('AMP update skipped')
        if step%1000==0:
            model.eval();deploy=model.deploy_copy().cuda().half().eval()
            scores=evaluator.evaluate({'candidate':lambda x:deploy(x.half()).float(),'adore-lite-v3':_LegacyAdapter(legacy)},device='cuda',perceptual=feature)
            summary=update_composite(scores['cases']);history.append({'step':step,'screening':scores,'summary':summary,'trainingLoss':float(np.mean(losses))})
            if summary['composite']<best['composite']:best={'step':step,'composite':summary['composite']}
            del deploy
        if step%100==0:
            state=state_snapshot(model,optimizer,scaler,step,config,data.rng.getstate(),indices)
            state['resumeProgress']={'trainingSeconds':progress['trainingSeconds']+time.perf_counter()-started,
                'samples':progress['samples']+data.samples,'codecSamples':progress['codecSamples']+data.codec_samples,'history':history,'best':best}
            dest=out/f'step-{step}-training-state.pt';pending=dest.with_suffix('.pt.pending');torch.save(state,pending);pending.replace(dest)
            index=json.loads((folder/'checkpoint-index.json').read_text());index['checkpoints'][str(dest.relative_to(folder))]=_sha(dest)
            index.update(lastCompleteStep=step,lastCheckpoint=str(dest));save_json(folder/'checkpoint-index.json',index)
            save_json(folder/'report.json',{**config,**state['resumeProgress'],'stepsExecuted':step,'optimizerUpdates':step,
                     'uniqueSpatialCrops':'NOT MEASURED across continuation','promoted':False})
            if step%1000==0:
                master={'model':state['model'],'config':config,'step':step};atomic_torch_save(master,folder/f'step-{step}-master-fp32.pt')
                atomic_copy_file(dest,folder/f'step-{step}-training-state.pt')
                if best['step']==step:atomic_torch_save(master,folder/'best-composite-master-fp32.pt')
            print(json.dumps({'step':step,'loss':float(np.mean(losses)),'lr':lr,'gradientNorm':float(grad),'updateSkipped':False}),flush=True);del state
    final=folder/'step-5000-master-fp32.pt'
    if not final.exists():
        indexed=Path(json.loads((folder/'checkpoint-index.json').read_text())['lastCheckpoint'])
        state=torch.load(indexed,map_location='cpu',weights_only=True);validate_c_resume(state,5000)
        if state['step']!=5000:raise ValueError('No complete final C state')
        atomic_torch_save({'model':state['model'],'config':config,'step':5000},final)
        atomic_copy_file(indexed,folder/'step-5000-training-state.pt')
        if state['resumeProgress']['best']['step']==5000:atomic_copy_file(final,folder/'best-composite-master-fp32.pt')
    if _sha(path)!=digest:raise ValueError('C source changed')
    print(json.dumps({'state':'PASS','stepsExecuted':5000,'run':RUN}),flush=True)


if __name__=='__main__':main()
