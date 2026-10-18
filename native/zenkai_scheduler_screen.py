"""NEW paired scheduler smoke from preserved initialization, never the failed step200 run."""
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
from hip_operation import save_json
from artifact_io import atomic_copy_file,atomic_torch_save
from train_zenkai import ROOT,_sha,learning_rate,update_composite,_LegacyAdapter
from zenkai_experiments import state_snapshot,require_codec_cache
from zenkai_smoke_resume import restore_training_state
from zenkai_ablation_skip import IdentitySkipProbe
from zenkai_factor_screens import SCHEDULE_KIND,validate_factor_initial
from zenkai_recipe_loss import make_experiment_loss
from lite_online_dataset import OnlineAnimeDataset
from zenkai_evaluation import ZenkaiFixedEvaluator

SOURCE='zenkai-ablation-spab-identity-skip-1k'
RUN='zenkai-ablation-skip-factor-scheduler-constant-v2-1k'


def schedule_rate(step):return learning_rate(step,5000) if step<=300 else 2e-4


def validate_snapshot(saved):
    cfg=saved.get('config',{});step=saved.get('step',-1)
    expected={'family':'ZenkaiIdentitySkipProbe','channels':32,'depth':5,'nativeScale':2,
              'spabIdentitySkipGain':1.,'loss':'zenkai_frequency','codecFraction':.25,'patchLR':128,
              'seed':929,'batch':4,'accumulation':2,'effectiveBatch':8,'scheduleTotalSteps':5000}
    if any(cfg.get(k)!=v for k,v in expected.items()):raise ValueError('Changed scheduler-screen protocol')
    initial=cfg.get('run')==SOURCE and step==0
    recovery=(cfg.get('run')==RUN and 100<=step<=1000 and step%100==0 and
              cfg.get('factorName')=='scheduler-constant' and cfg.get('schedulerExperiment')==SCHEDULE_KIND)
    if not initial and not recovery:raise ValueError('Only preserved initialization or new indexed recovery')
    required=('model','optimizer','scaler','rng','datasetRng','samplerIndices','nextSampleOffset','scheduler')
    if any(k not in saved for k in required):raise ValueError('Incomplete checkpoint')
    if not saved['model'] or any(v.dtype!=torch.float32 or not torch.isfinite(v).all() for v in saved['model'].values()):raise ValueError('Invalid FP32 master')
    rng=saved['rng']
    if any(k not in rng for k in ('python','torch','cuda','numpy')) or not rng['cuda']:raise ValueError('Missing RNG state')
    for state in [rng['torch'],*rng['cuda']]:
        if state.dtype!=torch.uint8 or state.ndim!=1 or state.numel()==0:raise ValueError('Malformed RNG')
    import random
    try:
        random.Random().setstate(rng['python']);random.Random().setstate(saved['datasetRng'])
    except (ValueError,TypeError) as error:raise ValueError('Malformed Python/dataset RNG') from error
    if saved['scheduler']!={'kind':'explicit cosine with 300-step warmup' if initial else SCHEDULE_KIND,
                             'totalSteps':5000,'completedStep':step}:raise ValueError('Wrong scheduler state')
    indices=saved['samplerIndices']
    if indices.dtype!=torch.long or indices.ndim!=1 or len(indices)<8000 or saved['nextSampleOffset']!=step*8:raise ValueError('Wrong sampler position')
    optimizer=saved['optimizer'];scaler=saved['scaler']
    if not optimizer.get('param_groups') or any(k not in scaler for k in ('scale','growth_factor','backoff_factor','growth_interval','_growth_tracker')) or scaler['scale']<=0:raise ValueError('Incomplete optimizer/scaler')
    if initial and optimizer.get('state')!={}:raise ValueError('Initialization optimizer is not empty')
    if recovery:
        if not optimizer.get('state'):raise ValueError('Missing optimizer moments')
        for state in optimizer['state'].values():
            if any(k not in state for k in ('step','exp_avg','exp_avg_sq')) or float(state['step'])!=step:
                raise ValueError('Wrong optimizer step')
            if any(not torch.isfinite(v).all() for v in state.values() if isinstance(v,torch.Tensor)):raise ValueError('Nonfinite moments')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--attempt',type=int,choices=(1,2,3),required=True);args=p.parse_args()
    origin=ROOT/'models/zenkai/experiments'/SOURCE;folder=ROOT/'models/zenkai/experiments'/RUN
    initial_path=origin/'initial-training-state.pt';pins=json.loads((origin/'metadata.json').read_text())['checkpointsSha256']
    if _sha(initial_path)!=pins[initial_path.name]:raise ValueError('Baseline initialization changed')
    original=torch.load(initial_path,map_location='cpu',weights_only=True);source=original['config'];path=args.checkpoint.resolve()
    if path==initial_path:digest=pins[initial_path.name]
    elif path.is_relative_to(folder/'attempts'):
        digest=json.loads((folder/'checkpoint-index.json').read_text())['checkpoints'].get(str(path.relative_to(folder)))
    else:raise ValueError('Invalid scheduler source path')
    if not digest or _sha(path)!=digest:raise ValueError('Snapshot integrity failed')
    saved=torch.load(path,map_location='cpu',weights_only=True);validate_snapshot(saved)
    if not torch.version.hip or not torch.cuda.is_available() or str(torch.__version__)!=source['torch'] or torch.version.hip!=source['hip']:raise RuntimeError('Training stack changed')
    for name,pin in source['sourceFilesSha256'].items():
        if name!='zenkai_experiments.py' and _sha(Path(__file__).with_name(name))!=pin:raise ValueError('Mathematical dependency changed: '+name)
    if _sha(Path(__file__).with_name('zenkai_ablation_skip.py'))!=source['wrapperSha256']:raise ValueError('Skip forward changed')
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    config={**source,'run':RUN,'plannedSteps':1000,'learningRate':[.0002,.0002],
        'factorName':'scheduler-constant','schedulerExperiment':SCHEDULE_KIND,'comparisonRun':SOURCE,
        'singleChangedVariable':'same warmup300 then constant2e-4','sourceRun':SOURCE,
        'initialSourceSha256':pins[initial_path.name],'wrapperSha256':_sha(Path(__file__)),
        'exportStatus':'Separate probe family, no ordinary export','releaseApproved':False,
        'sourceFilesSha256':{name:_sha(Path(__file__).with_name(name)) for name in
          (*source['sourceFilesSha256'],'zenkai_scheduler_screen.py','zenkai_smoke_resume.py','zenkai_ablation_skip.py','zenkai_recipe_loss.py')}}
    if saved['step'] and saved['config']!=config:raise ValueError('Recovery configuration changed')
    dataset_root=ROOT/'native/training-data/zenkai';cache=require_codec_cache(config)
    if _sha(dataset_root/'manifest.jsonl')!=source['datasetManifestSha256'] or _sha(cache/'complete.json')!=source['codecCacheCompleteSha256']:raise ValueError('Data manifest changed')
    model=IdentitySkipProbe().cuda().float();data=OnlineAnimeDataset(dataset_root,patch_lr=128,seed=929,codec_fraction=.25,codec_cache=cache)
    indices=saved['samplerIndices'].tolist()
    if any(i<0 or i>=len(data) for i in indices):raise ValueError('Sampler corrupt')
    batches=iter(DataLoader(data,batch_size=4,sampler=indices[saved['nextSampleOffset']:],num_workers=0))
    evaluator=ZenkaiFixedEvaluator(dataset_root/'validation-fixed.json',per_case=32)
    if evaluator.sha256!=source['fixedValidationSha256']:raise ValueError('Validation changed')
    os.environ.setdefault('TORCH_HOME',str(ROOT/'diagnostics/native-cache/torch'))
    import lpips
    feature=lpips.LPIPS(net='squeeze').cuda().eval().requires_grad_(False)
    from model_registry import build_registry
    registry=build_registry(ROOT);legacy=registry.load('adore-lite-v3',device='cuda',precision='fp16')
    if legacy.artifact['sha256']!=source['legacyV3Sha256']:raise ValueError('V3 changed')
    criterion=make_experiment_loss('zenkai_frequency').cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,betas=(.9,.99),weight_decay=1e-5)
    scaler=torch.amp.GradScaler('cuda');restore_training_state(model,optimizer,scaler,saved,data)
    if not folder.exists():
        folder.mkdir();save_json(folder/'config.json',config)
        initial=state_snapshot(model,optimizer,scaler,0,config,data.rng.getstate(),indices);initial['scheduler']['kind']=SCHEDULE_KIND
        pairing=validate_factor_initial(initial,original,'scheduler-constant');pairing.update(referenceSha256=pins[initial_path.name],reference=str(initial_path))
        save_json(folder/'initial-pairing.json',pairing);atomic_torch_save(initial,folder/'initial-training-state.pt')
        archive=folder/'source';archive.mkdir()
        for name in config['sourceFilesSha256']:shutil.copyfile(Path(__file__).with_name(name),archive/name)
        save_json(folder/'checkpoint-index.json',{'checkpoints':{},'lastCompleteStep':0})
        print(json.dumps({'initialPairing':'matched','newExperiment':RUN,'sourceStep':0}),flush=True)
    elif json.loads((folder/'config.json').read_text())!=config:raise ValueError('Run configuration changed')
    out=folder/'attempts'/f'attempt-{args.attempt}';out.mkdir(parents=True)
    progress=saved.get('resumeProgress',{'trainingSeconds':0.,'samples':0,'codecSamples':0,'history':[],'best':{}})
    started=time.perf_counter();history=copy.deepcopy(progress['history']);best=copy.deepcopy(progress['best'])
    save_json(out/'resume.json',{'step':saved['step'],'sha256':digest,'source':str(path),'stateRestored':True})
    for step in range(saved['step']+1,1001):
        model.train();optimizer.zero_grad(set_to_none=True);losses=[];lr=schedule_rate(step)
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
        if step==1000:
            model.eval();deploy=model.deploy_copy().cuda().half().eval()
            scores=evaluator.evaluate({'candidate':lambda x:deploy(x.half()).float(),'adore-lite-v3':_LegacyAdapter(legacy)},device='cuda',perceptual=feature)
            summary=update_composite(scores['cases']);best={'step':1000,'composite':summary['composite']}
            history.append({'step':step,'screening':scores,'summary':summary,'trainingLoss':float(np.mean(losses))});del deploy
        if step%100==0:
            state=state_snapshot(model,optimizer,scaler,step,config,data.rng.getstate(),indices);state['scheduler']['kind']=SCHEDULE_KIND
            state['resumeProgress']={'trainingSeconds':progress['trainingSeconds']+time.perf_counter()-started,
                'samples':progress['samples']+data.samples,'codecSamples':progress['codecSamples']+data.codec_samples,
                'history':history,'best':best}
            checkpoint=out/f'step-{step}-training-state.pt';pending=checkpoint.with_suffix('.pt.pending');torch.save(state,pending);pending.replace(checkpoint)
            index=json.loads((folder/'checkpoint-index.json').read_text());index['checkpoints'][str(checkpoint.relative_to(folder))]=_sha(checkpoint)
            index.update(lastCompleteStep=step,lastCheckpoint=str(checkpoint));save_json(folder/'checkpoint-index.json',index)
            save_json(folder/'report.json',{**config,**state['resumeProgress'],'stepsExecuted':step,'optimizerUpdates':step,
                 'uniqueSpatialCrops':'NOT MEASURED across recovery attempts','promoted':False})
            print(json.dumps({'step':step,'loss':float(np.mean(losses)),'lr':lr,'gradientNorm':float(grad),'updateSkipped':False}),flush=True)
            del state
    # Also materialize from an intact final recovery state if the process stopped after indexing it.
    final=torch.load(Path(json.loads((folder/'checkpoint-index.json').read_text())['lastCheckpoint']),map_location='cpu',weights_only=True)
    master={'model':final['model'],'config':config,'step':1000}
    atomic_torch_save(master,folder/'step-1000-master-fp32.pt');atomic_torch_save(master,folder/'best-composite-master-fp32.pt')
    atomic_copy_file(Path(json.loads((folder/'checkpoint-index.json').read_text())['lastCheckpoint']),folder/'step-1000-training-state.pt')
    print(json.dumps({'state':'PASS','stepsExecuted':1000,'run':RUN}),flush=True)


if __name__=='__main__':main()
