"""Reserved faithful Codec50 continuation to3000 TOTAL, including full-state HIP recovery."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parent/'.tools'))
import numpy as np
import torch
from torch.utils.data import DataLoader
from hip_operation import save_json
from artifact_io import atomic_copy_file,atomic_copy_file_new,atomic_torch_save,publish_training_checkpoint
from zenkai_experiments import state_snapshot,require_codec_cache
from zenkai_smoke_resume import restore_training_state
from zenkai_probe_evaluation import load_probe
from zenkai_recipe_loss import make_experiment_loss
from lite_online_dataset import OnlineAnimeDataset
from zenkai_evaluation import ZenkaiFixedEvaluator
from train_zenkai import ROOT,_sha,learning_rate,update_composite,_LegacyAdapter

SOURCE='zenkai-ablation-skip-factor-codec50-1k'
RUN='zenkai-ablation-skip-factor-codec50-3k-continuation'


def validate_codec_resume(snapshot,run,target):
    cfg=snapshot.get('config',{});step=snapshot.get('step',0)
    expected={'family':'ZenkaiIdentitySkipProbe','factorName':'codec50','codecFraction':.5,
        'loss':'zenkai_frequency','channels':32,'depth':5,'nativeScale':2,'spabIdentitySkipGain':1.,
        'scheduleTotalSteps':5000,'batch':4,'accumulation':2,'effectiveBatch':8,'patchLR':128,'seed':929}
    if run!=RUN or target!=3000 or any(cfg.get(k)!=v for k,v in expected.items()):
        raise ValueError('Only exact Codec50 protocol to3000 TOTAL')
    if cfg.get('run')==SOURCE:
        if step!=1000 or cfg.get('plannedSteps')!=1000:raise ValueError('Exact1k source required')
    elif cfg.get('run')==RUN:
        if cfg.get('sourceRun')!=SOURCE or cfg.get('plannedSteps')!=3000 or not 1100<=step<=3000 or step%100:
            raise ValueError('Invalid recovery identity/position')
    else:raise ValueError('Wrong source run')
    required=('model','optimizer','scaler','rng','datasetRng','samplerIndices','nextSampleOffset','scheduler')
    if any(k not in snapshot for k in required):raise ValueError('Incomplete full-state checkpoint')
    model=snapshot['model']
    if not model or any(not isinstance(v,torch.Tensor) or v.dtype!=torch.float32 or not torch.isfinite(v).all() for v in model.values()):
        raise ValueError('Finite FP32 master required')
    rng=snapshot['rng']
    if not isinstance(rng,dict) or any(k not in rng for k in ('python','torch','cuda','numpy')) or not rng['cuda']:
        raise ValueError('All Python/Torch/GPU/NumPy RNG states required')
    for value in [rng['torch'],*rng['cuda']]:
        if not isinstance(value,torch.Tensor) or value.dtype!=torch.uint8 or value.ndim!=1 or not value.numel():
            raise ValueError('Malformed RNG state')
    for state in (rng['python'],snapshot['datasetRng']):
        import random
        try:random.Random().setstate(state)
        except (TypeError,ValueError) as error:raise ValueError('Malformed Python/dataset RNG') from error
    if snapshot['scheduler']!={'kind':'explicit cosine with 300-step warmup','totalSteps':5000,'completedStep':step}:
        raise ValueError('Original5000-step scheduler required')
    indices=snapshot['samplerIndices']
    if not isinstance(indices,torch.Tensor) or indices.dtype!=torch.long or indices.ndim!=1 or len(indices)<target*8 or snapshot['nextSampleOffset']!=step*8:
        raise ValueError('Invalid sampler state/offset')
    optimizer=snapshot['optimizer'];scaler=snapshot['scaler']
    if not optimizer.get('state') or not optimizer.get('param_groups') or any(k not in scaler for k in ('scale','growth_factor','backoff_factor','growth_interval','_growth_tracker')) or scaler['scale']<=0:
        raise ValueError('Complete optimizer/scaler required')
    for state in optimizer['state'].values():
        if any(k not in state for k in ('step','exp_avg','exp_avg_sq')) or float(state['step'])!=step:
            raise ValueError('Optimizer steps differ from checkpoint')
        if any(not torch.isfinite(v).all() for v in state.values() if isinstance(v,torch.Tensor)):
            raise ValueError('Nonfinite optimizer state')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--attempt',type=int,required=True,choices=(1,2,3))
    parser.add_argument('--target',type=int,default=3000)
    args=parser.parse_args();path=args.checkpoint.resolve()
    origin=ROOT/'models/zenkai/experiments'/SOURCE
    output=ROOT/'models/zenkai/experiments'/RUN
    source=json.loads((origin/'config.json').read_text())
    pins=json.loads((origin/'metadata.json').read_text())['checkpointsSha256']
    digest=_sha(path)
    if path==origin/'step-1000-training-state.pt':
        if digest!=pins[path.name]:raise ValueError('Source state hash changed')
    elif path.is_relative_to(output/'attempts'):
        index=json.loads((output/'checkpoint-index.json').read_text())
        if index['checkpoints'].get(str(path.relative_to(output)))!=digest:raise ValueError('Recovery hash missing/mismatched')
    else:raise ValueError('Reserved source/recovery path required')
    saved=torch.load(path,map_location='cpu',weights_only=True)
    validate_codec_resume(saved,RUN,args.target)
    for key,value in source.items():
        if key not in ('run','plannedSteps') and saved['config'].get(key)!=value:raise ValueError('Changed config:'+key)
    for name,expected in source['sourceFilesSha256'].items():
        if _sha(origin/'source'/name)!=expected:raise ValueError('Archived source changed:'+name)
        # Original wrapper's CLI guard was subsequently fixed; it does not implement model/data/loss.
        if name!='zenkai_factor_screens.py' and _sha(Path(__file__).with_name(name))!=expected:
            raise ValueError('Mathematical/training dependency changed:'+name)
    if not torch.version.hip or not torch.cuda.is_available() or str(torch.__version__)!=source['torch'] or torch.version.hip!=source['hip']:
        raise RuntimeError('Training stack differs')
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    master=origin/'step-1000-master-fp32.pt'
    if _sha(master)!=pins[master.name]:raise ValueError('Master hash changed')
    model,_=load_probe(master)
    if saved['step']==1000 and any(not torch.equal(v,saved['model'][k]) for k,v in model.state_dict().items()):
        raise ValueError('Initial master/state differ')
    model=model.cuda().float()
    dataset_root=ROOT/'native/training-data/zenkai';cache=require_codec_cache(source)
    if _sha(dataset_root/'manifest.jsonl')!=source['datasetManifestSha256'] or _sha(cache/'complete.json')!=source['codecCacheCompleteSha256']:
        raise ValueError('Dataset/codec manifest changed')
    data=OnlineAnimeDataset(dataset_root,patch_lr=source['patchLR'],seed=source['seed'],codec_fraction=.5,codec_cache=cache)
    indices=saved['samplerIndices'].tolist()
    if any(i<0 or i>=len(data) for i in indices):raise ValueError('Sampler corrupt')
    batches=iter(DataLoader(data,batch_size=4,sampler=indices[saved['nextSampleOffset']:],num_workers=0))
    evaluator=ZenkaiFixedEvaluator(dataset_root/'validation-fixed.json',per_case=32)
    if evaluator.sha256!=source['fixedValidationSha256']:raise ValueError('Fixed validation changed')
    os.environ.setdefault('TORCH_HOME',str(ROOT/'diagnostics/native-cache/torch'))
    import lpips
    feature=lpips.LPIPS(net='squeeze').cuda().eval().requires_grad_(False)
    from model_registry import build_registry
    registry=build_registry(ROOT);legacy=registry.load('adore-lite-v3',device='cuda',precision='fp16')
    if legacy.artifact['sha256']!=source['legacyV3Sha256']:raise ValueError('HistoricalV3 changed')
    criterion=make_experiment_loss(source['loss']).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,betas=(.9,.99),weight_decay=1e-5)
    scaler=torch.amp.GradScaler('cuda')
    restore_training_state(model,optimizer,scaler,saved,data)
    source_report=json.loads((origin/'report.json').read_text())
    progress=saved.get('resumeProgress',{'history':source_report['history'],'best':source_report['best'],
        'trainingSeconds':source_report['trainingSeconds'],'samples':source_report['samples'],'codecSamples':source_report['codecSamples']})
    history=copy.deepcopy(progress['history']);best=copy.deepcopy(progress['best'])
    config={**source,'run':RUN,'sourceRun':SOURCE,'plannedSteps':3000,
        'continuedFromSha256':pins['step-1000-training-state.pt'],'resumeSourceSha256':_sha(Path(__file__)),
        'restoreHelperSha256':_sha(Path(__file__).with_name('zenkai_smoke_resume.py')),'releaseApproved':False}
    if not output.exists():
        output.mkdir(parents=True);save_json(output/'config.json',config)
        shutil.copytree(origin/'source',output/'source')
        shutil.copyfile(Path(__file__),output/'source/zenkai_codec_resume.py')
        shutil.copyfile(Path(__file__).with_name('zenkai_smoke_resume.py'),output/'source/zenkai_smoke_resume.py')
        atomic_copy_file_new(origin/'best-composite-master-fp32.pt',output/'best-composite-master-fp32.pt')
        save_json(output/'checkpoint-index.json',{'checkpoints':{},'lastCompleteStep':1000})
    elif json.loads((output/'config.json').read_text())!=config:raise ValueError('Continuation config/source changed')
    attempt=output/'attempts'/f'attempt-{args.attempt}';attempt.mkdir(parents=True)
    start=saved['step'];started=time.perf_counter()
    save_json(attempt/'resume.json',{'step':start,'source':str(path),'sha256':digest,'status':'PASS','stateRestored':True})
    for step in range(start+1,3001):
        model.train();optimizer.zero_grad(set_to_none=True);losses=[]
        lr=learning_rate(step,5000)
        for group in optimizer.param_groups:group['lr']=lr
        for _ in range(2):
            low,target=next(batches);low=low.cuda();target=target.cuda()
            with torch.autocast('cuda',dtype=torch.float16):prediction=model(low)
            loss,terms=criterion(prediction,target)
            if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss:{step}')
            losses.append(float(loss.detach()));scaler.scale(loss/2).backward()
        scaler.unscale_(optimizer);grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(grad):raise RuntimeError(f'Nonfinite gradient:{step}')
        old=scaler.get_scale();scaler.step(optimizer);scaler.update()
        if scaler.get_scale()<old:raise RuntimeError(f'AMP update skipped:{step}')
        if step%1000==0:
            model.eval();deploy=model.deploy_copy().cuda().half().eval()
            scores=evaluator.evaluate({'candidate':lambda x:deploy(x.half()).float(),'adore-lite-v3':_LegacyAdapter(legacy)},device='cuda',perceptual=feature)
            summary=update_composite(scores['cases'])
            history.append({'step':step,'screening':scores,'summary':summary,'trainingLoss':float(np.mean(losses))})
            if not best or summary['composite']<best['composite']:best={'step':step,'composite':summary['composite']}
            del deploy
        if step%100==0:
            elapsed=time.perf_counter()-started
            state=state_snapshot(model,optimizer,scaler,step,config,data.rng.getstate(),indices)
            state['resumeProgress']={'history':history,'best':best,'trainingSeconds':progress['trainingSeconds']+elapsed,
                'samples':progress['samples']+data.samples,'codecSamples':progress['codecSamples']+data.codec_samples}
            destination=attempt/f'step-{step}-training-state.pt'
            if destination.exists():raise FileExistsError('Preserve recovery snapshot')
            report={**config,**state['resumeProgress'],'stepsExecuted':step,'optimizerUpdates':step,
                'additionalStepsExecuted':step-1000,'uniqueSpatialCrops':'NOT MEASURED:original identity set unavailable',
                **data.unique_crops.report_fields('uniqueSpatialCropsThisAttempt'),'promoted':False}
            index_path=output/'checkpoint-index.json'
            index=json.loads(index_path.read_text())
            index=publish_training_checkpoint(state,destination,index_path,index,
                str(destination.relative_to(output)),step,str(destination),output/'report.json',report)
            if step%1000==0:
                checkpoint={'model':state['model'],'config':config,'step':step}
                atomic_torch_save(checkpoint,output/f'step-{step}-master-fp32.pt')
                atomic_copy_file(destination,output/f'step-{step}-training-state.pt')
                if best['step']==step:atomic_torch_save(checkpoint,output/'best-composite-master-fp32.pt')
            print(json.dumps({'step':step,'loss':float(np.mean(losses)),'lr':lr,'gradientNorm':float(grad),
                'updateSkipped':False,'resumeAttempt':args.attempt,'samplesTotal':report['samples'],'codecSamplesTotal':report['codecSamples']}),flush=True)
            del state
    if _sha(path)!=digest:raise ValueError('Resume source changed')
    print(json.dumps({'status':'PASS','stepsExecuted':3000,'output':str(output)}),flush=True)


if __name__=='__main__':main()
