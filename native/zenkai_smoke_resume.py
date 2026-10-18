"""Faithful continuation of the approved matched smoke pair1k→3k only."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parent/'.tools'))
import numpy as np
import torch
from torch.utils.data import DataLoader
from artifact_io import atomic_copy_file_new,atomic_json_write,atomic_torch_save,publish_training_checkpoint
from zenkai_experiments import state_snapshot,require_codec_cache
from zenkai_probe_evaluation import load_probe,load_reference_checkpoint
from train_zenkai import ROOT,_sha,learning_rate,update_composite,_LegacyAdapter
from lite_online_dataset import OnlineAnimeDataset
from zenkai_evaluation import ZenkaiFixedEvaluator
from zenkai_losses import ZenkaiLoss


def validate_resume(snapshot,run,target):
    pairs={'zenkai-ablation-frequency-1k':('Zenkai','zenkai-ablation-frequency-3k-continuation'),
        'zenkai-ablation-spab-identity-skip-1k':('ZenkaiIdentitySkipProbe','zenkai-ablation-spab-identity-skip-3k-continuation')}
    config=snapshot.get('config',{});source=config.get('run')
    if source not in pairs or (config.get('family'),run)!=pairs[source] or target!=3000 or snapshot.get('step')!=1000:
        raise ValueError('Only reviewed matched1k→3k smoke pair may resume')
    for key in ('optimizer','scaler','rng','datasetRng','samplerIndices'):
        if key not in snapshot:raise ValueError(f'Missing resume state:{key}')
    if config.get('scheduleTotalSteps')!=5000 or snapshot.get('scheduler',{}).get('totalSteps')!=5000 or snapshot['scheduler'].get('completedStep')!=1000:
        raise ValueError('Scheduler state changed')
    if snapshot.get('nextSampleOffset')!=1000*config.get('effectiveBatch',0):raise ValueError('Sampler position changed')
    if len(snapshot['samplerIndices'])<target*config['effectiveBatch']:raise ValueError('Sampler too short')


def restore_training_state(model,optimizer,scaler,snapshot,dataset):
    model.load_state_dict(snapshot['model'],strict=True)
    optimizer.load_state_dict(copy.deepcopy(snapshot['optimizer']))
    scaler.load_state_dict(snapshot['scaler'])
    dataset.rng.setstate(snapshot['datasetRng'])
    rng=snapshot['rng'];torch.set_rng_state(rng['torch'])
    if rng['cuda']:
        if not torch.cuda.is_available() or len(rng['cuda'])!=torch.cuda.device_count():raise ValueError('GPU RNG device count changed')
        torch.cuda.set_rng_state_all(rng['cuda'])
    random.setstate(rng['python'])
    state=rng['numpy'];np.random.set_state((state['kind'],state['keys'].numpy().astype(np.uint32),state['position'],state['hasGauss'],state['cachedGauss']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--run',required=True)
    parser.add_argument('--target',type=int,default=3000);args=parser.parse_args()
    source_path=args.checkpoint.resolve();snapshot=torch.load(source_path,map_location='cpu',weights_only=True)
    validate_resume(snapshot,args.run,args.target);source=snapshot['config'];source_run=source['run']
    if source_path!=ROOT/'models/zenkai/experiments'/source_run/'step-1000-training-state.pt':raise ValueError('Unexpected snapshot path')
    meta=json.loads((source_path.parent/'metadata.json').read_text(encoding='utf-8'))
    digest=_sha(source_path)
    if meta['checkpointsSha256'][source_path.name]!=digest:raise ValueError('Source snapshot changed')
    if not torch.version.hip or not torch.cuda.is_available() or str(torch.__version__)!=source['torch'] or torch.version.hip!=source['hip']:
        raise RuntimeError('Training stack differs')
    for name,expected in source['sourceFilesSha256'].items():
        if _sha(Path(__file__).with_name(name))!=expected:raise ValueError(f'Source training code changed:{name}')
    if source.get('wrapperSha256') and _sha(Path(__file__).with_name('zenkai_ablation_skip.py'))!=source['wrapperSha256']:
        raise ValueError('Skip wrapper changed')
    output=ROOT/'models/zenkai/experiments'/args.run
    if output.exists():raise FileExistsError('Preserving existing continuation')
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    master=source_path.parent/'step-1000-master-fp32.pt'
    model,saved=load_reference_checkpoint(master) if source['family']=='Zenkai' else load_probe(master)
    if any(not torch.equal(value,snapshot['model'][key]) for key,value in model.state_dict().items()):raise ValueError('Snapshot/master differ')
    model=model.cuda().float()
    dataset_root=ROOT/'native/training-data/zenkai'
    if _sha(dataset_root/'manifest.jsonl')!=source['datasetManifestSha256']:raise ValueError('Training manifest changed')
    cache=require_codec_cache(source)
    if _sha(cache/'complete.json')!=source['codecCacheCompleteSha256']:raise ValueError('Codec cache manifest changed')
    data=OnlineAnimeDataset(dataset_root,patch_lr=source['patchLR'],seed=source['seed'],codec_fraction=source['codecFraction'],codec_cache=cache)
    indices=snapshot['samplerIndices'].tolist();offset=snapshot['nextSampleOffset']
    if any(index<0 or index>=len(data) for index in indices):raise ValueError('Sampler corrupt')
    batches=iter(DataLoader(data,batch_size=source['batch'],sampler=indices[offset:],num_workers=0))
    evaluator=ZenkaiFixedEvaluator(dataset_root/'validation-fixed.json',per_case=32)
    if evaluator.sha256!=source['fixedValidationSha256']:raise ValueError('Validation changed')
    os.environ.setdefault('TORCH_HOME',str(ROOT/'diagnostics/native-cache/torch'))
    import lpips
    feature=lpips.LPIPS(net='squeeze').cuda().eval().requires_grad_(False)
    from model_registry import build_registry
    registry=build_registry(ROOT);legacy=registry.load('adore-lite-v3',device='cuda',precision='fp16')
    if legacy.artifact['sha256']!=source['legacyV3Sha256']:raise ValueError('Historical V3 changed')
    criterion=ZenkaiLoss(source['loss']).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,betas=(.9,.99),weight_decay=1e-5)
    scaler=torch.amp.GradScaler('cuda')
    # All constructors, including DataLoader iterator creation, precede RNG restore.
    restore_training_state(model,optimizer,scaler,snapshot,data)
    source_report=json.loads((source_path.parent/'report.json').read_text(encoding='utf-8'))
    config={**source,'run':args.run,'plannedSteps':3000,'continuedFrom':str(source_path),'continuedFromSha256':digest,
        'sourceRun':source_run,'resumeSourceSha256':_sha(Path(__file__)),'releaseApproved':False}
    output.mkdir(parents=True);atomic_json_write(config,output/'config.json')
    atomic_copy_file_new(source_path.parent/'best-composite-master-fp32.pt',output/'best-composite-master-fp32.pt')
    index_path=output/'checkpoint-index.json'
    index=publish_training_checkpoint(snapshot,output/'source-1000-training-state.pt',index_path,
        {'checkpoints':{},'lastCompleteStep':0,'lastCheckpoint':None},'source-1000-training-state.pt',1000,
        'source-1000-training-state.pt')
    history=list(source_report['history']);best=dict(source_report['best']);started=time.perf_counter()
    for step in range(1001,3001):
        model.train();optimizer.zero_grad(set_to_none=True);losses=[]
        lr=learning_rate(step,5000)
        for group in optimizer.param_groups:group['lr']=lr
        for _ in range(source['accumulation']):
            low,target=next(batches);low=low.cuda();target=target.cuda()
            with torch.autocast('cuda',dtype=torch.float16):prediction=model(low)
            loss,terms=criterion(prediction,target)
            if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss:{step}')
            losses.append(float(loss.detach()));scaler.scale(loss/source['accumulation']).backward()
        scaler.unscale_(optimizer);grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(grad):raise RuntimeError(f'Nonfinite gradient:{step}')
        old=scaler.get_scale();scaler.step(optimizer);scaler.update()
        if scaler.get_scale()<old:raise RuntimeError(f'AMP update skipped:{step}; stop')
        if step%100==0:print(json.dumps({'step':step,'loss':float(np.mean(losses)),'gradientNorm':float(grad),
            'lr':lr,'scaler':scaler.get_scale(),'updateSkipped':False,'samplesSinceResume':data.samples,
            'secondsPerAdditionalStep':(time.perf_counter()-started)/(step-1000)}),flush=True)
        if step%1000==0:
            model.eval();deploy=model.deploy_copy().cuda().half().eval()
            scores=evaluator.evaluate({'candidate':lambda x:deploy(x.half()).float(),'adore-lite-v3':_LegacyAdapter(legacy)},device='cuda',perceptual=feature)
            summary=update_composite(scores['cases'])
            state=state_snapshot(model,optimizer,scaler,step,config,data.rng.getstate(),indices)
            checkpoint={'model':state['model'],'config':config,'step':step}
            atomic_torch_save(checkpoint,output/f'step-{step}-master-fp32.pt')
            if summary['composite']<best['composite']:
                best={'step':step,'composite':summary['composite']};atomic_torch_save(checkpoint,output/'best-composite-master-fp32.pt')
            history.append({'step':step,'screening':scores,'summary':summary,'trainingLoss':float(np.mean(losses))})
            seconds=time.perf_counter()-started
            report={**config,'stepsExecuted':step,'optimizerUpdates':step,'additionalStepsExecuted':step-1000,
                'trainingSeconds':source_report['trainingSeconds']+seconds,'resumeTrainingSeconds':seconds,
                'samples':source_report['samples']+data.samples,'codecSamples':source_report['codecSamples']+data.codec_samples,
                **data.unique_crops.report_fields('uniqueSpatialCropsSinceResume'),'previousUniqueSpatialCrops':source_report['uniqueSpatialCrops'],
                'uniqueSpatialCrops':'NOT MEASURED:prior crop identity set not archived; no sum presented as union',
                'history':history,'best':best,'promoted':False}
            key=f'step-{step}-training-state.pt'
            index=publish_training_checkpoint(state,output/key,index_path,index,key,step,key,output/'report.json',report)
            print(json.dumps({'validationStep':step,'composite':summary['composite'],'best':best}),flush=True)
            del deploy,state
    if _sha(source_path)!=digest:raise ValueError('Source snapshot changed')
    registry.unload('adore-lite-v3')
    print(json.dumps({'finished':True,'stepsExecuted':3000,'output':str(output),'releaseApproved':False}),flush=True)


if __name__=='__main__':main()
