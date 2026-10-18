"""Separate matched V3 control and short Zenkai experiments; no product changes."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import re
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parent/'.tools'))
import numpy as np
import torch
from torch.utils.data import DataLoader,RandomSampler
from lite_v3 import AdoreLiteV3
from zenkai_model import ZenkaiLite
from zenkai_losses import ZenkaiLoss
from train_zenkai import ROOT,learning_rate,update_composite,_LegacyAdapter,_sha
from lite_online_dataset import OnlineAnimeDataset
from zenkai_evaluation import ZenkaiFixedEvaluator


def build_experiment_model(kind):
    if kind=='control-v3':return AdoreLiteV3(24,4,.2)
    if kind=='zenkai-a':return ZenkaiLite(32,5)
    if kind=='zenkai-b':return ZenkaiLite(32,6)
    raise ValueError('Unknown experiment architecture')


def load_control_checkpoint(path):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    config=saved.get('config',{})
    expected={'family':'AdoreControl','channels':24,'blocks':4,'residualScale':.2,'nativeScale':2}
    if any(config.get(key)!=value for key,value in expected.items()) or saved.get('step',0)<1:
        raise ValueError('Not the exact reviewed retrained V3 control')
    state=saved.get('model',{})
    if not state or any(value.dtype!=torch.float32 or not torch.isfinite(value).all() for value in state.values()):
        raise ValueError('Control master must be finite FP32')
    model=build_experiment_model('control-v3').eval();model.load_state_dict(state,strict=True)
    return model,{'step':saved['step'],'config':config}


def protocol_config(source,kind,run,steps,loss):
    if kind=='control-v3' and loss!=source['loss']:
        raise ValueError('Control must use the exact screening loss')
    result={**source,'run':run,'plannedSteps':steps,'scheduleTotalSteps':5000,
            'experimentKind':kind,'loss':loss,'lossWeights':ZenkaiLoss(loss).weights,
            'releaseApproved':False,'resumeStateAvailable':True}
    if kind=='control-v3':
        result.update(family='AdoreControl',name='Adore V3 Retrained Control',architecture='Exact historical AdoreLiteV3',channels=24,
            blocks=4,depth=4,residualScale=.2)
        result.pop('tailConvolutions',None)
    else:result.update(family='Zenkai',channels=32,depth=5 if kind=='zenkai-a' else 6)
    return result


def require_codec_cache(config):
    cache=Path(config['codecCache']) if config.get('codecCache') else None
    if config.get('codecFraction',0)>0 and (cache is None or not (cache/'complete.json').is_file()):
        raise ValueError('Configured codec cache is missing; refusing to change the training distribution')
    return cache


def validate_experiment_identity(kind,run,steps):
    if not 1<=steps<=5000:raise ValueError('Experiments capped at 5000 steps')
    if kind=='control-v3':
        if run!='adore-v3-retrained-control-5k' or steps!=5000:raise ValueError('Control must be exact5k with reserved identity')
    elif kind in ('zenkai-a','zenkai-b'):
        if not re.fullmatch(r'zenkai-ablation-[a-z0-9-]+',run):raise ValueError('Zenkai must use a separate ablation identity')
    else:raise ValueError('Unknown experiment architecture')


def cpu_tree(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {key:cpu_tree(item) for key,item in value.items()}
    if isinstance(value,list):return [cpu_tree(item) for item in value]
    if isinstance(value,tuple):return tuple(cpu_tree(item) for item in value)
    return value


def state_snapshot(model,optimizer,scaler,step,config,dataset_rng,sampler_indices):
    numpy_state=np.random.get_state()
    return {'model':{key:value.detach().cpu().float().clone() for key,value in model.state_dict().items()},
        'optimizer':cpu_tree(optimizer.state_dict()),'scaler':scaler.state_dict(),'step':step,'config':config,
        'rng':{'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
               'python':random.getstate(),'numpy':{'kind':numpy_state[0],
               'keys':torch.from_numpy(numpy_state[1].astype(np.int64)), 'position':numpy_state[2],
               'hasGauss':numpy_state[3],'cachedGauss':numpy_state[4]}},
        'datasetRng':dataset_rng,'samplerIndices':torch.as_tensor(sampler_indices,dtype=torch.long),
        'nextSampleOffset':step*config.get('effectiveBatch',8),
        'scheduler':{'kind':'explicit cosine with 300-step warmup','totalSteps':5000,'completedStep':step}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=('control-v3','zenkai-a','zenkai-b'),required=True)
    parser.add_argument('--run',required=True);parser.add_argument('--steps',type=int,default=5000)
    parser.add_argument('--loss',choices=('zenkai_frequency','zenkai_baseline','zenkai_frequency_edge'),default='zenkai_frequency')
    parser.add_argument('--interval',type=int,default=1000)
    args=parser.parse_args()
    try:validate_experiment_identity(args.kind,args.run,args.steps)
    except ValueError as error:parser.error(str(error))
    if args.interval<1:parser.error('Interval must be positive')
    if not torch.cuda.is_available() or not torch.version.hip:raise RuntimeError('Configured ROCm required')
    output=ROOT/'models/zenkai/experiments'/args.run
    if output.exists():raise FileExistsError('Refusing to overwrite experiment')
    source=json.loads((ROOT/'models/zenkai/experiments/zenkai-lite-v1-a-5k/config.json').read_text(encoding='utf-8'))
    config=protocol_config(source,args.kind,args.run,args.steps,args.loss)
    torch.set_num_threads(2);torch.manual_seed(config['seed']);random.seed(config['seed']);np.random.seed(config['seed'])
    torch.backends.cudnn.benchmark=False
    model=build_experiment_model(args.kind).cuda()
    config['parameters']=sum(p.numel() for p in model.parameters())
    config['referenceProtocolRun']=source['run'];config['sourceCodeSha256']=_sha(Path(__file__))
    config['sourceFilesSha256']={name:_sha(Path(__file__).with_name(name)) for name in (
        'zenkai_experiments.py','train_zenkai.py','zenkai_model.py','zenkai_losses.py',
        'lite_v3.py','lite_online_dataset.py','lite_degradations.py','zenkai_evaluation.py')}
    config['gitWorkingTree']='Uncommitted experimental changes; source file hashes recorded'
    config['torch']=str(torch.__version__);config['hip']=torch.version.hip;config['gpu']=torch.cuda.get_device_name()
    if config['torch']!=source['torch'] or config['hip']!=source['hip']:raise RuntimeError('Training stack differs from A/B')
    dataset=ROOT/'native/training-data/zenkai'
    if _sha(dataset/'manifest.jsonl')!=source['datasetManifestSha256']:raise RuntimeError('Training manifest changed')
    codec_cache=require_codec_cache(config)
    config['codecCacheCompleteSha256']=_sha(codec_cache/'complete.json') if codec_cache else None
    data=OnlineAnimeDataset(dataset,patch_lr=config['patchLR'],seed=config['seed'],
        codec_fraction=config['codecFraction'],codec_cache=codec_cache)
    sampler=RandomSampler(data,replacement=True,num_samples=5000*config['effectiveBatch'],
        generator=torch.Generator().manual_seed(config['seed']))
    indices=list(sampler)
    batches=iter(DataLoader(data,batch_size=config['batch'],sampler=indices,num_workers=0))
    evaluator=ZenkaiFixedEvaluator(dataset/'validation-fixed.json',per_case=32)
    if evaluator.sha256!=source['fixedValidationSha256']:raise RuntimeError('Validation manifest changed')
    os.environ.setdefault('TORCH_HOME',str(ROOT/'diagnostics/native-cache/torch'))
    import lpips
    feature=lpips.LPIPS(net='squeeze').cuda().eval().requires_grad_(False)
    from model_registry import build_registry
    registry=build_registry(ROOT);legacy=registry.load('adore-lite-v3',device='cuda',precision='fp16')
    if legacy.artifact['sha256']!=source['legacyV3Sha256']:raise RuntimeError('Historical V3 changed')
    criterion=ZenkaiLoss(args.loss).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,betas=(.9,.99),weight_decay=1e-5)
    scaler=torch.amp.GradScaler('cuda')
    output.mkdir(parents=True)
    (output/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    torch.save(state_snapshot(model,optimizer,scaler,0,config,data.rng.getstate(),indices),output/'initial-training-state.pt')
    started=time.perf_counter();history=[];best={};updates=0
    for step in range(1,args.steps+1):
        model.train();optimizer.zero_grad(set_to_none=True);losses=[]
        lr=learning_rate(step,5000)
        for group in optimizer.param_groups:group['lr']=lr
        for accumulation in range(config['accumulation']):
            low,target=next(batches);low=low.cuda();target=target.cuda()
            with torch.autocast('cuda',dtype=torch.float16):prediction=model(low)
            loss,terms=criterion(prediction,target)
            if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss at {step}')
            losses.append(float(loss.detach()));scaler.scale(loss/config['accumulation']).backward()
        scaler.unscale_(optimizer);grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(grad):raise RuntimeError(f'Nonfinite gradient at {step}')
        old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
        if scaler.get_scale()<old_scale:raise RuntimeError(f'AMP update skipped at {step}; stop for diagnosis')
        updates+=1
        if step%100==0:
            print(json.dumps({'step':step,'loss':float(np.mean(losses)),'lossTerms':{key:float(v.detach()) for key,v in terms.items()},
                'lr':lr,'gradientNorm':float(grad),'samples':data.samples,'uniqueCrops':len(data.unique_crops),
                'codecSamples':data.codec_samples,'updateSkipped':False,'secondsPerStep':(time.perf_counter()-started)/step}),flush=True)
        if step%args.interval==0 or step==args.steps:
            model.eval();deploy=model.deploy_copy() if hasattr(model,'deploy_copy') else copy.deepcopy(model)
            deploy=deploy.cuda().half().eval()
            scores=evaluator.evaluate({'candidate':lambda x:deploy(x.half()).float(),'adore-lite-v3':_LegacyAdapter(legacy)},
                device='cuda',perceptual=feature)
            summary=update_composite(scores['cases'])
            snapshot=state_snapshot(model,optimizer,scaler,step,config,data.rng.getstate(),indices)
            torch.save(snapshot,output/f'step-{step}-training-state.pt')
            master={'model':snapshot['model'],'config':config,'step':step}
            torch.save(master,output/f'step-{step}-master-fp32.pt')
            if not best or summary['composite']<best['composite']:
                best={'step':step,'composite':summary['composite']};torch.save(master,output/'best-composite-master-fp32.pt')
            history.append({'step':step,'screening':scores,'summary':summary,'trainingLoss':float(np.mean(losses))})
            report={**config,'stepsExecuted':step,'optimizerUpdates':updates,'trainingSeconds':time.perf_counter()-started,
                'samples':data.samples,'uniqueSpatialCrops':len(data.unique_crops),'codecSamples':data.codec_samples,
                'history':history,'best':best,'promoted':False}
            (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            print(json.dumps({'validationStep':step,'composite':summary['composite'],'best':best}),flush=True)
            del deploy,snapshot
    registry.unload('adore-lite-v3')
    print(json.dumps({'finished':True,'stepsExecuted':args.steps,'output':str(output),'releaseApproved':False}),flush=True)


if __name__=='__main__':main()
