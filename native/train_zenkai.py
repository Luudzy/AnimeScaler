"""Matched 5k Zenkai A/B screening. No automatic model promotion."""
import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE/'.tools'))
from server import ROOT
os.environ.setdefault('TORCH_HOME',str(ROOT/'diagnostics/native-cache/torch'))
import numpy as np
import torch
from artifact_io import atomic_torch_save, atomic_json_write
from torch.utils.data import DataLoader,RandomSampler
from lite_online_dataset import OnlineAnimeDataset
from training_amp import amp_optimizer_step
from zenkai_evaluation import ZenkaiFixedEvaluator
from zenkai_losses import PRESETS,ZenkaiLoss
from zenkai_model import ZenkaiLite


def validate_options(depth,steps,run):
    if depth not in (5,6):raise ValueError('Depth must be 5 or 6')
    if type(steps) is not int or steps<1:raise ValueError('Steps must be positive')
    if not re.fullmatch(r'zenkai-lite-v1-(?:a-5k|b-5k|smoke(?:-\d+)?)',run):
        raise ValueError('Run name must be a reviewed Zenkai A, B or smoke run')


def learning_rate(step,total):
    return (1e-5+(2e-4-1e-5)*.5*(1+math.cos(math.pi*(step-1)/max(1,total-1))))*min(1,step/300)


def update_composite(report):
    keys=('mae','edgeMae','laplacianMae','ssim','lpipsSqueeze','msSsim')
    result={case:dict(values['candidate']) for case,values in report.items()}
    ratios=[]
    for case,values in report.items():
        candidate=values['candidate'];previous=values['adore-lite-v3']
        ratios.append(candidate['mae']/max(previous['mae'],1e-8)
                      +.3*candidate['edgeMae']/max(previous['edgeMae'],1e-8)
                      +.1*candidate['laplacianMae']/max(previous['laplacianMae'],1e-8)
                      +.1*(1-candidate['ssim'])/max(1-previous['ssim'],1e-6)
                      +.15*candidate['lpipsSqueeze']/max(previous['lpipsSqueeze'],1e-6)
                      +.1*(1-candidate['msSsim'])/max(1-previous['msSsim'],1e-6))
    return {'byCase':result,'composite':float(np.mean(ratios)),
            'separateMetricsRetained':list(keys)}


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class _LegacyAdapter(torch.nn.Module):
    def __init__(self,source):
        super().__init__();self.source=source
        # Register the frozen network so the evaluator can discover its FP16 dtype.
        self.network=source.backend.model
    def forward(self,value):return self.source.infer(value)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--depth',type=int,choices=(5,6),required=True)
    parser.add_argument('--steps',type=int,default=5000)
    parser.add_argument('--run',required=True)
    parser.add_argument('--seed',type=int,default=929)
    parser.add_argument('--patch-lr',type=int,choices=(96,128),default=128)
    parser.add_argument('--batch',type=int,default=4)
    parser.add_argument('--accumulation',type=int,default=2)
    parser.add_argument('--loss',choices=tuple(PRESETS),default='zenkai_frequency')
    parser.add_argument('--interval',type=int,default=1000)
    parser.add_argument('--per-case',type=int,default=32)
    parser.add_argument('--codec-fraction',type=float,default=.25)
    args=parser.parse_args()
    try:validate_options(args.depth,args.steps,args.run)
    except ValueError as error:parser.error(str(error))
    if min(args.batch,args.accumulation,args.interval,args.per_case)<1:parser.error('batch/accumulation/interval/per-case must be positive')
    if not 0<=args.codec_fraction<=1:parser.error('codec-fraction must be between zero and one')
    if not torch.cuda.is_available() or not torch.version.hip:raise RuntimeError('Zenkai screening requires the configured ROCm GPU')
    dataset=ROOT/'native/training-data/zenkai';fixed=dataset/'validation-fixed.json'
    output=ROOT/'models/zenkai/experiments'/args.run
    if not (dataset/'manifest.jsonl').is_file() or not fixed.is_file():raise RuntimeError('Validated Zenkai manifests are missing')
    if output.exists():raise FileExistsError(f'Refusing to overwrite experiment: {output}')
    if '-smoke' in args.run and args.steps>50:parser.error('Smoke run is capped at 50 steps')
    if args.steps==5000 and args.run.endswith('-5k') and args.interval>args.steps:parser.error('No validation would be recorded')
    torch.set_num_threads(2);torch.manual_seed(args.seed);random.seed(args.seed);np.random.seed(args.seed)
    torch.backends.cudnn.benchmark=False
    output.mkdir(parents=True)
    codec_cache=ROOT/'native/training-data/v31/codec-full'
    if not (codec_cache/'complete.json').is_file():codec_cache=None
    model=ZenkaiLite(channels=32,depth=args.depth).cuda()
    parameter_count=sum(parameter.numel() for parameter in model.parameters())
    data=OnlineAnimeDataset(dataset,patch_lr=args.patch_lr,seed=args.seed,
                            codec_fraction=args.codec_fraction,codec_cache=codec_cache)
    rng=random.Random(args.seed)
    sampler=RandomSampler(data,replacement=True,num_samples=args.steps*args.batch*args.accumulation,
                          generator=torch.Generator().manual_seed(args.seed))
    batches=iter(DataLoader(data,batch_size=args.batch,sampler=sampler,num_workers=0))
    evaluator=ZenkaiFixedEvaluator(fixed,per_case=args.per_case)
    import lpips
    feature=lpips.LPIPS(net='squeeze').cuda().eval().requires_grad_(False)
    weights=ZenkaiLoss(args.loss).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,betas=(.9,.99),weight_decay=1e-5)
    scaler=torch.amp.GradScaler('cuda')
    registry=None
    from model_registry import build_registry
    registry=build_registry(ROOT)
    legacy=registry.load('adore-lite-v3','torch',device='cuda',precision='fp16')
    config={'family':'Zenkai','name':'Zenkai Lite V1 experimental','run':args.run,'architecture':'ESPAN-inspired x2, GRep SPAB',
        'channels':32,'depth':args.depth,'tailConvolutions':2,'nativeScale':2,'parameters':parameter_count,
        'seed':args.seed,'optimizer':'AdamW','learningRate':[2e-4,1e-5],'betas':[.9,.99],'weightDecay':1e-5,
        'batch':args.batch,'accumulation':args.accumulation,'effectiveBatch':args.batch*args.accumulation,
        'patchLR':args.patch_lr,'patchHR':args.patch_lr*2,'degradation':'online clean/heavy plus codec-intra pairs',
        'codecFraction':args.codec_fraction if codec_cache else 0,'codecCache':str(codec_cache) if codec_cache else None,
        'loss':args.loss,'lossWeights':weights.weights,'datasetManifestSha256':_sha(dataset/'manifest.jsonl'),
        'fixedValidationSha256':evaluator.sha256,'legacyV3Sha256':legacy.artifact['sha256'],
        'gpu':torch.cuda.get_device_name(),'hip':torch.version.hip,'torch':str(torch.__version__),
        'gitCommit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'realVideoValidation':'REAL VIDEO VALIDATION NOT EXECUTED','releaseApproved':False}
    atomic_json_write(config,output/'config.json')
    started=time.perf_counter();history=[];best={};optimizer_updates=0
    model.train()
    for step in range(1,args.steps+1):
        model.train();optimizer.zero_grad(set_to_none=True);step_losses=[];last_terms={}
        lr=learning_rate(step,args.steps)
        for group in optimizer.param_groups:group['lr']=lr
        for _ in range(args.accumulation):
            low,target=next(batches);low=low.cuda();target=target.cuda()
            with torch.autocast('cuda',dtype=torch.float16):prediction=model(low)
            loss,last_terms=weights(prediction,target)
            if not torch.isfinite(loss):raise RuntimeError(f'Non-finite loss at step {step}')
            step_losses.append(float(loss.detach()))
            scaler.scale(loss/args.accumulation).backward()
        scaler.unscale_(optimizer)
        gradient_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(gradient_norm):raise RuntimeError(f'Non-finite gradients at step {step}')
        did_update=amp_optimizer_step(scaler,optimizer);optimizer_updates+=did_update
        if step%100==0:
            torch.cuda.synchronize()
            print(json.dumps({'step':step,'loss':float(np.mean(step_losses)),'lossTerms':{k:float(v.detach()) for k,v in last_terms.items()},
                'lr':lr,'samples':data.samples,**data.unique_crops.report_fields('uniqueCrops'),'codecSamples':data.codec_samples,
                'gradientNorm':float(gradient_norm),'scaler':scaler.get_scale(),'updateSkipped':not did_update,
                'secondsPerStep':(time.perf_counter()-started)/step,'allocatedMiB':torch.cuda.memory_allocated()/1048576}),flush=True)
        if step%args.interval==0 or step==args.steps:
            master={k:v.detach().float().cpu().clone() for k,v in model.state_dict().items()}
            model.eval();deploy=model.deploy_copy().cuda().half().eval()
            with torch.inference_mode():
                scores=evaluator.evaluate({'candidate':lambda x:deploy(x.half()).float(),
                                           'adore-lite-v3':_LegacyAdapter(legacy)},device='cuda',perceptual=feature)
            summary=update_composite(scores['cases'])
            checkpoint={'model':master,'step':step,'config':config}
            atomic_torch_save(checkpoint,output/f'step-{step}-master-fp32.pt');atomic_torch_save(checkpoint,output/'last-master-fp32.pt')
            candidates={'mae':np.mean([v['candidate']['mae'] for v in scores['cases'].values()]),
                        'edge':np.mean([v['candidate']['edgeMae'] for v in scores['cases'].values()]),
                        'ssim':-np.mean([v['candidate']['ssim'] for v in scores['cases'].values()]),
                        'lpips':np.mean([v['candidate']['lpipsSqueeze'] for v in scores['cases'].values()]),
                        'composite':summary['composite']}
            for key,value in candidates.items():
                if key not in best or value<best[key]:best[key]=float(value);atomic_torch_save(checkpoint,output/f'best-{key}-master-fp32.pt')
            history.append({'step':step,'screening':scores,'summary':summary,'bestScores':dict(best)})
            report={**config,'stepsExecuted':step,'optimizerUpdates':optimizer_updates,'trainingSeconds':time.perf_counter()-started,
                    **data.unique_crops.report_fields(),'samples':data.samples,'codecSamples':data.codec_samples,
                    'hardSamples':data.hard_samples,'history':history,'promoted':False,
                    'realVideoValidation':'REAL VIDEO VALIDATION NOT EXECUTED'}
            atomic_json_write(report,output/'report.json')
            deploy.cpu().float();del deploy
            print(json.dumps({'validationStep':step,'composite':summary['composite'],'best':best,
                              'secondsPerStep':(time.perf_counter()-started)/step}),flush=True)
    print(json.dumps({'finished':True,'promoted':False,'output':str(output)}),flush=True)


if __name__=='__main__':main()
