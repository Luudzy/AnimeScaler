"""Train against original-image targets; promote nothing without held-out evidence."""
import argparse
import hashlib
import io
import json
import random
import time
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from server import ROOT
from artifact_io import atomic_json_write,atomic_torch_save
import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.nn import functional as F
from lite_model import AdoreLite
from training_metrics import quality_report


def training_lr(step,steps,start,end):
    return end+(start-end)*.5*(1+math.cos(math.pi*min(1,step/max(1,steps-1))))


def pair(image, rng, augment=True, robust=False):
    size=96
    if isinstance(image,Path):
        with Image.open(image) as source:
            x=rng.randrange(source.width-size+1);y=rng.randrange(source.height-size+1)
            image=source.crop((x,y,x+size,y+size)).convert('RGB')
    x=rng.randrange(image.width-size+1);y=rng.randrange(image.height-size+1)
    target=image.crop((x,y,x+size,y+size))
    if augment and rng.random()<.5: target=target.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    low=target
    if augment and robust and rng.random()<.2:
        intermediate=rng.randrange(54,89)
        low=low.resize((intermediate,intermediate),Image.Resampling.BICUBIC)
    if augment and rng.random()<.3: low=low.filter(ImageFilter.GaussianBlur(rng.uniform(.2,.6)))
    low=low.resize((48,48),rng.choice([Image.Resampling.BICUBIC,Image.Resampling.LANCZOS]))
    if augment and rng.random()<.25:
        buffer=io.BytesIO();low.save(buffer,format='JPEG',quality=rng.randrange(75,96));buffer.seek(0)
        low=Image.open(buffer).convert('RGB')
    values=[np.array(value,dtype=np.float32)/255 for value in (low,target)]
    if augment and robust and rng.random()<.2:
        noise=np.random.default_rng(rng.randrange(2**32)).normal(0,rng.uniform(.2,1.5)/255,values[0].shape)
        values[0]=np.clip(values[0]+noise,0,1).astype(np.float32)
    return tuple(torch.from_numpy(value.transpose(2,0,1).copy()) for value in values)


def edges(value):
    return value[...,1:]-value[...,:-1],value[...,1:,:]-value[...,:-1,:]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=Path,default=ROOT/'native/training-data/v2')
    parser.add_argument('--steps',type=int,default=1800)
    parser.add_argument('--channels',type=int,choices=[16,24],default=16)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--learning-rate',type=float,default=2e-4)
    parser.add_argument('--end-learning-rate',type=float)
    parser.add_argument('--robust-degradation',action='store_true')
    parser.add_argument('--validation-interval',type=int,default=0)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.steps<1:parser.error('Positive step count required')
    if not math.isfinite(args.learning_rate) or args.learning_rate<=0:parser.error('Positive finite learning rate required')
    end_lr=args.learning_rate if args.end_learning_rate is None else args.end_learning_rate
    if not math.isfinite(end_lr) or not 0<end_lr<=args.learning_rate:parser.error('Invalid end learning rate')
    if args.validation_interval<0:parser.error('Invalid validation interval')
    if not torch.cuda.is_available() or not torch.version.hip:raise RuntimeError('ROCm required')
    torch.set_num_threads(2);torch.manual_seed(args.seed)
    rng=random.Random(args.seed)
    rows=[json.loads(line) for line in (args.dataset/'manifest.jsonl').read_text().splitlines()]
    groups={key:[] for key in ['train','validation','test']}
    seen=set()
    for row in rows:
        path=args.dataset/row['file'];digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=row['sha256'] or digest in seen:raise ValueError('Invalid or duplicate training image')
        seen.add(digest)
        with Image.open(path) as image:
            if min(image.size)>=96:groups[row['split']].append(path)
    if len(groups['train'])<32 or min(len(groups['validation']),len(groups['test']))<8:
        raise ValueError('Need >=32 training and >=8 images in each held-out split')
    baseline=AdoreLite().cuda().eval()
    with np.load(ROOT/'models/adore-lite-experimental.npz',allow_pickle=False) as weights:
        baseline.load_state_dict({key:torch.from_numpy(weights[key].copy()) for key in weights.files})
    model=AdoreLite(args.channels).cuda()
    if args.channels==16:model.load_state_dict(baseline.state_dict())
    optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate)
    best_state=None;best_score=float('inf');best_step=0;validation_curve=[]
    started=time.perf_counter()
    model.train()
    for step in range(args.steps):
        for group in optimizer.param_groups:group['lr']=training_lr(step,args.steps,args.learning_rate,end_lr)
        pairs=[pair(rng.choice(groups['train']),rng,robust=args.robust_degradation) for _ in range(8)]
        x=torch.stack([p[0] for p in pairs]).cuda();target=torch.stack([p[1] for p in pairs]).cuda()
        actual=model(x);ae=edges(actual);te=edges(target)
        loss=F.l1_loss(actual,target)+.15*(F.l1_loss(ae[0],te[0])+F.l1_loss(ae[1],te[1]))
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        if step%200==0:print(f'Lite{args.channels} {step}/{args.steps} loss={loss.item():.6f}',flush=True)
        if args.validation_interval and ((step+1)%args.validation_interval==0 or step+1==args.steps):
            model.eval();check_rng=random.Random(123);scores=[]
            with torch.inference_mode():
                for image in groups['validation']:
                    low,high=pair(image,check_rng,augment=False)
                    actual=model(low[None].cuda()).clamp(0,1);target=high[None].cuda()
                    a,b=edges(actual),edges(target)
                    scores.append(float(F.l1_loss(actual,target)+.15*(F.l1_loss(a[0],b[0])+F.l1_loss(a[1],b[1]))))
            score=float(np.mean(scores));validation_curve.append({'step':step+1,'score':score})
            if score<best_score:
                best_score=score;best_step=step+1
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            print(f'Validation {step+1}: {score:.6f}; best={best_step}',flush=True);model.train()
    if best_state is not None:model.load_state_dict(best_state)
    model.eval()
    metrics={}
    with torch.inference_mode():
        for split in ['validation','test']:
            scores={'candidate':[],'previous':[],'bilinear':[]}
            edge_scores={key:[] for key in scores}
            eval_rng=random.Random(123 if split=='validation' else 456)
            for image in groups[split]:
                for _ in range(4):
                    low,high=pair(image,eval_rng,augment=False)
                    x=low.unsqueeze(0).cuda();target=high.unsqueeze(0).cuda()
                    values={'candidate':model(x).clamp(0,1),'previous':baseline(x).clamp(0,1),
                            'bilinear':F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False)}
                    for key,value in values.items():
                        scores[key].append(float(F.l1_loss(value,target)))
                        a,b=edges(value),edges(target)
                        edge_scores[key].append(float((F.l1_loss(a[0],b[0])+F.l1_loss(a[1],b[1]))/2))
            metrics[split]={key:{'mae':float(np.mean(scores[key])),'edgeMae':float(np.mean(edge_scores[key]))} for key in scores}
    path=args.output or ROOT/f'models/adore-lite-v2-{args.channels}.npz'
    master={k:v.detach().float().cpu().clone() for k,v in model.state_dict().items()}
    atomic_torch_save(master,path.with_name(path.stem+'-master-fp32.pt'))
    # Keep a resumable FP32 artifact; benchmark a separate FP16 deployment.
    np.savez_compressed(path,**{k:v.numpy() for k,v in master.items()})
    np.savez_compressed(path.with_name(path.stem+'-deploy-fp16.npz'),**{k:v.half().numpy() for k,v in master.items()})
    model.half()
    timings=[]
    with torch.inference_mode():
        x=torch.rand(1,3,360,640,device='cuda',dtype=torch.float16)
        for _ in range(4):model(x)
        torch.cuda.synchronize()
        for _ in range(20):
            begin=time.perf_counter();model(x);torch.cuda.synchronize();timings.append((time.perf_counter()-begin)*1000)
    gate=quality_report(metrics)
    report={'architecture':f'AdoreLite{args.channels}-v1','weightsSha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'datasetSha256':hashlib.sha256((args.dataset/'manifest.jsonl').read_bytes()).hexdigest(),
            'splits':{k:len(v) for k,v in groups.items()},'steps':args.steps,'metrics':metrics,
            'seed':args.seed,'learningRate':args.learning_rate,'endLearningRate':end_lr,
            'robustDegradation':args.robust_degradation,'selectedStep':best_step or args.steps,'validationCurve':validation_curve,
            'inferenceP50Ms':float(np.median(timings)),'inferenceP95Ms':float(np.percentile(timings,95)),
            **gate,'trainingSeconds':time.perf_counter()-started,
            'evaluation':'Synthetic 2x original-image reference; no temporal video validation'}
    atomic_json_write(report,path.with_suffix('.json'))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
