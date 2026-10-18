"""Bounded local distillation prototype; holdout images never enter training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from server import ROOT  # Sets project-local MIOpen cache before importing torch.
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from lite_model import AdoreLite
from spandrel.architectures.RealCUGAN.__arch.upcunet_v3 import UpCunet2x_fast


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--images', nargs='+', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=600)
    args=parser.parse_args()
    if len(args.images)<4 or args.steps<1: parser.error('At least 4 images and a positive step count required')
    if not torch.version.hip or not torch.cuda.is_available(): raise RuntimeError('ROCm GPU required')
    digests=[hashlib.sha256(p.read_bytes()).hexdigest() for p in args.images]
    if len(set(digests)) != len(digests): raise ValueError('Duplicate images would contaminate the holdout')
    torch.set_num_threads(2); torch.manual_seed(42); random.seed(42)
    teacher=UpCunet2x_fast().eval().cuda().half()
    with np.load(ROOT/'models/adore-native.npz',allow_pickle=False) as archive:
        teacher.load_state_dict({key:torch.from_numpy(archive[key].copy()) for key in archive.files},strict=True)
    pairs=[]
    with torch.inference_mode():
        for path in args.images:
            image=Image.open(path).convert('RGB')
            width=640; height=max(40,int(width*image.height/image.width)//4*4)
            pixels=np.asarray(image.resize((width,height),Image.Resampling.LANCZOS),dtype=np.float32)/255
            x=torch.from_numpy(pixels.transpose(2,0,1).copy()).unsqueeze(0).cuda()
            y=(teacher(x.half()*.7+.15)-.15)/.7
            pairs.append((x.cpu().clone(),y.float().cpu().clone()))
    del teacher
    torch.cuda.empty_cache()
    # Clone outside inference mode: training must be allowed to save tensors.
    train=[(x.clone().cuda(),y.clone().cuda()) for x,y in pairs[:-1]]
    holdout=(pairs[-1][0].clone().cuda(),pairs[-1][1].clone().cuda())
    model=AdoreLite().cuda().train()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
    patch=48; batch=8
    started=time.perf_counter()
    for step in range(args.steps):
        xs=[]; ys=[]
        for _ in range(batch):
            x,y=random.choice(train)
            yy=random.randrange(0,x.shape[2]-patch+1); xx=random.randrange(0,x.shape[3]-patch+1)
            xs.append(x[:,:,yy:yy+patch,xx:xx+patch]); ys.append(y[:,:,yy*2:(yy+patch)*2,xx*2:(xx+patch)*2])
        x=torch.cat(xs); target=torch.cat(ys)
        if random.random()<.5: x=x.flip(-1); target=target.flip(-1)
        optimizer.zero_grad(set_to_none=True)
        actual=model(x)
        pixel=F.l1_loss(actual,target)
        edge=F.l1_loss(actual[:,:,:,1:]-actual[:,:,:,:-1],target[:,:,:,1:]-target[:,:,:,:-1])
        loss=pixel+.2*edge
        loss.backward(); optimizer.step()
        if step%100==0: print(f'Step {step}/{args.steps}, loss={float(loss.detach()):.6f}',flush=True)
    model=model.eval().half()
    with torch.inference_mode():
        x,reference=holdout
        result=model(x.half()).float()
        baseline=F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False)
        error=float((result-reference).abs().mean()); baseline_error=float((baseline-reference).abs().mean())
        if not bool(torch.isfinite(result).all()): raise RuntimeError('Nonfinite output; refusing to export')
        timings=[]
        for i in range(25):
            start=time.perf_counter(); model(x.half()); torch.cuda.synchronize()
            if i>=5: timings.append((time.perf_counter()-start)*1000)
    output=ROOT/'models/adore-lite-experimental.npz'
    np.savez(output,**{key:value.detach().cpu().float().numpy() for key,value in model.state_dict().items()})
    report={'experimental':True,'architecture':'AdoreLite16-v1','steps':args.steps,'trainImageSha256':digests[:-1],
      'holdoutImageSha256':digests[-1],'sourceCount':len(args.images),'trainingSeconds':time.perf_counter()-started,
      'weightsSha256':hashlib.sha256(output.read_bytes()).hexdigest(), 'holdoutMeanAbsoluteError':error,
      'bilinearMeanAbsoluteError':baseline_error,'improvesOverBilinear':error<baseline_error,
      'inputShape':list(x.shape),'gpuMedianMs':float(np.median(timings)),'gpuP95Ms':float(np.percentile(timings,95)),
      'limitations':'Three screenshots from one anime; one held-out screenshot. Not evidence of general quality, temporal stability or 30 FPS end-to-end.'}
    (ROOT/'models/adore-lite-experimental.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
