"""Explicit loss experiments; detail uses standard Gaussian MS-SSIM."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'.tools'))
import torch
from pytorch_msssim import ms_ssim
from lite_losses import loss_terms,laplacian

PRESETS={'baseline':{'charbonnier':1.,'edge':.2,'laplacian':.1,'ssim':.05},
         'detail':{'charbonnier':1.,'edge':.3,'laplacian':.2,'ms_ssim':.05},
         'perceptual':{'charbonnier':1.,'edge':.3,'laplacian':.15,'ms_ssim':.05,'perceptual':.01}}

def reconstruction_loss(pred,target,preset='detail',edge_weight=0.,perceptual=None):
    pred=pred.float();target=target.float()
    terms=loss_terms(pred,target)
    if edge_weight:
        mask=laplacian(target).abs().mean(1,keepdim=True)
        mask=mask/(mask.amax(dim=(-2,-1),keepdim=True)+1e-6)
        terms['charbonnier']=(((pred-target).square()+1e-6).sqrt()*(1+edge_weight*mask)).mean()
    weights=PRESETS[preset]
    if 'ms_ssim' in weights:terms['ms_ssim']=1-ms_ssim(pred.clamp(0,1),target,data_range=1,size_average=True)
    if 'perceptual' in weights:
        if perceptual is None:raise ValueError('Perceptual preset requires frozen feature network')
        terms['perceptual']=perceptual(pred.clamp(0,1)*2-1,target*2-1).mean()
    return sum(weights[k]*terms[k] for k in weights),{k:float(terms[k].detach()) for k in weights}
