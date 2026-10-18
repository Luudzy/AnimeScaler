"""ESPAN-inspired frequency supervision with explicit Zenkai choices.

Paper equations 7-9: https://openaccess.thecvf.com/content/CVPR2025W/NTIRE/
papers/Wang_Expanded_SPAN_for_Efficient_Super-Resolution_CVPRW_2025_paper.pdf
The paper states 5x5 Gaussian high-pass and block-wise DCT, but leaves sigma,
padding, block size and normalization unspecified. Here: sigma=1, replicate
padding, 8x8 orthonormal DCT-II, RGB, mean absolute coefficient error. These
are documented adaptation choices, not the authors' released training code.
The historical lite_losses.py presets remain available separately.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from lite_losses import edge_loss


PRESETS={
    'zenkai_baseline':{'charbonnier':1.},
    'zenkai_frequency':{'charbonnier':1.,'hf':1.,'dct':.001},
    'zenkai_frequency_edge':{'charbonnier':1.,'hf':1.,'dct':.001,'edge':.2},
    # AnimeVideo-style restoration recipe: keep Charbonnier dominant while
    # making lineart/high-frequency reconstruction visible without GAN artifacts.
    'zenkai_animevideo':{'charbonnier':1.,'hf':1.35,'dct':.0015,'edge':.3},
}


def dct_basis(size=8,device=None):
    position=torch.arange(size,device=device,dtype=torch.float32)
    frequency=position[:,None]
    basis=torch.cos(math.pi*(position+.5)*frequency/size)*math.sqrt(2/size)
    basis[0]*=1/math.sqrt(2)
    return basis


def gaussian_kernel(device=None):
    axis=torch.arange(-2,3,device=device,dtype=torch.float32)
    kernel=torch.exp(-axis.square()/2);kernel/=kernel.sum()
    return (kernel[:,None]*kernel[None,:])[None,None]


def high_frequency(image,kernel=None):
    with torch.autocast(device_type=image.device.type,enabled=False):
        image=image.float()
        if kernel is None:kernel=gaussian_kernel(image.device)
        channels=image.shape[1]
        smooth=F.conv2d(F.pad(image,(2,2,2,2),mode='replicate'),kernel.float().expand(channels,1,5,5),groups=channels)
        return image-smooth


def block_dct(image,basis=None):
    with torch.autocast(device_type=image.device.type,enabled=False):
        image=image.float()
        if basis is None:basis=dct_basis(device=image.device)
        size=basis.shape[0];height,width=image.shape[-2:]
        image=F.pad(image,(0,(-width)%size,0,(-height)%size),mode='replicate')
        batch,channels,height,width=image.shape
        blocks=image.reshape(batch,channels,height//size,size,width//size,size).permute(0,1,2,4,3,5)
        return basis.float()@blocks@basis.float().T


class ZenkaiLoss(nn.Module):
    def __init__(self,preset='zenkai_frequency'):
        super().__init__()
        if preset not in PRESETS:raise ValueError(f'Unknown loss preset: {preset}')
        self.preset=preset;self.weights=PRESETS[preset].copy()
        self.register_buffer('basis',dct_basis())
        self.register_buffer('gaussian',gaussian_kernel())

    def forward(self,pred,target):
        if pred.shape!=target.shape:raise ValueError('Loss input shape mismatch')
        with torch.autocast(device_type=pred.device.type,enabled=False):
            pred=pred.float();target=target.float()
            terms={'charbonnier':((pred-target).square()+1e-6).sqrt().mean()}
            if 'hf' in self.weights:
                terms['hf']=(high_frequency(pred,self.gaussian)-high_frequency(target,self.gaussian)).abs().mean()
            if 'dct' in self.weights:
                terms['dct']=(block_dct(pred,self.basis)-block_dct(target,self.basis)).abs().mean()
            if 'edge' in self.weights:terms['edge']=edge_loss(pred,target)
            return sum(self.weights[name]*value for name,value in terms.items()),terms
