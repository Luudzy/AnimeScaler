"""Zenkai x2 adaptation of the verified ESPAN deployment graph.

The pure-PyTorch reference keeps the paper submission's 9x9 stem, attention,
feature taps, 1x1 concatenation fusion, and PixelShuffle head. Zenkai changes
the output scale to x2, uses a bilinear residual output, and exposes depth five
or six (full SPAB blocks followed by the reference two-convolution tail).
Training-only GRep branches follow the verified paper figure; unspecified
initialization scaling, bias and padding are stated here as Zenkai choices.
See docs/zenkai-espan-research.md and native/vendor/espan/.
"""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F


class GRepConv(nn.Module):
    """Six-branch 3x3-equivalent layer; convert using to_deploy()."""
    def __init__(self,in_channels,out_channels,bias=True):
        super().__init__()
        if not bias:raise ValueError('GRep adaptation currently requires branch biases')
        self.in_channels=in_channels;self.out_channels=out_channels
        self.branch_scale=1/math.sqrt(6)
        self.direct3=nn.Conv2d(in_channels,out_channels,3,padding=1,bias=True)
        self.direct1=nn.Conv2d(in_channels,out_channels,1,bias=True)
        self.seq13=nn.ModuleList(nn.Sequential(
            nn.Conv2d(in_channels,in_channels,1,bias=False),
            nn.Conv2d(in_channels,out_channels,3,padding=1,bias=True)) for _ in range(4))

    def forward(self,value):
        result=self.direct3(value)+self.direct1(value)
        for branch in self.seq13:result=result+branch(value)
        return result*self.branch_scale

    @staticmethod
    def _fuse_pair(branch):
        pointwise,spatial=branch[0],branch[1]
        kernel=torch.einsum('omuv,mi->oiuv',spatial.weight,pointwise.weight[:,:,0,0])
        return kernel,spatial.bias

    def to_deploy(self):
        branches=[(self.direct3.weight,self.direct3.bias)]
        pointwise=self.direct1.weight.new_zeros(self.out_channels,self.in_channels,3,3)
        pointwise[:,:,1,1]=self.direct1.weight[:,:,0,0]
        branches.append((pointwise,self.direct1.bias))
        branches.extend(self._fuse_pair(branch) for branch in self.seq13)
        kernel=sum(item[0] for item in branches)*self.branch_scale
        bias=sum(item[1] for item in branches)*self.branch_scale
        result=nn.Conv2d(self.in_channels,self.out_channels,3,padding=1,bias=True,
                         device=kernel.device,dtype=kernel.dtype)
        with torch.no_grad():result.weight.copy_(kernel);result.bias.copy_(bias)
        return result


def _deploy_copy(module):
    for name,child in list(module.named_children()):
        if isinstance(child,GRepConv):setattr(module,name,child.to_deploy())
        else:_deploy_copy(child)
    return module


class SPAB(nn.Module):
    def __init__(self,channels,convolutions=3):
        super().__init__()
        self.convolutions=nn.ModuleList(GRepConv(channels,channels) for _ in range(convolutions))
        self.activation=nn.SiLU()

    def forward(self,value):
        result=value
        for index,convolution in enumerate(self.convolutions):
            result=convolution(result)
            if index+1<len(self.convolutions):result=self.activation(result)
        return (result+value)*(torch.sigmoid(result)-.5)


class ZenkaiLite(nn.Module):
    def __init__(self,channels=32,depth=5,scale=2):
        super().__init__()
        if channels<4 or channels%4:raise ValueError('channels must be a positive multiple of four')
        if depth not in (5,6):raise ValueError('depth must be 5 or 6')
        if scale!=2:raise ValueError('Zenkai Lite V1 supports native x2 only')
        self.config={'family':'Zenkai','id':'zenkai-lite-v1-experimental','architecture':'ESPAN-inspired GRep SPAB x2',
                     'channels':channels,'depth':depth,'scale':scale,'legacy':False,'deploy':False,
                     'stemKernel':9,'tailConvolutions':2,'branchScale':'1/sqrt(6)',
                     'output':'bilinear x2 plus zero-initialized PixelShuffle residual'}
        self.stem=nn.Conv2d(3,channels,9,padding=4,bias=True)
        self.blocks=nn.ModuleList(SPAB(channels,3) for _ in range(depth-1))
        self.tail=SPAB(channels,2)
        self.post=GRepConv(channels,channels)
        self.concat_fuse=nn.Conv2d(channels*4,channels,1,bias=True)
        self.upsample=GRepConv(channels,12)
        nn.init.zeros_(self.upsample.direct3.weight)
        nn.init.zeros_(self.upsample.direct3.bias)
        nn.init.zeros_(self.upsample.direct1.weight)
        nn.init.zeros_(self.upsample.direct1.bias)
        for branch in self.upsample.seq13:
            nn.init.zeros_(branch[1].weight);nn.init.zeros_(branch[1].bias)

    def forward(self,value):
        first=self.stem(value);current=first;taps=[]
        for block in self.blocks:
            current=block(current);taps.append(current)
        current=self.tail(current)
        last=self.post(current)
        fused=self.concat_fuse(torch.cat((first,last,taps[0],current),dim=1))
        residual=F.pixel_shuffle(self.upsample(fused),2)
        base=F.interpolate(value,scale_factor=2,mode='bilinear',align_corners=False)
        return base+residual

    def deploy_copy(self):
        result=_deploy_copy(copy.deepcopy(self))
        result.config={**self.config,'deploy':True}
        return result


class _ESPANBlock(nn.Module):
    def __init__(self,channels,convolutions):
        super().__init__();self.layers=nn.ModuleList(nn.Conv2d(channels,channels,3,padding=1) for _ in range(convolutions));self.activation=nn.SiLU()
    def forward(self,value):
        result=value
        for i,layer in enumerate(self.layers):
            result=layer(result)
            if i+1<len(self.layers):result=self.activation(result)
        return (result+value)*(torch.sigmoid(result)-.5)


class ESPANDeploy(nn.Module):
    """CPU-safe PyTorch reference of vendored official submission (native x4)."""
    def __init__(self,channels=32):
        super().__init__()
        self.stem=nn.Conv2d(3,channels,9,padding=4)
        self.blocks=nn.ModuleList(_ESPANBlock(channels,3) for _ in range(5))
        self.tail=_ESPANBlock(channels,2);self.post=nn.Conv2d(channels,channels,3,padding=1)
        self.fuse=nn.Conv2d(channels*4,channels,1)
        self.head=nn.Sequential(nn.Conv2d(channels,3*16,3,padding=1),nn.PixelShuffle(4))
    def forward(self,value):
        first=self.stem(value);current=first;taps=[]
        for block in self.blocks:current=block(current);taps.append(current)
        current=self.tail(current);last=self.post(current)
        return self.head(self.fuse(torch.cat((first,last,taps[0],current),dim=1)))
