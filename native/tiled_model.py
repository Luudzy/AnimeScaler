"""Bounded-memory inference with context and feathered overlap for large frames.

Global attention statistics may differ from whole-image inference. This is a
memory fallback, not an assertion of pixel equivalence for global-pooling nets.
"""
import torch
from torch import nn

class TiledModel(nn.Module):
    def __init__(self,model,scale,max_pixels=1280*720,tile=512,overlap=64,context=32):
        super().__init__();self.model=model;self.scale=scale;self.max_pixels=max_pixels
        self.tile=tile;self.overlap=overlap;self.context=context
        if tile<=overlap or min(tile,overlap,context)<0:raise ValueError('Invalid tile configuration')
    def forward(self,x):
        h,w=x.shape[-2:]
        if h*w<=self.max_pixels:return self.model(x)
        s=self.scale;t=self.tile;c=self.context
        ys=sorted({min(y,max(0,h-t)) for y in range(0,h,t-self.overlap)})
        xs=sorted({min(v,max(0,w-t)) for v in range(0,w,t-self.overlap)})
        result=x.new_zeros((x.shape[0],3,h*s,w*s));weights=x.new_zeros((1,1,h*s,w*s))
        for y in ys:
            for v in xs:
                y1=min(h,y+t);x1=min(w,v+t);top=max(0,y-c);left=max(0,v-c)
                bottom=min(h,y1+c);right=min(w,x1+c)
                tile=self.model(x[...,top:bottom,left:right])
                tile=tile[...,(y-top)*s:(y1-top)*s,(v-left)*s:(x1-left)*s]
                hy,wx=tile.shape[-2:];fy=x.new_ones(hy);fx=x.new_ones(wx)
                oy=min(self.overlap*s,hy//2);ox=min(self.overlap*s,wx//2)
                if oy:
                    ramp=torch.linspace(1/(oy+1),1,oy,device=x.device,dtype=x.dtype)
                    if y:fy[:oy]=ramp
                    if y1<h:fy[-oy:]=ramp.flip(0)
                if ox:
                    ramp=torch.linspace(1/(ox+1),1,ox,device=x.device,dtype=x.dtype)
                    if v:fx[:ox]=ramp
                    if x1<w:fx[-ox:]=ramp.flip(0)
                weight=(fy[:,None]*fx[None,:])[None,None]
                result[...,y*s:y1*s,v*s:x1*s].add_(tile*weight)
                weights[...,y*s:y1*s,v*s:x1*s].add_(weight)
        return result/weights.clamp_min(1e-6)
