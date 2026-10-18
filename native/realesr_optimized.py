"""Inference-only SRVGG optimization; no retraining, pruning or input reduction.

The x2 and x1.5 heads are linear compositions of the *original* final
convolution, PixelShuffle4 and bilinear resize (align_corners=False).
Real arithmetic is equivalent; FP16 rounding differs and must be measured.
The original x4 head/weights remain available for every other requested size.
"""
import torch
from torch import nn
from torch.nn import functional as F


class RealESRAnimeVideo(nn.Module):
    def __init__(self, model, max_pixels=1920*1080, tile=768):
        super().__init__()
        if tile <= 0 or tile % 2 or max_pixels < 1:
            raise ValueError('RealESR tiles must have a positive even core size')
        body = getattr(model, 'body', None)
        if getattr(model, 'upscale', None) != 4 or body is None or len(body) < 3:
            raise ValueError('Expected local SRVGG x4 architecture')
        for i, layer in enumerate(body):
            if i % 2 == 0:
                if (not isinstance(layer, nn.Conv2d) or layer.kernel_size != (3,3)
                        or layer.stride != (1,1) or layer.padding != (1,1)
                        or layer.dilation != (1,1) or layer.groups != 1 or layer.bias is None):
                    raise ValueError('Expected stride-one padded SRVGG convolutions')
            elif not isinstance(layer, nn.PReLU):
                raise ValueError('Expected pointwise SRVGG PReLU')
        last = body[-1]
        if last.out_channels != 48 or body[0].in_channels != 3:
            raise ValueError('Expected RGB PixelShuffle4 output')
        self.model = model.eval().float().requires_grad_(False)
        self.max_pixels, self.tile = max_pixels, tile
        # One input pixel of receptive-field radius per 3x3 convolution.
        # Round halo to even so stride-two x1.5 heads retain global alignment.
        radius = sum(isinstance(layer, nn.Conv2d) for layer in body)
        self.context = radius + radius % 2
        self.head2 = nn.Conv2d(last.in_channels, 12, 3, padding=1)
        self.head15 = nn.Conv2d(last.in_channels, 27, 4, stride=2, padding=1)
        self.base15 = nn.Conv2d(3, 27, 2, stride=2, groups=3, bias=False)
        self._compose_heads(last)
        self.requires_grad_(False)
        self.eval()

    def _compose_heads(self, last):
        with torch.no_grad():
            # Each x2 sample averages one 2x2 group of x4 shuffle phases.
            self.head2.weight.copy_(last.weight.reshape(3,2,2,2,2,last.in_channels,3,3)
                                    .mean(dim=(2,4)).reshape_as(self.head2.weight))
            self.head2.bias.copy_(last.bias.reshape(3,2,2,2,2).mean(dim=(2,4)).reshape(12))
            self.head15.weight.zero_()
            self.head15.bias.zero_()
            self.base15.weight.zero_()
            # Three requested pixels per two input pixels. The middle sample
            # crosses an LR-cell boundary; shift its convolution in the 4x4 kernel.
            positions = ((0,5/6), (3,1/2), (6,1/6))
            for c in range(3):
                for py,(y,fy) in enumerate(positions):
                    for px,(x,fx) in enumerate(positions):
                        channel = c*9 + py*3 + px
                        for dy,wy in ((0,1-fy),(1,fy)):
                            for dx,wx in ((0,1-fx),(1,fx)):
                                iy,ix = y+dy,x+dx
                                sy,sx = iy//4,ix//4
                                source = c*16 + (iy%4)*4 + ix%4
                                weight = wy*wx
                                self.head15.weight[channel,:,sy:sy+3,sx:sx+3].add_(last.weight[source],alpha=weight)
                                self.head15.bias[channel].add_(last.bias[source],alpha=weight)
                                self.base15.weight[channel,0,sy,sx].add_(weight)

    def _run(self, x, mode):
        out = x
        for layer in self.model.body[:-1]:
            out = layer(out)
        if mode == 2:
            out = self.head2(out)
            out.reshape(x.shape[0],3,4,*x.shape[-2:]).add_(x.unsqueeze(2))
            return F.pixel_shuffle(out,2)
        if mode == 1.5:
            out = self.head15(out)
            out.add_(self.base15(x))
            return F.pixel_shuffle(out,3)
        out = self.model.body[-1](out)
        # Add nearest-neighbor residual in packed space: no full x4 base tensor.
        out.reshape(x.shape[0],3,16,*x.shape[-2:]).add_(x.unsqueeze(2))
        return F.pixel_shuffle(out,4)

    def _tiled(self, x, mode):
        h,w = x.shape[-2:]
        if h*w <= self.max_pixels:
            return self._run(x,mode)
        s = mode
        result = x.new_empty((x.shape[0],3,int(h*s),int(w*s)))
        # Disjoint output cores + sufficient halo. No feathering, accumulation
        # buffers, weight maps or overlapping core inference.
        for y in range(0,h,self.tile):
            for v in range(0,w,self.tile):
                end_y,end_x = min(h,y+self.tile),min(w,v+self.tile)
                top,left = max(0,y-self.context),max(0,v-self.context)
                bottom,right = min(h,end_y+self.context),min(w,end_x+self.context)
                output = self._run(x[...,top:bottom,left:right],mode)
                core = output[...,int((y-top)*s):int((end_y-top)*s),
                              int((v-left)*s):int((end_x-left)*s)]
                result[...,int(y*s):int(end_y*s),int(v*s):int(end_x*s)].copy_(core)
        return result

    def forward_to(self, x, output_size):
        h,w = x.shape[-2:]
        size = tuple(output_size)
        mode = 2 if size == (h*2,w*2) else 4
        if h%2 == 0 and w%2 == 0 and size == (h*3//2,w*3//2):
            mode = 1.5
        output = self._tiled(x,mode)
        return output if output.shape[-2:] == size else F.interpolate(output,size=size,mode='bilinear',align_corners=False)

    def forward(self, x):
        return self._tiled(x,4)
