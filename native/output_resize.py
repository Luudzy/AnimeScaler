"""Explicit output filter policy; AA downsampling computes in FP32 on ROCm."""
from torch.nn import functional as F

def resize_output(value,size,mode='bilinear'):
    if tuple(value.shape[-2:])==tuple(size):return value
    if mode=='bicubic-aa':
        return F.interpolate(value.float(),size=size,mode='bicubic',align_corners=False,antialias=True)
    if mode=='area':return F.interpolate(value,size=size,mode='area')
    if mode not in ('bilinear','bicubic'):raise ValueError('Unknown output filter')
    return F.interpolate(value,size=size,mode=mode,align_corners=False)
