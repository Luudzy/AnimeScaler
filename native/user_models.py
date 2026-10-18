"""Explicit catalog of user-added checkpoints; safe weights-only loading."""
import copy
import torch
from torch import nn

USER_MODELS = {
    'hfa2k-cugan': {'file':'2xHFA2kReal-CUGAN.pth','architecture':'RealCUGAN','scale':2,'label':'HFA2K Real-CUGAN'},
    'realesr-anime': {'file':'realesr-animevideov3.pth','architecture':'Compact','scale':4,'label':'RealESR AnimeVideo v3'},
    'span-mssim': {'file':'4xmssim_span_pretrain.pth','architecture':'SPAN','scale':4,'label':'SPAN MS-SSIM'},
}

def fuse_span(module):
    """Materialize the existing eval equation once, in FP32, before FP16 conversion."""
    count=0
    for name,child in list(module.named_children()):
        if type(child).__name__=='Conv3XC':
            child.update_params()
            conv=copy.deepcopy(child.eval_conv)
            replacement=nn.Sequential(conv,nn.LeakyReLU(.05)) if child.has_relu else conv
            setattr(module,name,replacement);count+=1
        else:count+=fuse_span(child)
    return count

def load_user_model(root,name,optimize=True):
    from spandrel import ModelLoader
    spec=USER_MODELS[name]
    state=torch.load(root/'models'/spec['file'],map_location='cpu',weights_only=True)
    descriptor=ModelLoader().load_from_state_dict(state)
    if descriptor.architecture.id!=spec['architecture'] or descriptor.scale!=spec['scale']:
        raise ValueError('Checkpoint architecture/scale differs from catalog')
    model=descriptor.model.eval().float().requires_grad_(False)
    fused=fuse_span(model) if optimize and spec['architecture']=='SPAN' else 0
    return model,{'scale':descriptor.scale,'supportsHalf':descriptor.supports_half,'fusedConvolutions':fused}
