import torch
from torch.nn import functional as F

def edge_loss(a,b):
    return ((a[...,1:]-a[...,:-1]-b[...,1:]+b[...,:-1]).abs().mean()+
            (a[...,1:,:]-a[...,:-1,:]-b[...,1:,:]+b[...,:-1,:]).abs().mean())/2

def laplacian(x):
    x=F.pad(x,(1,1,1,1),mode='replicate')
    return x[...,1:-1,:-2]+x[...,1:-1,2:]+x[...,:-2,1:-1]+x[...,2:,1:-1]-4*x[...,1:-1,1:-1]

def ssim(a,b):
    # Local 11x11 uniform-window SSIM; explicitly not MS-SSIM or LPIPS.
    pool=lambda x:F.avg_pool2d(x,11,stride=1)
    ma,mb=pool(a),pool(b)
    va=(pool(a*a)-ma*ma).clamp_min(0);vb=(pool(b*b)-mb*mb).clamp_min(0)
    cov=pool(a*b)-ma*mb
    return (((2*ma*mb+.01**2)*(2*cov+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))).mean()

def loss_terms(pred,target):
    pred=pred.float();target=target.float()
    return {'charbonnier':((pred-target).square()+1e-6).sqrt().mean(),
            'edge':edge_loss(pred,target),'laplacian':(laplacian(pred)-laplacian(target)).abs().mean(),
            'ssim':1-ssim(pred,target)}

LOSS_WEIGHTS={'charbonnier':1.,'edge':.2,'laplacian':.1,'ssim':.05}
def metrics(pred,target):
    pred=pred.float().clamp(0,1);target=target.float()
    mse=(pred-target).square().mean()
    return {'mae':float((pred-target).abs().mean()),'mse':float(mse),'psnr':float(-10*torch.log10(mse.clamp_min(1e-12))),
            'ssim':float(ssim(pred,target)),'edgeMae':float(edge_loss(pred,target)),
            'laplacianMae':float((laplacian(pred)-laplacian(target)).abs().mean())}
