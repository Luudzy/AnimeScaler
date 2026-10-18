"""Measured operator costs on the actual ROCm GPU, no architecture assumptions."""
import sys,argparse,json,datetime
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from server import ROOT
from benchmark_timing import timed_iterations
import torch
from torch import nn
from torch.nn import functional as F

def operations(x,channels):
    make=lambda module:module.to(device=x.device,dtype=x.dtype).eval()
    return {'conv3x3':make(nn.Conv2d(channels,channels,3,padding=1)),
            'conv1x1':make(nn.Conv2d(channels,channels,1)),
            'depthwise3x3':make(nn.Conv2d(channels,channels,3,padding=1,groups=channels)),
            'prelu':make(nn.PReLU(channels)), 'silu':nn.SiLU(),
            'add':lambda v:v+.1,'mul':lambda v:v*.9,'mul_add':lambda v:v*.9+.1,
            'nearest':lambda v:F.interpolate(v,scale_factor=2,mode='nearest'),
            'bilinear':lambda v:F.interpolate(v,scale_factor=2,mode='bilinear',align_corners=False),
            'pixelshuffle':lambda v:F.pixel_shuffle(v,2)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--warmup',type=int,default=50);p.add_argument('--iterations',type=int,default=500)
    p.add_argument('--channels',type=int,nargs='+',default=[32,48,64]);p.add_argument('--shape',action='append')
    p.add_argument('--output',type=Path,default=ROOT/'diagnostics/zenkai-gpu-ops.json');args=p.parse_args()
    if not torch.cuda.is_available() or not torch.version.hip:raise RuntimeError('ROCm GPU required')
    if any(c<4 or c%4 for c in args.channels):p.error('Channels must be positive multiples of four')
    shapes=[tuple(map(int,s.lower().split('x'))) for s in (args.shape or ['640x360','960x540','1280x720'])]
    torch.set_num_threads(2);torch.manual_seed(929)
    report={'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gpu':torch.cuda.get_device_name(),'hip':torch.version.hip,
            'scope':'Isolated eager operators on NCHW GPU inputs; synchronized completion vs CPU submit; not model FPS',
            'complete':False,'cases':[]}
    with torch.inference_mode():
        for dtype in [torch.float16,torch.float32]:
            for c in args.channels:
                for w,h in shapes:
                    x=torch.rand(1,c,h,w,device='cuda',dtype=dtype)
                    for name,op in operations(x,c).items():
                        torch.cuda.reset_peak_memory_stats()
                        timing=timed_iterations(lambda:op(x),torch.cuda.synchronize,args.warmup,args.iterations)
                        row={'operation':name,'channels':c,'input':[w,h],'precision':str(dtype),**timing,
                             'peakAllocatedMiB':torch.cuda.max_memory_allocated()/1048576}
                        report['cases'].append(row);args.output.write_text(json.dumps(report,indent=2))
                    print(json.dumps({'shape':[w,h],'channels':c,'precision':str(dtype),'completedCases':len(report['cases'])}),flush=True)
    report['complete']=True;args.output.write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
