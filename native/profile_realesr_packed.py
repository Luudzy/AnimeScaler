"""Isolated ROCm parity and infer_to benchmark. Does not enable production kernels."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import torch
from torch.nn import functional as F
from benchmark_timing import timed_iterations, fill_distinct_input
from model_registry import build_registry

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
p.add_argument('--width', type=int, default=1280)
p.add_argument('--height', type=int, default=808)
p.add_argument('--ow', type=int, default=3424)
p.add_argument('--oh', type=int, default=2160)
p.add_argument('--warmup', type=int, default=5)
p.add_argument('--iterations', type=int, default=20)
p.add_argument('--autotune', action='store_true')
p.add_argument('--baseline-only', action='store_true')
a = p.parse_args()
if a.output.exists(): raise FileExistsError(a.output)
torch.set_num_threads(2)
torch.backends.cudnn.benchmark = a.autotune
torch.manual_seed(929)
report = {'state':'running', 'autotune':a.autotune, 'torch':str(torch.__version__),
          'hip':torch.version.hip, 'gpu':torch.cuda.get_device_name(),
          'scope':'infer_to only; no browser/socket; unchanged inputs and weights', 'parity':[]}
a.output.parent.mkdir(parents=True, exist_ok=True)
def save(): a.output.write_text(json.dumps(report, indent=2))
save()
try:
    with torch.inference_mode():
        if not a.baseline_only:
            from realesr_packed_resize import packed_resize
        for h,w,oh,ow in ([] if a.baseline_only else [(1,1,3,5),(7,9,17,23),(8,12,32,48),(32,40,48,60)]):
            x=torch.rand(1,3,h,w,device='cuda',dtype=torch.float16)
            packed=torch.randn(1,48,h,w,device='cuda',dtype=torch.float16)
            ref=F.interpolate(F.pixel_shuffle(packed+ x.repeat_interleave(16,dim=1),4),
                              size=(oh,ow),mode='bilinear',align_corners=False)
            actual=packed_resize(packed,x,(oh,ow))
            delta=(ref.float()-actual.float()).abs()
            row={'shape':[h,w,oh,ow],'max':delta.max().item(),'mean':delta.mean().item()}
            report['parity'].append(row); save()
            if row['max'] > .004 or row['mean'] > .0005: raise ValueError(row)
        root=Path(__file__).resolve().parents[1]
        registry=build_registry(root)
        sr=registry.load('realesr-anime',device='cuda',precision='fp16')
        m=sr.backend.model
        report['artifact']=sr.artifact
        x=torch.empty(1,3,a.height,a.width,device='cuda',dtype=torch.float16)
        size=(a.oh,a.ow)
        if a.baseline_only:
            report['shape']=[a.width,a.height,a.ow,a.oh]
            fill_distinct_input(x,0)
            output_bytes=sr.infer_to(x,size).cpu().numpy().tobytes()
            report['outputSha256']=hashlib.sha256(output_bytes).hexdigest()
            del output_bytes
            torch.cuda.reset_peak_memory_stats()
            report['currentInferTo']=timed_iterations(lambda:sr.infer_to(x,size),torch.cuda.synchronize,
                a.warmup,a.iterations,lambda i:fill_distinct_input(x,i))
            report['currentInferTo']['peakAllocatedMiB']=torch.cuda.max_memory_allocated()/1048576
            report['state']='complete';save();print(report['currentInferTo'],flush=True)
            sys.exit(0)
        def fused():
            out=x
            for layer in m.model.body: out=layer(out)
            return packed_resize(out,x,size)
        fill_distinct_input(x,0)
        expected=sr.infer_to(x,size)
        actual=fused()
        delta=(expected.float()-actual.float()).abs()
        report['modelParity']={'max':delta.max().item(),'mean':delta.mean().item(),
            'uint8Max':((expected.clamp(0,1)*255).round()-(actual.clamp(0,1)*255).round()).abs().max().item()}
        save()
        if report['modelParity']['max']>.004 or report['modelParity']['mean']>.0005:
            raise ValueError('Full-resolution parity failed')
        del expected,actual,delta
        report['shape']=[a.width,a.height,a.ow,a.oh]
        for name,op in [('currentInferTo',lambda:sr.infer_to(x,size)),('fusedInferTo',fused)]:
            torch.cuda.reset_peak_memory_stats()
            report[name]=timed_iterations(op,torch.cuda.synchronize,a.warmup,a.iterations,
                                         lambda i:fill_distinct_input(x,i))
            report[name]['peakAllocatedMiB']=torch.cuda.max_memory_allocated()/1048576
            save(); print(name,report[name],flush=True)
        report['state']='complete';save()
except Exception as error:
    report['state']='HIP_INFRA_FAILURE' if 'HIP' in str(error) else 'failed'
    report['error']=str(error);save();raise
