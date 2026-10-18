"""Benchmark an exported Zenkai checkpoint with distinct frames and GPU barriers."""
import argparse, json, time
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from zenkai_realesr_lite import ZenkaiRealESRLite

def run(model, shape, dtype, warmup=50, iterations=500):
    x = torch.empty(shape, device='cuda', dtype=dtype)
    def prepare(i):
        g = torch.Generator(device='cuda').manual_seed(929 + i)
        x.copy_(torch.rand(shape, device='cuda', dtype=dtype, generator=g))
    def op():
        model(x)
    for i in range(warmup):
        prepare(i); op(); torch.cuda.synchronize()
    times=[]
    for i in range(iterations):
        prepare(i + warmup); torch.cuda.synchronize()
        t=time.perf_counter(); op(); torch.cuda.synchronize(); times.append((time.perf_counter()-t)*1000)
    return {'input':[shape[-1],shape[-2]],'warmup':warmup,'iterations':iterations,
            'meanMs':float(np.mean(times)),'medianMs':float(np.median(times)),
            'p95Ms':float(np.percentile(times,95)),'p99Ms':float(np.percentile(times,99)),
            'minMs':float(min(times)),'maxMs':float(max(times)),
            'fpsEquivalent':float(1000/np.mean(times)),
            'peakAllocatedMiB':float(torch.cuda.max_memory_allocated()/1048576)}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    payload=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    model=ZenkaiRealESRLite().cuda().eval(); model.load_state_dict(payload['model'],strict=True)
    report={'state':'complete','checkpoint':str(a.checkpoint.resolve()),'backend':'ROCm/HIP','cases':[]}
    for dtype,name in [(torch.float16,'fp16'),(torch.float32,'fp32')]:
        model.to(dtype); torch.cuda.reset_peak_memory_stats()
        for shape in [(1,3,360,640),(1,3,720,1280)]:
            row=run(model,shape,dtype); row['precision']=name; report['cases'].append(row); print(json.dumps(row),flush=True)
    a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')

if __name__=='__main__': main()
