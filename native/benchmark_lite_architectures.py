import sys,time,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from server import ROOT
import torch,numpy as np
from lite_model import AdoreLite
from lite_v3 import AdoreLiteV3
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(2);torch.manual_seed(929)
results=[]
with torch.inference_mode():
    for name,model in [('v2-16',AdoreLite()),('v3-16',AdoreLiteV3(16)),('v3-24',AdoreLiteV3(24)),('v3-32',AdoreLiteV3(32))]:
        model=model.cuda().half().eval();params=sum(p.numel() for p in model.parameters())
        for w,h in [(640,360),(1280,720)]:
            x=torch.rand(1,3,h,w,device='cuda',dtype=torch.float16);times=[];torch.cuda.reset_peak_memory_stats()
            for i in range(115):
                x[...,0,0]=i/115;torch.cuda.synchronize();start=time.perf_counter();out=model(x);torch.cuda.synchronize()
                if i>=15:times.append((time.perf_counter()-start)*1000)
            row={'model':name,'parameters':params,'input':[w,h],'output':[w*2,h*2],
                 'medianMs':float(np.median(times)),'meanMs':float(np.mean(times)),'p95Ms':float(np.percentile(times,95)),
                 'p99Ms':float(np.percentile(times,99)),'peakAllocatedMiB':torch.cuda.max_memory_allocated()/1048576}
            results.append(row);print(json.dumps(row),flush=True)
        del model,out,x;torch.cuda.empty_cache()
(ROOT/'diagnostics/lite-architecture-benchmark.json').write_text(json.dumps({'scope':'Eager FP16 model only, random weights, 15 warmup + 100 unique calls. No renderer/cache/browser.','gpu':torch.cuda.get_device_name(),'cases':results},indent=2))
