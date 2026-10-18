"""Unique-frame renderer timing of a trained V3.1 candidate, no browser claim."""
import sys,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from server import ROOT,Renderer
from protocol import HEADER,MAGIC
from lite_v31 import AdoreLiteV31
import numpy as np
import torch

def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);args=p.parse_args()
    saved=torch.load(args.candidate/'best-lpipsSqueeze.pt',map_location='cpu',weights_only=True);config=saved['config']
    model=AdoreLiteV31(config['variant'],config['residual_scale']);model.load_state_dict(saved['model'])
    renderer=Renderer();renderer.models['v31']=model.cuda().half().eval();renderer.info['models'].append('v31')
    results=[]
    for w,h,ow,oh in [(640,360,1280,720),(848,480,1696,960),(1280,720,1920,1080),(1920,1080,3840,2160)]:
        pixels=np.random.default_rng(929).integers(0,256,(h,w,4),dtype=np.uint8);times=[]
        torch.cuda.reset_peak_memory_stats()
        for i in range(115):
            pixels[0,0,:3]=[i,0,0]
            _,duration=renderer.render(HEADER.pack(MAGIC,i,w,h,ow,oh)+pixels.tobytes(),'v31')
            if renderer.last_reused:raise RuntimeError('Benchmark reused frame')
            if i>=15:times.append(duration)
        results.append({'input':[w,h],'output':[ow,oh],'averageMs':float(np.mean(times)),'medianMs':float(np.median(times)),
                        'p95Ms':float(np.percentile(times,95)),'p99Ms':float(np.percentile(times,99)),
                        'fps':1000/float(np.mean(times)),'peakAllocatedMiB':torch.cuda.max_memory_allocated()/1048576})
    (args.candidate/'benchmark.json').write_text(json.dumps({'scope':'Renderer incl copies/encoding; not browser FPS; 15 warmups + 100 unique frames','checkpoint':saved['step'],'cases':results},indent=2))
if __name__=='__main__':main()
