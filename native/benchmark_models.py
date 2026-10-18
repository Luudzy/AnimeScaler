"""Real local bridge timing, including upload/download, excluding warmup."""
import json
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from server import Renderer, ROOT
from protocol import HEADER, MAGIC


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',default='adore-lite')
    parser.add_argument('--iterations',type=int,default=100)
    parser.add_argument('--warmup',type=int,default=15)
    args=parser.parse_args()
    if args.iterations<1 or args.warmup<1:parser.error('Positive counts required')
    renderer=Renderer()
    report={'device':renderer.info,'scope':'Renderer only, including GPU copies and encoding; no browser or WebSocket',
            'iterations':args.iterations,'warmup':args.warmup,'reuseAllowed':False,'cases':[]}
    for model in (renderer.info['models'] if args.model=='all' else [args.model]):
        for w,h,ow,oh in [(640,360,1280,720),(848,480,1696,960),(1280,720,2560,1440),(1280,720,1920,1080),(1920,1080,3840,2160)]:
            pixels=np.random.default_rng(42).integers(0,256,(h,w,4),dtype=np.uint8)
            times=[]
            for index in range(args.warmup+args.iterations):
                # Change RGB on every call, including the first timed call.
                pixels[0,0,:3]=[(index>>shift)&255 for shift in (0,8,16)]
                data=HEADER.pack(MAGIC,index,w,h,ow,oh)+pixels.tobytes()
                _,duration=renderer.render(data,model)
                if renderer.last_reused:raise RuntimeError('Benchmark reused a frame')
                if index>=args.warmup:times.append(duration)
            case={'model':model,'input':[w,h],'output':[ow,oh],'p50Ms':float(np.median(times)),
                  'p95Ms':float(np.percentile(times,95)),'p99Ms':float(np.percentile(times,99)),
                  'meanMs':float(np.mean(times)),'minMs':min(times),'maxMs':max(times),'fps':1000/float(np.mean(times))}
            report['cases'].append(case);print(json.dumps(case),flush=True)
            (ROOT/'diagnostics/native-model-benchmark.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
