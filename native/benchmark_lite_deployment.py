"""Apples-to-apples trained model and renderer timing with unique RGB frames."""
import sys,json,time,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from server import Renderer,ROOT
from protocol import HEADER,MAGIC
from lite_model import AdoreLite
from lite_v3 import AdoreLiteV3
from output_resize import resize_output
import torch,numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);args=p.parse_args()
    renderer=Renderer();report={'scope':'Native renderer including GPU copies/encoding, not browser FPS; 15 warmup +100 unique frames','gpu':renderer.info,'cases':[]}
    for name,file in [('lite-v1','adore-lite-v1.npz')]:
        with np.load(ROOT/'models'/file,allow_pickle=False) as w:
            model=AdoreLite(w['body.0.weight'].shape[0]);model.load_state_dict({k:torch.from_numpy(w[k].copy()) for k in w.files})
        renderer.models[name]=model.cuda().half().eval();renderer.info['models'].append(name)
    saved=torch.load(args.candidate/'best_composite-master-fp32.pt',map_location='cpu',weights_only=True);c=saved['config']
    model=AdoreLiteV3(c['channels'],c['blocks'],c['residualScale']);model.load_state_dict(saved['model'])
    renderer.models['lite-v3']=model.cuda().half().eval();renderer.info['models'].append('lite-v3')
    for name in ['lite-v1','adore-lite','lite-v3']:
        for w,h,ow,oh in [(640,360,1280,720),(848,480,1696,960),(1280,720,2560,1440),(1280,720,1920,1080),(1920,1080,3840,2160)]:
            frame=np.random.default_rng(929).integers(0,256,(h,w,4),dtype=np.uint8);times=[]
            torch.cuda.reset_peak_memory_stats()
            for i in range(115):
                frame[0,0,:3]=[i,0,0];data=HEADER.pack(MAGIC,i,w,h,ow,oh)+frame.tobytes()
                _,duration=renderer.render(data,name)
                if renderer.last_reused:raise RuntimeError('Unexpected reuse')
                if i>=15:times.append(duration)
            row={'model':name,'input':[w,h],'output':[ow,oh],'meanMs':float(np.mean(times)),'medianMs':float(np.median(times)),
                 'p95Ms':float(np.percentile(times,95)),'p99Ms':float(np.percentile(times,99)),'minMs':min(times),'maxMs':max(times),
                 'fps':1000/float(np.mean(times)),'peakAllocatedMiB':torch.cuda.max_memory_allocated()/1048576}
            report['cases'].append(row);print(json.dumps(row),flush=True)
    with torch.inference_mode():
        x=torch.rand(1,3,1440,2560,device='cuda',dtype=torch.float16)
        report['downsample1440to1080']={}
        for mode in ['bilinear','bicubic','bicubic-aa','area']:
            times=[]
            for i in range(115):
                x[...,0,0]=i/115;torch.cuda.synchronize();start=time.perf_counter();y=resize_output(x,(1080,1920),mode);torch.cuda.synchronize()
                if i>=15:times.append((time.perf_counter()-start)*1000)
            report['downsample1440to1080'][mode]={'medianMs':float(np.median(times)),'p95Ms':float(np.percentile(times,95))}
    (args.candidate/'deployment-benchmark.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
