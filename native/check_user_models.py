"""FP16 correctness and measured unique-frame performance on the local GPU."""
import json,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from server import ROOT,Renderer
from user_models import USER_MODELS,load_user_model
from protocol import HEADER,MAGIC
import torch,numpy as np

def main():
    torch.set_num_threads(2);torch.manual_seed(929)
    report={}
    renderer=Renderer()
    for name,spec in USER_MODELS.items():
        fp32,metadata=load_user_model(ROOT,name);fp32=fp32.cuda()
        x=torch.rand(1,3,48,64,device='cuda')
        with torch.inference_mode():reference=fp32(x).float();fp16=fp32.half();actual=fp16(x.half()).float()
        delta=(reference-actual).abs()
        if not torch.isfinite(actual).all() or float(delta.mean())>.005 or float(delta.max())>.05:raise ValueError(f'FP16 parity failed: {name}')
        result={**metadata,'fp16Mae':float(delta.mean()),'fp16MaxError':float(delta.max()),'cases':[]}
        del fp32,fp16,reference,actual,x,delta
        for w,h,ow,oh in [(640,360,1280,720),(1280,720,1920,1080),(1920,1080,3840,2160)]:
            pixels=np.random.default_rng(929).integers(0,256,(h,w,4),dtype=np.uint8);times=[]
            torch.cuda.reset_peak_memory_stats()
            for i in range(25):
                pixels[0,0,:3]=[i,0,0]
                body,duration=renderer.render(HEADER.pack(MAGIC,i,w,h,ow,oh)+pixels.tobytes(),name,'rgb')
                assert len(body)==24+ow*oh*3 and not renderer.last_reused
                if i>=5:times.append(duration)
            result['cases'].append({'input':[w,h],'output':[ow,oh],'meanMs':float(np.mean(times)),
                                    'medianMs':float(np.median(times)),'p95Ms':float(np.percentile(times,95)),
                                    'peakAllocatedMiB':torch.cuda.max_memory_allocated()/1048576})
            report[name]=result
            (ROOT/'diagnostics/user-model-gpu.json').write_text(json.dumps({'scope':'ROCm FP16; 5 warmups+20 unique frames; renderer, not browser FPS','models':report},indent=2))
            print(json.dumps({'model':name,**result['cases'][-1]}),flush=True)
        report[name]=result
        (ROOT/'diagnostics/user-model-gpu.json').write_text(json.dumps({'scope':'ROCm FP16; 5 warmups+20 unique frames; renderer, not browser FPS','models':report},indent=2))
        # Do not retain every large graph/model while moving to another model.
        renderer.graph=None;renderer.shape=None
    print('All user models passed',flush=True)
if __name__=='__main__':main()
