"""Isolated torch.compile probe for the exact RealESR 2x path; never production."""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import torch
from model_registry import build_registry

p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--warmup',type=int,default=3); p.add_argument('--iterations',type=int,default=10)
a=p.parse_args();
if a.output.exists(): raise FileExistsError(a.output)
torch.set_num_threads(2); root=Path(__file__).resolve().parents[1]
report={'state':'running','scope':'isolated compile probe; not production; FP16 head2 exact 2x','torch':str(torch.__version__),'hip':torch.version.hip,'gpu':torch.cuda.get_device_name()}
a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2))
try:
  reg=build_registry(root); sr=reg.load('realesr-anime',device='cuda',precision='fp16'); m=sr.backend.model.eval()
  x=torch.rand(1,3,360,640,device='cuda',dtype=torch.float16)
  with torch.inference_mode():
    ref=m.forward_to(x,(720,1280)); torch.cuda.synchronize()
    def eager(): return m.forward_to(x,(720,1280))
    compiled=torch.compile(m,backend='inductor',mode='reduce-overhead',dynamic=False)
    def comp():
      y=compiled._run(x,2)
      return y if y.shape[-2:]==(720,1280) else torch.nn.functional.interpolate(y,size=(720,1280),mode='bilinear',align_corners=False)
    for _ in range(a.warmup): eager()
    torch.cuda.synchronize(); t=[]
    for _ in range(a.iterations):
      t0=time.perf_counter(); eager(); torch.cuda.synchronize(); t.append((time.perf_counter()-t0)*1000)
    for _ in range(a.warmup): comp()
    torch.cuda.synchronize(); c=[]
    for _ in range(a.iterations):
      t0=time.perf_counter(); comp(); torch.cuda.synchronize(); c.append((time.perf_counter()-t0)*1000)
    out=comp(); d=(ref.float()-out.float()).abs(); report.update({'eagerMedianMs':sorted(t)[len(t)//2],'compiledMedianMs':sorted(c)[len(c)//2],'eagerSamplesMs':t,'compiledSamplesMs':c,'parityMax':d.max().item(),'parityMean':d.mean().item(),'state':'complete'})
  a.output.write_text(json.dumps(report,indent=2))
except Exception as e:
  report.update({'state':'failed','error':str(e)}); a.output.write_text(json.dumps(report,indent=2)); raise
