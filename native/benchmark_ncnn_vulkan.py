"""Persistent NCNN Vulkan benchmark; same dimensions as the ROCm probe."""
import argparse, hashlib, json, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from ncnn_vulkan_backend import NcnnVulkanBackend

p=argparse.ArgumentParser()
p.add_argument('--model-dir',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--width',type=int,default=640)
p.add_argument('--height',type=int,default=360)
p.add_argument('--warmup',type=int,default=5)
p.add_argument('--iterations',type=int,default=20)
a=p.parse_args()
if a.output.exists(): raise FileExistsError(a.output)
a.output.parent.mkdir(parents=True,exist_ok=True)
report={'state':'running','model':'realesr-animevideov3-x2','width':a.width,'height':a.height,'warmup':a.warmup,'iterations':a.iterations,'scope':'persistent NCNN Vulkan; input conversion and GPU completion included; PNG/socket/browser excluded'}
backend=NcnnVulkanBackend(a.model_dir)
try:
  frames=[np.random.default_rng(929+i).integers(0,256,(a.height,a.width,3),dtype=np.uint8).tobytes() for i in range(a.warmup+a.iterations)]
  for frame in frames[:a.warmup]: backend.infer_bytes(frame,a.width,a.height,2)
  samples=[]
  for frame in frames[a.warmup:]:
    t=time.perf_counter(); out=backend.infer_bytes(frame,a.width,a.height,2); samples.append((time.perf_counter()-t)*1000)
  ordered=sorted(samples)
  report.update({'medianMs':ordered[len(ordered)//2],'meanMs':sum(samples)/len(samples),'p95Ms':ordered[max(0,int(np.ceil(len(ordered)*.95))-1)],'minMs':min(samples),'maxMs':max(samples),'fps':1000/(sum(samples)/len(samples)),'outputShape':list(out.shape),'outputSha256':hashlib.sha256(out.tobytes()).hexdigest(),'state':'complete'})
finally:
  backend.close()
a.output.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
