"""Paired real 1080p input -> 4K check, including the former tile boundaries."""
import argparse,hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'native'))
import server
import torch,numpy as np
from PIL import Image
from model_registry import build_registry

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
torch.set_num_threads(2)
input_path=HERE/'real1080-input-60.png'
image=np.asarray(Image.open(input_path).convert('RGB')).copy()
if image.shape!=(1080,1920,3):raise ValueError('Expected actual 1080p input')
x=torch.from_numpy(image).permute(2,0,1).unsqueeze(0).cuda().half()/255
results=[]
with torch.inference_mode():
    for enabled in (False,True):
        os.environ['ANIMESCALER_REALESR_OPTIMIZE']='1' if enabled else '0'
        model=build_registry(ROOT).load('realesr-anime',device='cuda',precision='fp16')
        result=model.infer_to(x,(2160,3840)).float().cpu()
        results.append(result)
        model.close()
    before,after=results;delta=(before-after).abs()
    b=(before.clamp(0,1)*255).round().byte()
    a=(after.clamp(0,1)*255).round().byte()
    quantized=(a.to(torch.int16)-b.to(torch.int16)).abs()
    report={'status':'PASS' if delta.max().item()<=2/255 and delta.mean().item()<=0.0005 else 'FAIL',
            'input':[1920,1080],'output':[3840,2160],'inputSha256':hashlib.sha256(input_path.read_bytes()).hexdigest(),
            'source':'User-supplied local 2160p video, CPU Lanczos 1080p decode at60seconds',
            'maxAbsError':delta.max().item(),'meanAbsError':delta.mean().item(),
            'maxQuantizedDifference':quantized.max().item(),'meanQuantizedDifference':quantized.float().mean().item(),
            'scope':'Existing feathered 512-tile FP16 path versus optimized whole-frame composed x2 head'}
    for name,value in [('before',b),('after',a)]:
        # Contains both an old horizontal and vertical core-overlap boundary.
        crop=value[0,:,760:1080,760:1080].permute(1,2,0).numpy()
        Image.fromarray(crop).save(HERE/f'fullhd-old-tile-boundary-{name}.png')
args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report),flush=True)
if report['status']!='PASS':raise RuntimeError('Full HD equivalence gate failed')
