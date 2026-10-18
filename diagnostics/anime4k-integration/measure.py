"""Paired experiments on frozen local frames. One GPU process, no frame reuse."""
import argparse,datetime,gc,hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'native'))
import server
import numpy as np,torch
from PIL import Image,ImageDraw
from torch.nn import functional as F
from model_registry import build_registry
from benchmark_timing import timed_iterations,fill_distinct_input
from evaluate_real_video import verified_pairs
from lite_losses import metrics

IDS=['realesr-anime','anime4k-s','anime4k-m','realesr-anime4k-s-360']
CASES=[(360,640,720,1280),(720,1280,1080,1920),(1080,1920,2160,3840)]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(name,dtype='fp16',device='cuda'):
    return build_registry(ROOT).load(name,device=device,precision=dtype,memory_format=torch.contiguous_format)
def image(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert('RGB')).copy()).permute(2,0,1)[None].cuda().half()/255
def rgb(x):return x[0].permute(1,2,0).clamp(0,1).mul(255).round().byte().cpu().numpy()
def montage(path,values):
    crop=(440,140,840,540)
    canvas=Image.new('RGB',(400*len(values),425),'#161616');draw=ImageDraw.Draw(canvas)
    for i,(name,v) in enumerate(values):
        draw.text((i*400+8,5),name,fill='white')
        canvas.paste(Image.fromarray(rgb(v)).crop(crop),(i*400,25))
    canvas.save(path)

def parity(args,report):
    from anime4k_model import Anime4K
    report['cases']=[]
    for variant in ['S','M']:
        path=ROOT/f'models/anime4k/Anime4K_Upscale_CNN_x2_{variant}.glsl'
        cpu=Anime4K.from_file(path);gpu=Anime4K.from_file(path).cuda().half()
        for h,w,oh,ow in [(17,23,34,46),(32,48,48,72),(32,48,128,192)]:
            x=torch.rand((1,3,h,w),generator=torch.Generator().manual_seed(929))
            with torch.inference_mode():a=cpu.forward_to(x,(oh,ow));b=gpu.forward_to(x.cuda().half(),(oh,ow)).cpu().float()
            error=(a-b).abs();row={'variant':variant,'input':[w,h],'output':[ow,oh],
                'maxAbsError':error.max().item(),'meanAbsError':error.mean().item()}
            report['cases'].append(row)
        del cpu,gpu;gc.collect();torch.cuda.empty_cache()
    report['tolerance']={'maxAbs':2/255,'meanAbs':.0005}
    report['mpvGpuParity']='NOT MEASURED; CPU equation oracle and ROCm precision comparison only'
    report['status']='PASS' if all(r['maxAbsError']<=2/255 and r['meanAbsError']<=.0005 for r in report['cases']) else 'FAIL'

def quality(args,report):
    models={name:load(name) for name in IDS}
    manifests=sorted((ROOT/'diagnostics/video-sequences/rezero-real-01').glob('clip-*/complete.json'))
    if len(manifests)!=3:raise ValueError('Expected exactly three frozen clips')
    report['sequences']=[];report['frameCount']=0
    with torch.inference_mode():
        for manifest in manifests:
            _,pairs=verified_pairs(manifest)
            if len(pairs)!=48:raise ValueError('Expected same 48 frames per clip')
            all_rows={name:[] for name in ['bilinear']+IDS};previous={};old_reference=None
            for index,pair in enumerate(pairs):
                # verified_pairs preserves presentation order and checks SHA256.
                x=image(pair['lr']);ref=image(pair['hr']).float()
                outputs={'bilinear':F.interpolate(x,size=(720,1280),mode='bilinear',align_corners=False)}
                outputs.update({name:m.infer_to(x,(720,1280)) for name,m in models.items()})
                for name,out in outputs.items():
                    out=out.float().clamp(0,1)
                    row={'index':index,**metrics(out,ref)}
                    if old_reference is not None:row['temporalMae']=((out-previous[name])-(ref-old_reference)).abs().mean().item()
                    all_rows[name].append(row);previous[name]=out
                old_reference=ref
                if index in (0,24):montage(args.output.parent/f'{manifest.parent.name}-frame{index}.png',[('reference',ref)]+list(outputs.items()))
                report['frameCount']+=1
            averages={name:{key:float(np.mean([r[key] for r in rows if key in r])) for key in ['mae','mse','psnr','ssim','edgeMae','laplacianMae','temporalMae']} for name,rows in all_rows.items()}
            report['sequences'].append({'manifest':str(manifest.relative_to(ROOT)),'sha256':sha(manifest),'summary':averages,'frames':all_rows})
            print(json.dumps({'clip':manifest.parent.name,'summary':averages}),flush=True)
        report['higherResolutionDifferences']=[]
        for manifest in manifests:
            _,pairs=verified_pairs(manifest)
            x=image(pairs[24]['hr']);base=models['realesr-anime'].infer_to(x,(1080,1920))
            values=[('RealESR',base)]
            for name in IDS[1:]:
                out=models[name].infer_to(x,(1080,1920));delta=(out.float()-base.float()).abs()
                report['higherResolutionDifferences'].append({'clip':manifest.parent.name,'model':name,
                    'input':[1280,720],'output':[1920,1080],'maxAbsDifference':delta.max().item(),
                    'meanAbsDifference':delta.mean().item(),'scope':'Difference to RealESR, not ground-truth quality'})
                values.append((name,out))
            # Correct full-size side-by-side images; no implicit resize in crops.
            montage(args.output.parent/f'{manifest.parent.name}-1080p-model-difference.png',values)
    report['combo360Note']='At 360->720 the combo is exactly the existing RealESR x2 path; Anime4K runs only for larger requested output.'
    report['temporalScope']='Reference-compensated adjacent-frame delta MAE, no optical flow or human motion/flicker certification'
    report['status']='PASS' if report['frameCount']==144 else 'FAIL'

def performance(args,report):
    report['cases']=[]
    with torch.inference_mode():
        for h,w,oh,ow in CASES:
            for name in IDS:
                m=load(name);x=torch.empty((1,3,h,w),device='cuda',dtype=torch.float16)
                fill_distinct_input(x,0);graph=None
                if m.metadata.capture_graph_max_pixels>=h*w:
                    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):m.infer_to(x,(oh,ow))
                    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,stream=stream):captured=m.infer_to(x,(oh,ow))
                torch.cuda.reset_peak_memory_stats()
                timing=timed_iterations(graph.replay if graph else lambda:m.infer_to(x,(oh,ow)),torch.cuda.synchronize,50,500,lambda i:fill_distinct_input(x,i))
                row={'model':name,'input':[w,h],'output':[ow,oh],'status':'PASS','graph':graph is not None,**timing,
                    'peakAllocatedMiB':torch.cuda.max_memory_allocated()/1048576,
                    'peakReservedMiB':torch.cuda.max_memory_reserved()/1048576,
                    'scope':'Model inference + requested resize, excludes transfers/socket/player',
                    'modelInferenceOnly':'NOT MEASURED separately','postResizeOnly':'NOT MEASURED separately'}
                report['cases'].append(row);print(json.dumps(row),flush=True)
                m.close();del m,x,graph;gc.collect();torch.cuda.empty_cache()
    report['status']='PASS'

def renderer(args,report):
    from protocol import HEADER,MAGIC
    r=server.Renderer();report['cases']=[]
    with torch.inference_mode():
        for name in IDS:
            for h,w,oh,ow in CASES:
                packet=[None];index=[0]
                def prepare(i):
                    index[0]=i
                    pixels=np.random.default_rng(929+i).integers(0,256,(h,w,4),dtype=np.uint8)
                    packet[0]=HEADER.pack(MAGIC,i,w,h,ow,oh)+pixels.tobytes()
                def operation():
                    body,ms=r.render(packet[0],name,'rgb',False)
                    if r.last_reused or HEADER.unpack_from(body)[1]!=index[0] or len(body)!=HEADER.size+ow*oh*3:
                        raise AssertionError('Actual bridge invalid output/reuse')
                timing=timed_iterations(operation,torch.cuda.synchronize,5,30,prepare)
                row={'model':name,'input':[w,h],'output':[ow,oh],'status':'PASS',**timing,
                    'graph':r.graph is not None,'scope':'Actual Renderer copies + compute + RGB readback/serialization; excludes socket/player'}
                report['cases'].append(row);print(json.dumps(row),flush=True)
    report['status']='PASS'

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['parity','quality','performance','renderer']);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    if not torch.version.hip or not torch.cuda.is_available():raise RuntimeError('ROCm GPU required')
    report={'stage':args.stage,'status':'INCOMPLETE','gpu':torch.cuda.get_device_name(),'hip':torch.version.hip,
        'torch':str(torch.__version__),'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'precision':'FP16','frameReuse':False,'scriptSha256':sha(Path(__file__)),
        'sources':{f:sha(ROOT/'native'/f) for f in ['anime4k_model.py','anime4k_registry.py','realesr_optimized.py','model_registry.py','server.py']},
        'artifacts':{n:json.loads((ROOT/f'models/anime4k/{n}.json').read_text()) for n in IDS[1:]}}
    globals()[args.stage](args,report)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    if report['status']!='PASS':raise RuntimeError('Gate failed')
    print('COMPLETE '+args.stage,flush=True)
if __name__=='__main__':main()
