"""Reproducible paired optimization evaluation; GPU operations are sequential."""
import argparse, gc, hashlib, json, os, sys, datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'native'))
import server
import torch,numpy as np
from PIL import Image
from model_registry import build_registry
from benchmark_timing import timed_iterations,fill_distinct_input

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def load(enabled):
    os.environ['ANIMESCALER_REALESR_OPTIMIZE']='1' if enabled else '0'
    return build_registry(ROOT).load('realesr-anime',device='cuda',precision='fp16',memory_format=torch.contiguous_format)

def measure_case(enabled,h,w,oh,ow,warmup,iterations):
    model=load(enabled)
    x=torch.empty((1,3,h,w),device='cuda',dtype=torch.float16)
    fill_distinct_input(x,0)
    graph=None
    if enabled and h*w<=model.metadata.capture_graph_max_pixels:
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):model.infer_to(x,(oh,ow))
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):captured=model.infer_to(x,(oh,ow))
    torch.cuda.reset_peak_memory_stats()
    timing=timed_iterations(graph.replay if graph else lambda:model.infer_to(x,(oh,ow)),torch.cuda.synchronize,
                            warmup,iterations,lambda i:fill_distinct_input(x,i))
    row={'enabled':enabled,'input':[w,h],'output':[ow,oh],**timing,
         'peakAllocatedMiB':torch.cuda.max_memory_allocated()/1048576,
         'peakReservedMiB':torch.cuda.max_memory_reserved()/1048576,'graph':graph is not None,
         'scope':'Completion-timed model + requested resize/composed head. Excludes copies, transport and browser.',
         'postResize':'FUSED INTO DERIVED HEAD' if enabled and (oh/h,ow/w) in ((2,2),(1.5,1.5)) else ('NOT APPLICABLE' if (oh,ow)==(h*4,w*4) else 'INCLUDED IN TOTAL'),
         'status':'PASS'}
    model.close()
    return row

def performance(args,report):
    report['cases']=[]
    cases=[(360,640,720,1280),(720,1280,1080,1920),(1080,1920,2160,3840),(360,640,1440,2560)]
    with torch.inference_mode():
        for h,w,oh,ow in cases:
            for enabled in (False,True):
                row=measure_case(enabled,h,w,oh,ow,args.warmup,args.iterations)
                report['cases'].append(row)
                print(json.dumps(row),flush=True)
                gc.collect();torch.cuda.empty_cache()
    report['status']='PASS'

def tensor_image(path):
    array=np.asarray(Image.open(path).convert('RGB')).copy()
    return torch.from_numpy(array).permute(2,0,1).unsqueeze(0).cuda().half()/255

def rgb(value):
    return value[0].permute(1,2,0).clamp(0,1).mul(255).round().byte().cpu().numpy()

def quality(args,report):
    before,after=load(False),load(True)
    report['sequences']=[]
    manifests=sorted((ROOT/'diagnostics/video-sequences/rezero-real-01').glob('clip-*/complete.json'))
    if len(manifests)!=3:raise ValueError('Expected the three existing real-video manifests')
    count=0;all_rows=[]
    with torch.inference_mode():
        for manifest in manifests:
            sequence=json.loads(manifest.read_text(encoding='utf-8'))
            if sequence['frameCount']!=48:raise ValueError('Expected 48 existing frames per clip')
            rows=[];temporal=[];previous=None
            for frame in sequence['frames']:
                lr=manifest.parent/frame['lr'];hr=manifest.parent/frame['hr']
                if sha(lr)!=frame['lrSha256'] or sha(hr)!=frame['hrSha256']:raise ValueError('Frame identity changed')
                x=tensor_image(lr)
                b=before.infer_to(x,(720,1280)).float()
                a=after.infer_to(x,(720,1280)).float()
                delta=a-b
                reference=tensor_image(hr).float()
                row={'index':frame['index'],'maxAbsError':delta.abs().max().item(),'meanAbsError':delta.abs().mean().item(),
                     'beforeReferenceMAE':(b.clamp(0,1)-reference).abs().mean().item(),
                     'afterReferenceMAE':(a.clamp(0,1)-reference).abs().mean().item(),
                     'quantizedMaxDifference':int(np.abs(rgb(a).astype(np.int16)-rgb(b).astype(np.int16)).max())}
                if previous is not None:
                    pb,pa,pr=previous
                    temporal.append({'before':((b-pb)-(reference-pr)).abs().mean().item(),
                                     'after':((a-pa)-(reference-pr)).abs().mean().item(),
                                     'optimizationDelta':((a-b)-(pa-pb)).abs().mean().item()})
                previous=(b,a,reference)
                rows.append(row);all_rows.append(row);count+=1
                if frame['index']==0:
                    for name,value in [('before',b),('after',a)]:
                        image=Image.fromarray(rgb(value));ww,hh=image.size
                        image.crop((ww//2-160,hh//2-160,ww//2+160,hh//2+160)).save(args.output.parent/f'{manifest.parent.name}-{name}.png')
            report['sequences'].append({'manifest':str(manifest.relative_to(ROOT)),'sha256':sha(manifest),
                                        'frames':rows,'temporal':temporal,
                                        'beforeTemporalMAE':float(np.mean([t['before'] for t in temporal])),
                                        'afterTemporalMAE':float(np.mean([t['after'] for t in temporal]))})
            print('quality clip complete: '+manifest.parent.name,flush=True)
        # Same actual scenes, native x4 and 720p->1080p head including all borders.
        report['otherHeads']=[]
        for manifest in manifests:
            sequence=json.loads(manifest.read_text(encoding='utf-8'))
            for frame in sequence['frames'][:3]:
                for key,size in [('lr',(1440,2560)),('hr',(1080,1920))]:
                    x=tensor_image(manifest.parent/frame[key]);b=before.infer_to(x,size);a=after.infer_to(x,size)
                    delta=(a.float()-b.float()).abs()
                    report['otherHeads'].append({'manifest':str(manifest.relative_to(ROOT)),'index':frame['index'],
                      'output':list(size[::-1]),'maxAbsError':delta.max().item(),'meanAbsError':delta.mean().item()})
        report['frameCount']=count
        report['maxAbsError']=max(r['maxAbsError'] for r in all_rows+report['otherHeads'])
        report['meanAbsError']=float(np.mean([r['meanAbsError'] for r in all_rows+report['otherHeads']]))
        report['quantizedMaxDifference']=max(r['quantizedMaxDifference'] for r in all_rows)
        report['tolerance']={'fp16MaxAbs':2/255,'fp16MeanAbs':0.0005,'scope':'Equivalence to existing FP16 path; not a new model-quality certification'}
        report['status']='PASS' if count==144 and report['maxAbsError']<=2/255 and report['meanAbsError']<=0.0005 else 'FAIL'
        before.close();after.close()

def renderer(args,report):
    from protocol import HEADER,MAGIC
    report['cases']=[]
    with torch.inference_mode():
        for enabled in (False,True):
            os.environ['ANIMESCALER_REALESR_OPTIMIZE']='1' if enabled else '0'
            r=server.Renderer()
            for h,w,oh,ow in [(360,640,720,1280),(720,1280,1080,1920),(1080,1920,2160,3840)]:
                packet=[None];index=[0]
                def prepare(i):
                    index[0]=i
                    pixels=np.random.default_rng(929+i).integers(0,256,(h,w,4),dtype=np.uint8)
                    packet[0]=HEADER.pack(MAGIC,i,w,h,ow,oh)+pixels.tobytes()
                def operation():
                    result,duration=r.render(packet[0],'realesr-anime','rgb',False)
                    if r.last_reused:raise AssertionError('Frame reuse in benchmark')
                    if len(result)!=HEADER.size+ow*oh*3:raise AssertionError('Wrong bridge raster')
                    if HEADER.unpack_from(result)[1]!=index[0]:raise AssertionError('Wrong frame id')
                    return result
                timing=timed_iterations(operation,torch.cuda.synchronize,5,30,prepare)
                row={'enabled':enabled,'input':[w,h],'output':[ow,oh],'graph':r.graph is not None,
                     'status':'PASS','scope':'Actual Renderer.render: copies, conversion, GPU, readback, RGB serialization; excludes socket/browser',**timing}
                report['cases'].append(row);print(json.dumps(row),flush=True)
            r.registry.unload('realesr-anime');r.registry.unload('adore')
            del r;gc.collect();torch.cuda.empty_cache()
    report['status']='PASS'

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['performance','quality','renderer'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--warmup',type=int,default=50)
    p.add_argument('--iterations',type=int,default=500);args=p.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    if not torch.version.hip or not torch.cuda.is_available():raise RuntimeError('ROCm GPU required')
    report={'stage':args.stage,'status':'INCOMPLETE','gpu':torch.cuda.get_device_name(),'hip':torch.version.hip,
            'torch':str(torch.__version__),'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'weightsSha256':sha(ROOT/'models/realesr-animevideov3.pth'),'precision':'FP16',
            'frameReuse':False,'graphPolicy':'Optimized only, <=640x360; target-size-aware graph',
            'scriptSha256':sha(Path(__file__)),'sourceSha256':{f:sha(ROOT/'native'/f) for f in ['realesr_optimized.py','model_registry.py','inference_backends.py','server.py']}}
    globals()[args.stage](args,report)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    if report['status']!='PASS':raise RuntimeError('Equivalence gate failed')
    print('COMPLETE '+args.stage,flush=True)

if __name__=='__main__':main()
