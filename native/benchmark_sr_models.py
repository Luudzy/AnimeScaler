"""Unified registry benchmark. Completion timing, unique input, explicit scope."""
import sys,json,csv,argparse,subprocess,hashlib,datetime,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from server import ROOT
from model_registry import build_registry
from benchmark_timing import timed_iterations,fill_distinct_input
from baseline_reports import merge_baseline_report
import torch,numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',action='append');p.add_argument('--backend',choices=['torch','onnx'],default='torch')
    p.add_argument('--warmup',type=int,default=50);p.add_argument('--iterations',type=int,default=500)
    p.add_argument('--input',type=Path);p.add_argument('--output',type=Path,default=ROOT/'diagnostics/zenkai-baselines')
    p.add_argument('--zenkai-export',action='append',help='Explicit experimental registry entry: zenkai-ID=EXPORT_DIRECTORY')
    p.add_argument('--retrained-control',type=Path)
    p.add_argument('--probe-checkpoint',type=Path)
    p.add_argument('--probe-step',type=int,choices=(1000,3000,5000),default=3000)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = os.environ.get('ANIMESCALER_MIOPEN_BENCHMARK', '0') == '1'
    registry=build_registry(ROOT)
    from zenkai_artifacts import add_zenkai_exports
    add_zenkai_exports(registry,args.zenkai_export)
    if args.retrained_control:
        from zenkai_artifacts import register_control_checkpoint
        register_control_checkpoint(registry,args.retrained_control)
    if args.probe_checkpoint:
        from zenkai_artifacts import register_probe_checkpoint
        register_probe_checkpoint(registry,args.probe_checkpoint,args.probe_step)
    names=args.model or ['adore-lite-v3','realesr-anime','hfa2k-cugan']
    cases=json.loads(args.input.read_text()) if args.input else [{'width':640,'height':360,'outputWidth':1280,'outputHeight':720},{'width':1280,'height':720,'outputWidth':1920,'outputHeight':1080}]
    if args.backend=='torch' and (not torch.cuda.is_available() or not torch.version.hip):raise RuntimeError('This ROCm benchmark requires HIP GPU')
    report={'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'backend':args.backend,
            'gpu':torch.cuda.get_device_name() if args.backend=='torch' else None,'precision':'fp16' if args.backend=='torch' else 'fp32',
            'driver':'NOT AVAILABLE','hip':torch.version.hip,'torch':str(torch.__version__),
            'gitCommit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'scope':'Model plus requested output resize; completion barrier; no renderer copies, encoding, socket or browser. ONNX CPU is separately identified.',
            'provider':'ROCm' if args.backend=='torch' else 'CPUExecutionProvider',
            'inputProtocol':'v2: distinct full random frames; seed929+iteration; preparation excluded; no frame reuse',
            'executionPath':'infer_to (same output dispatch as renderer)',
            'autotune':torch.backends.cudnn.benchmark,
            'componentProtocol':'Independent native infer and post-resize reference passes; these are not components of optimized infer_to and must not be summed or subtracted',
            'cases':[],'complete':False}
    for name in names:
        meta=registry.get(name)
        memory_format = torch.channels_last if os.environ.get('ANIMESCALER_CHANNELS_LAST', '0') == '1' and args.backend == 'torch' else None
        model=registry.load(name,args.backend,device='cuda' if args.backend=='torch' else 'cpu',precision='fp16' if args.backend=='torch' else 'fp32',memory_format=memory_format)
        for case in cases:
            w,h,ow,oh=(case[k] for k in ['width','height','outputWidth','outputHeight'])
            x=np.random.default_rng(929).random((1,3,h,w),dtype=np.float32)
            if args.backend=='torch':
                x=torch.from_numpy(x).cuda().half()
                if memory_format is not None:x=x.contiguous(memory_format=memory_format)
                torch.cuda.reset_peak_memory_stats()
            def prepare(i):
                if args.backend=='torch':fill_distinct_input(x,i)
                else:x[:]=np.random.default_rng(929+i).random(x.shape,dtype=np.float32)
            def operation():
                return model.infer_to(x,(oh,ow))
            timing=timed_iterations(operation,model.backend.synchronize,args.warmup,args.iterations,prepare)
            components={'modelInference':'NOT MEASURED','postResize':'NOT MEASURED','totalMeasured':dict(timing)}
            if args.backend=='torch':
                components['modelInference']=timed_iterations(lambda:model.infer(x),model.backend.synchronize,
                    args.warmup,args.iterations,prepare)
                if (h*meta.native_scale,w*meta.native_scale)==(oh,ow):
                    components['postResize']={'status':'NOT APPLICABLE','reason':'Native output equals requested dimensions'}
                else:
                    resize_input=[None]
                    def prepare_resize(i):
                        prepare(i);resize_input[0]=model.infer(x);model.backend.synchronize()
                    components['postResize']=timed_iterations(lambda:torch.nn.functional.interpolate(resize_input[0],
                        size=(oh,ow),mode='bilinear',align_corners=False),model.backend.synchronize,
                        args.warmup,args.iterations,prepare_resize)
            row={'model':meta.id,'nativeScale':meta.native_scale,'neuralOutput':[w*meta.native_scale,h*meta.native_scale],**case,
                 'sha256':hashlib.sha256(Path(meta.weights_path).read_bytes()).hexdigest(),
                 'artifactPath':model.artifact['path'],'artifactSha256':model.artifact['sha256'],'artifactKind':model.artifact['kind'],
                 'parameters':sum(v.numel() for v in model.backend.model.parameters()) if args.backend=='torch' else None,
                 'peakAllocatedMiB':torch.cuda.max_memory_allocated()/1048576 if args.backend=='torch' else None,
                 'components':components,**timing}
            report['cases'].append(row);(args.output/'benchmark_results.json').write_text(json.dumps(report,indent=2));print(json.dumps(row),flush=True)
        registry.unload(name)
    report['complete']=True
    (args.output/'benchmark_results.json').write_text(json.dumps(report,indent=2))
    with (args.output/'benchmark_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(report['cases'][0]));writer.writeheader();writer.writerows(report['cases'])
    lines=['# Unified baseline performance','','All values MEASURED. '+report['scope'],'','| Model | Input | Output | Median ms | P95 ms |','|---|---|---|---:|---:|']
    lines.extend(f"| {r['model']} | {r['width']}x{r['height']} | {r['outputWidth']}x{r['outputHeight']} | {r['medianMs']:.2f} | {r['p95Ms']:.2f} |" for r in report['cases'])
    (args.output/'benchmark_summary.md').write_text('\n'.join(lines)+'\n')
    baseline_path=args.output/'baseline_report.json'
    previous=json.loads(baseline_path.read_text(encoding='utf-8')) if baseline_path.exists() else {}
    baseline_path.write_text(json.dumps(merge_baseline_report(previous,report),indent=2),encoding='utf-8')
if __name__=='__main__':main()
