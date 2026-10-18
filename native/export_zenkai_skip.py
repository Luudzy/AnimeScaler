"""Explicit experimental IdentitySkip export; never registers or promotes a model."""
import argparse
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import torch
from export_zenkai import ROOT, TOLERANCES, sha256, parity_inputs, parity_metrics, require_parity, configure_cache
from zenkai_ablation_skip import IdentitySkipProbe
from zenkai_model import GRepConv
from zenkai_probe_evaluation import load_probe

RUNS=('zenkai-ablation-spab-identity-skip-3k-continuation',
      'zenkai-ablation-skip-factor-codec50-3k-continuation','zenkai-lite-v1-c-codec50-5k')
FAMILY='ZenkaiIdentitySkipProbe'


def load_candidate(path):
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('Checkpoint outside workspace')
    raw=torch.load(path,map_location='cpu',weights_only=True)
    step=5000 if raw.get('config',{}).get('run')=='zenkai-lite-v1-c-codec50-5k' else 3000
    model,saved=load_probe(path,step)
    if saved['config']['run'] not in RUNS or model.config['family']!=FAMILY:
        raise ValueError('Unreviewed identity skip candidate')
    return model,saved


def load_deploy(folder,report):
    path=Path(folder)/'deploy-fp32.pt'
    if report['family']!=FAMILY or sha256(path)!=report['artifacts'][path.name]['sha256']:
        raise ValueError('Wrong family or mutated deployment artifact')
    saved=torch.load(path,map_location='cpu',weights_only=True)
    model=IdentitySkipProbe().deploy_copy().eval()
    if saved['sourceSha256']!=report['checkpointSha256'] or saved['config']!=model.config:
        raise ValueError('Deployment configuration or source hash changed')
    state=saved['model']
    if any(v.dtype!=torch.float32 or not torch.isfinite(v).all() for v in state.values()):
        raise ValueError('Deployment weights must be finite FP32')
    model.load_state_dict(state,strict=True)
    return model


def export_candidate(checkpoint,output):
    checkpoint=Path(checkpoint).resolve();output=Path(output).resolve()
    if not output.is_relative_to(ROOT/'models/zenkai') or output.exists():
        raise ValueError('New experimental export inside models/zenkai required')
    pin=sha256(checkpoint);source,saved=load_candidate(checkpoint)
    master={k:v.clone() for k,v in source.state_dict().items()}
    deploy=source.deploy_copy().eval();half=copy.deepcopy(deploy).half().eval()
    if any(isinstance(m,GRepConv) for m in deploy.modules()):raise ValueError('Incomplete fusion')
    report={'schemaVersion':1,'family':FAMILY,'id':saved['config']['run']+'-experimental',
            'status':'experimental','state':'validating','nativeScale':2,'step':saved['step'],
            'checkpoint':str(checkpoint),'checkpointSha256':pin,'training':saved['config'],
            'architecture':deploy.config,'trainingParameters':sum(p.numel() for p in source.parameters()),
            'deployParameters':sum(p.numel() for p in deploy.parameters()),'masterPrecision':'fp32',
            'deployPrecision':'fp16','tolerances':TOLERANCES,'cases':[],'artifacts':{},
            'releaseApproved':False,'promoted':False,'qualityGate':'NOT EVALUATED BY EXPORT',
            'validationDevice':'cpu','torch':str(torch.__version__),'onnxGpuValidation':'NOT MEASURED',
            'scope':'Synthetic numeric export parity only; quality, video and latency recorded separately'}
    output.mkdir(parents=True)
    try:
        inputs=list(parity_inputs())
        with torch.inference_mode():
            for name,x in inputs:
                expected=source(x);fusion=parity_metrics(expected,deploy(x));precision=parity_metrics(expected,half(x.half()))
                require_parity(fusion,**TOLERANCES['fp32']);require_parity(precision,**TOLERANCES['fp16'])
                report['cases'].append({'name':name,'shape':list(x.shape),
                    'inputSha256':__import__('hashlib').sha256(x.numpy().tobytes()).hexdigest(),
                    'fusionFp32':fusion,'deployFp16':precision})
        torch.save({'model':deploy.state_dict(),'config':deploy.config,'sourceSha256':pin},output/'deploy-fp32.pt')
        np.savez(output/'deploy-fp16.npz',**{k:v.detach().numpy() for k,v in half.state_dict().items()})
        import onnx
        import onnxruntime as ort
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        for precision,model in (('fp32',deploy),('fp16',half)):
            dtype=torch.float32 if precision=='fp32' else torch.float16
            path=output/f'dynamic-{precision}.onnx'
            torch.onnx.export(model,torch.zeros(1,3,24,32,dtype=dtype),str(path),
                input_names=['input'],output_names=['output'],opset_version=17,dynamo=False,
                dynamic_axes={'input':{2:'height',3:'width'},'output':{2:'out_height',3:'out_width'}})
            onnx.checker.check_model(str(path))
            session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
            with torch.inference_mode():
                for row,(_,x) in zip(report['cases'],inputs):
                    score=parity_metrics(source(x),session.run(None,{'input':x.to(dtype).numpy()})[0])
                    require_parity(score,**TOLERANCES[precision]);row['onnx'+precision.upper()]=score
            del session
        if sha256(checkpoint)!=pin or any(not torch.equal(v,source.state_dict()[k]) for k,v in master.items()):
            raise ValueError('FP32 master changed during export')
        for path in output.iterdir():
            if path.is_file():report['artifacts'][path.name]={'sha256':sha256(path),'bytes':path.stat().st_size}
        # Reload the saved deployment, rather than validating only the in-memory copy.
        restored=load_deploy(output,report)
        with torch.inference_mode():
            for row,(_,x) in zip(report['cases'],inputs):
                score=parity_metrics(source(x),restored(x));require_parity(score,**TOLERANCES['fp32'])
                row['reloadedDeployFp32']=score
        report.update(state='complete',parityApproved=True,onnxProvider='CPUExecutionProvider',
                      onnx=str(onnx.__version__),onnxruntime=str(ort.__version__))
    except Exception as error:
        report.update(state='failed',parityApproved=False,error=str(error));raise
    finally:
        (output/'export-report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def validate_runtime(folder,output):
    configure_cache();folder=Path(folder).resolve();output=Path(output).resolve()
    if not folder.is_relative_to(ROOT/'models/zenkai') or not output.is_relative_to(ROOT) or output.exists():
        raise ValueError('Existing workspace export and new workspace report required')
    if not torch.version.hip or not torch.cuda.is_available():raise RuntimeError('ROCm required')
    exported=json.loads((folder/'export-report.json').read_text())
    if exported['state']!='complete' or not exported['parityApproved']:raise ValueError('CPU export incomplete')
    for name,artifact in exported['artifacts'].items():
        if sha256(folder/name)!=artifact['sha256']:raise ValueError('Export artifact changed: '+name)
    if sha256(exported['checkpoint'])!=exported['checkpointSha256']:raise ValueError('Master changed')
    source,_=load_candidate(exported['checkpoint']);source=source.cuda()
    deploy=load_deploy(folder,exported).cuda();half=copy.deepcopy(deploy).half()
    report={'state':'validating','family':FAMILY,'checkpointSha256':exported['checkpointSha256'],
            'exportReportSha256':sha256(folder/'export-report.json'),'device':'cuda','hip':torch.version.hip,
            'torch':str(torch.__version__),'gpu':torch.cuda.get_device_name(),'cases':[],
            'releaseApproved':False,'onnxGpuValidation':'NOT MEASURED','scope':'Reloaded PyTorch deploy ROCm parity'}
    try:
        with torch.inference_mode():
            for name,x in parity_inputs():
                x=x.cuda();expected=source(x)
                fusion=parity_metrics(expected,deploy(x));precision=parity_metrics(expected,half(x.half()))
                require_parity(fusion,**TOLERANCES['fp32']);require_parity(precision,**TOLERANCES['fp16'])
                report['cases'].append({'name':name,'fusionFp32':fusion,'deployFp16':precision})
        report.update(state='complete',parityApproved=True)
    except Exception as error:
        report.update(state='failed',parityApproved=False,error=str(error));raise
    finally:
        output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--checkpoint',type=Path);g.add_argument('--validate-runtime',type=Path)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();torch.set_num_threads(2)
    r=validate_runtime(args.validate_runtime,args.output) if args.validate_runtime else export_candidate(args.checkpoint,args.output)
    print(json.dumps({'state':r['state'],'output':str(args.output),'promoted':False}))


if __name__=='__main__':main()
