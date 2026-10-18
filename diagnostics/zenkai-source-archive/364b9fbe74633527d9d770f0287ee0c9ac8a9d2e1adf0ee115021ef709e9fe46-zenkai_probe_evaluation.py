"""Full fixed-codec evaluation of separate-family smoke probes, no registration."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE/'.tools'))
import torch
from export_zenkai import ROOT,configure_cache,load_checkpoint,sha256
from zenkai_ablation_residual import ResidualScaleProbe
from zenkai_ablation_skip import IdentitySkipProbe
from zenkai_evaluation import ZenkaiFixedEvaluator
from model_registry import build_registry
from train_zenkai import _LegacyAdapter


def load_reference_checkpoint(path,expected_step=1000):
    model,saved=load_checkpoint(path)
    config=saved['config']
    runs={1000:'zenkai-ablation-frequency-1k',3000:'zenkai-ablation-frequency-3k-continuation'}
    if expected_step not in runs or saved['step']!=expected_step or config.get('run')!=runs[expected_step] or config.get('depth')!=5:
        raise ValueError('Not the matched A-architecture frequency1k reference')
    return model,saved


def load_probe(path,expected_step=1000):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    config=saved.get('config',{})
    families={
        'ZenkaiResidualScaleProbe':(ResidualScaleProbe,'zenkai-ablation-output-residual-02-1k','outputResidualScale',.2),
        'ZenkaiIdentitySkipProbe':(IdentitySkipProbe,'zenkai-ablation-spab-identity-skip-1k','spabIdentitySkipGain',1.)}
    family=config.get('family')
    if family not in families:raise ValueError('Unreviewed probe family')
    factory,run,key,value=families[family]
    if expected_step==3000 and family=='ZenkaiIdentitySkipProbe':run='zenkai-ablation-spab-identity-skip-3k-continuation'
    elif expected_step!=1000:raise ValueError('Unreviewed probe stage')
    if family=='ZenkaiIdentitySkipProbe' and config.get('run')=='zenkai-ablation-spab-identity-skip-legacy-loss-1k':
        if expected_step!=1000 or config.get('loss')!='zenkai_legacy_v3' or config.get('comparisonRun')!='zenkai-ablation-spab-identity-skip-1k':
            raise ValueError('Unreviewed skip loss comparison')
        run='zenkai-ablation-spab-identity-skip-legacy-loss-1k'
    expected={'channels':32,'depth':5,'nativeScale':2,'run':run,key:value}
    if saved.get('step')!=expected_step or any(config.get(k)!=v for k,v in expected.items()):
        raise ValueError('Probe configuration differs from reviewed smoke')
    state=saved.get('model',{})
    if not state or any(not isinstance(v,torch.Tensor) or v.dtype!=torch.float32 or not torch.isfinite(v).all() for v in state.values()):
        raise ValueError('Probe master must be finite FP32')
    model=factory().eval();model.load_state_dict(state,strict=True)
    return model,{'config':config,'step':saved['step']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=('1k','3k','skip-loss-1k'),default='1k')
    args=parser.parse_args();output=args.output.resolve()
    if not output.is_relative_to(ROOT) or output.exists():parser.error('New workspace output required')
    configure_cache(ROOT);os.environ.setdefault('TORCH_HOME',str(ROOT/'diagnostics/native-cache/torch'))
    if not torch.cuda.is_available() or not torch.version.hip:raise RuntimeError('ROCm required')
    torch.set_num_threads(2)
    output.mkdir(parents=True)
    parity_status={'state':'validating','models':{},'releaseApproved':False}
    evaluator=ZenkaiFixedEvaluator(ROOT/'native/training-data/zenkai/validation-fixed.json',per_case=32)
    evaluator.attach_codec(ROOT/'native/training-data/zenkai/codec-validation-fixed/complete.json')
    models={};artifacts={};digests={};parity={}
    assignments=[('zenkai-frequency-1k','zenkai-ablation-frequency-1k'),
                     ('zenkai-output-gain-02-1k','zenkai-ablation-output-residual-02-1k'),
                     ('zenkai-identity-skip-1k','zenkai-ablation-spab-identity-skip-1k')]
    step=1000
    if args.stage=='3k':
        step=3000
        assignments=[('zenkai-frequency-3k','zenkai-ablation-frequency-3k-continuation'),
            ('zenkai-identity-skip-3k','zenkai-ablation-spab-identity-skip-3k-continuation')]
    if args.stage=='skip-loss-1k':
        assignments=[('skip-frequency-1k','zenkai-ablation-spab-identity-skip-1k'),
            ('skip-legacy-loss-1k','zenkai-ablation-spab-identity-skip-legacy-loss-1k')]
    for name,run in assignments:
        path=ROOT/'models/zenkai/experiments'/run/f'step-{step}-master-fp32.pt'
        model,saved=load_reference_checkpoint(path,step) if name.startswith('zenkai-frequency-') else load_probe(path,step)
        config=saved['config']
        if evaluator.sha256!=config['fixedValidationSha256']:raise ValueError('Validation manifest changed')
        digest=sha256(path);digests[path]=digest
        deploy=model.deploy_copy().cuda().float().eval();source=model.cuda().float().eval()
        from export_zenkai import parity_inputs,parity_metrics,require_parity,TOLERANCES
        cases=[]
        half=copy.deepcopy(deploy).half()
        with torch.inference_mode():
            for case,x in parity_inputs():
                x=x.cuda();expected=source(x)
                fusion=parity_metrics(expected,deploy(x));precision=parity_metrics(expected,half(x.half()))
                cases.append({'name':case,'fusionFp32':fusion,'deployFp16':precision})
                parity_status['models'][name]={'cases':cases,'parityApproved':False}
                try:
                    require_parity(fusion,**TOLERANCES['fp32']);require_parity(precision,**TOLERANCES['fp16'])
                except Exception as error:
                    parity_status.update(state='failed',error=str(error),failedModel=name,failedCase=case)
                    (output/'parity.json').write_text(json.dumps(parity_status,indent=2),encoding='utf-8')
                    raise
        parity[name]={'scope':'ROCm PyTorch, no ONNX export','cases':cases,'parityApproved':True}
        parity_status['models'][name]=parity[name]
        (output/'parity.json').write_text(json.dumps(parity_status,indent=2),encoding='utf-8')
        models[name]=lambda x,net=half:net(x.half()).float()
        artifacts[name]={'path':str(path),'sha256':digest,'step':saved['step'],'config':config}
        del source,deploy
    registry=build_registry(ROOT);historical=registry.load('adore-lite-v3',device='cuda',precision='fp16')
    if historical.artifact['sha256']!=config['legacyV3Sha256']:raise ValueError('Historical V3 changed')
    parity_status['state']='complete'
    (output/'parity.json').write_text(json.dumps(parity_status,indent=2),encoding='utf-8')
    models['adore-lite-v3']=_LegacyAdapter(historical);artifacts['adore-lite-v3']=historical.artifact
    import lpips
    feature=lpips.LPIPS(net='squeeze').cuda().eval().requires_grad_(False)
    try:
        report=evaluator.evaluate(models,device='cuda',perceptual=feature)
        if any(sha256(path)!=digest for path,digest in digests.items()):raise ValueError('Checkpoint changed')
        report.update(state='complete',models=artifacts,precision='fp16',backend='ROCm',gpu=torch.cuda.get_device_name(),
            torch=str(torch.__version__),hip=torch.version.hip,parity=parity,releaseApproved=False,
            realVideoValidation='NOT MEASURED',performance='NOT MEASURED',visualReview='NOT EXECUTED')
        (output/'quality.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    finally:registry.unload('adore-lite-v3')
    print(json.dumps({'state':'complete','output':str(output),'releaseApproved':False}))


if __name__=='__main__':main()
