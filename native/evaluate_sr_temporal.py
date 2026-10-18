"""Generic known-reference temporal screening, explicitly separate from footage."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from lite_evaluation import load_patches, tensor


def temporal_delta_mae(output, previous_output, target, previous_target):
    if any(value.shape != target.shape for value in (output,previous_output,previous_target)):
        raise ValueError('Temporal shape mismatch')
    value = ((output-previous_output)-(target-previous_target)).abs().mean()
    if not torch.isfinite(value):
        raise ValueError('Non-finite temporal output')
    return float(value)


def evaluate_synthetic_temporal(models, patches, device='cuda', dtype=torch.float32, sequences=20, frames=12):
    if not len(patches) or sequences<1 or frames<2:
        raise ValueError('Temporal evaluation requires images and at least two frames')
    indices = np.linspace(0,len(patches)-1,min(sequences,len(patches)),dtype=int)
    scores = {name:[] for name in ['bilinear','bicubic',*models]}
    with torch.inference_mode():
        for index in indices:
            image = Image.fromarray(patches[index])
            if min(image.size) < 224 or frames > 48:
                raise ValueError('Pan must stay inside the 224px reference image')
            previous = None
            for frame in range(frames):
                high = image.transform((192,192),Image.Transform.AFFINE,
                                       (1,0,8+frame*.5,0,1,8+frame*.25),Image.Resampling.BICUBIC)
                low = high.resize((96,96),Image.Resampling.LANCZOS)
                target,x = tensor(high,device),tensor(low,device)
                outputs = {'bilinear':F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False),
                           'bicubic':tensor(low.resize(high.size,Image.Resampling.BICUBIC),device)}
                outputs.update({name:model(x.to(dtype)).float().clamp(0,1) for name,model in models.items()})
                if previous is not None:
                    old_target,old_outputs = previous
                    for name,value in outputs.items():
                        scores[name].append(temporal_delta_mae(value,old_outputs[name],target,old_target))
                previous = target,outputs
    return {'scope':'Synthetic subpixel pans from validation stills; not real footage or flicker certification.',
            'sequences':len(indices),'framesPerSequence':frames,
            'validationIndices':[int(i) for i in indices],
            'temporalDeltaMae':{name:float(np.mean(values)) for name,values in scores.items()},
            'realVideoValidation':'NOT EXECUTED'}


def main():
    from server import ROOT
    from model_registry import build_registry
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',action='append',required=True)
    parser.add_argument('--dataset',type=Path,default=ROOT/'native/training-data/v3-hfa2k')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--zenkai-export',action='append')
    parser.add_argument('--fixed-manifest',type=Path,help='Verified Zenkai validation crops instead of legacy random crops')
    args=parser.parse_args()
    torch.set_num_threads(2)
    registry=build_registry(ROOT)
    from zenkai_artifacts import add_zenkai_exports
    add_zenkai_exports(registry,args.zenkai_export)
    models={}
    artifacts={}
    for name in args.model:
        source=registry.load(name,'torch',device='cuda',precision='fp16')
        def infer(x,source=source):
            output=source.infer(x)
            return F.interpolate(output,size=(x.shape[-2]*2,x.shape[-1]*2),mode='bilinear',align_corners=False) if source.metadata.native_scale!=2 else output
        models[name]=infer
        artifacts[name]=source.artifact
    if args.fixed_manifest:
        from zenkai_evaluation import ZenkaiFixedEvaluator
        evaluator=ZenkaiFixedEvaluator(args.fixed_manifest,per_case=20)
        records=evaluator.samples['clean']
        patches=[np.array(evaluator.render_sample(row)[1]) for row in records]
    else:patches=load_patches(size=224,dataset=args.dataset)
    report=evaluate_synthetic_temporal(models,patches,dtype=torch.float16)
    report['models']=artifacts
    if args.fixed_manifest:
        report['fixedManifestSha256']=evaluator.sha256
        report['sourceSha256']=[row['sourceSha256'] for row in records]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    for name in args.model:registry.unload(name)


if __name__=='__main__':main()
