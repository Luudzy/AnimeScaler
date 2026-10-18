"""Matched fixed-image quality and blind visual panels, explicit experimental opt-in."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE/'.tools'))
os.environ.setdefault('TORCH_HOME',str(ROOT/'diagnostics/native-cache/torch'))
import torch
from torch.nn import functional as F
from evaluate_zenkai_baselines import EvaluationAdapter
from generate_sr_comparison import write_comparison
from model_registry import build_registry
from zenkai_artifacts import add_zenkai_exports
from zenkai_evaluation import ZenkaiFixedEvaluator


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zenkai-export',action='append')
    parser.add_argument('--retrained-control',type=Path)
    parser.add_argument('--model',action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--precision',choices=('fp32','fp16'),default='fp16')
    parser.add_argument('--per-case',type=int,default=32)
    parser.add_argument('--visual-count',type=int,default=2)
    parser.add_argument('--no-perceptual',action='store_true')
    parser.add_argument('--blind',action='store_true')
    parser.add_argument('--codec',type=Path,default=ROOT/'native/training-data/zenkai/codec-validation-fixed/complete.json')
    args=parser.parse_args()
    output=args.output.resolve()
    if not output.is_relative_to(ROOT):parser.error('Output must remain inside workspace')
    if output.exists():parser.error('Output exists; choose a new directory')
    if args.visual_count<0:parser.error('visual-count must be nonnegative')
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('GPU unavailable')
    torch.set_num_threads(2)
    registry=build_registry(ROOT)
    add_zenkai_exports(registry,args.zenkai_export)
    if args.retrained_control:
        from zenkai_artifacts import register_control_checkpoint
        register_control_checkpoint(registry,args.retrained_control)
    models={name:EvaluationAdapter(registry.load(name,'torch',device=args.device,precision=args.precision)) for name in args.model}
    evaluator=ZenkaiFixedEvaluator(ROOT/'native/training-data/zenkai/validation-fixed.json',per_case=args.per_case)
    evaluator.attach_codec(args.codec)
    feature=None
    if not args.no_perceptual:
        import lpips
        feature=lpips.LPIPS(net='squeeze').to(args.device).eval().requires_grad_(False)
    output.mkdir(parents=True)
    keys={};counts={case:0 for case in evaluator.samples}

    def image(value):
        pixels=value[0].float().clamp(0,1).cpu().permute(1,2,0).numpy()
        return Image.fromarray(np.uint8(np.rint(pixels*255)))

    def panels(case,sample,low,target,values):
        index=counts[case]
        if index>=args.visual_count:return
        folder=output/'comparisons'/case/f'{index:02d}'
        images={'input-nearest':image(F.interpolate(low,scale_factor=2,mode='nearest')),
                **{name:image(value) for name,value in values.items()},'reference':image(target)}
        keys[f'{case}/{index:02d}']={'sourceSha256':sample['sourceSha256'],
            'key':write_comparison(images,folder,blind=args.blind,seed=731+index)}
        counts[case]+=1

    try:
        report=evaluator.evaluate(models,device=args.device,perceptual=feature,on_outputs=panels)
        report.update(state='complete',device=args.device,precision=args.precision,inferenceBackend='torch',
                      backend='ROCm' if args.device=='cuda' and torch.version.hip else args.device,
                      gpu=torch.cuda.get_device_name() if args.device=='cuda' else None,
                      hip=torch.version.hip,torch=str(torch.__version__),
                      models={name:model.source.artifact for name,model in models.items()},
                      promoted=False,releaseApproved=False,visualReview='NOT EXECUTED',
                      humanBlindPreferenceStudy='NOT EXECUTED',performance='NOT MEASURED')
        (output/'quality.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        (output/'blind-key.json').write_text(json.dumps(keys,indent=2),encoding='utf-8')
    finally:
        for name in args.model:registry.unload(name)
    print(json.dumps({'state':'complete','output':str(output),'cases':report['sampleCounts'],'promoted':False}))


if __name__=='__main__':main()
