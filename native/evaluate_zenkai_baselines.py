"""Same-input registry baseline screening; never evaluates final-hidden images."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from server import ROOT
import torch
import numpy as np
from PIL import Image, ImageDraw
from torch.nn import functional as F
from model_registry import build_registry
from lite_v31_evaluation import FixedEvaluator
from lite_evaluation import load_patches
from evaluate_sr_temporal import evaluate_synthetic_temporal


class EvaluationAdapter(torch.nn.Module):
    """Compare native 2x/4x networks at the same 2x target resolution."""
    def __init__(self, source):
        super().__init__()
        self.source = source
        self.network = source.backend.model

    def forward(self, value):
        result = self.source.infer(value)
        target = (value.shape[-2] * 2, value.shape[-1] * 2)
        if result.shape[-2:] != target:
            result = F.interpolate(result, size=target, mode='bilinear', align_corners=False)
        if not torch.isfinite(result).all():
            raise ValueError('Non-finite baseline output')
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-domain', type=int, default=32)
    parser.add_argument('--output', type=Path, default=ROOT/'diagnostics/zenkai-baselines')
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError('ROCm GPU required for matched FP16 baseline evaluation')
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    registry = build_registry(ROOT)
    names = ['adore-lite-v3', 'realesr-anime', 'hfa2k-cugan']
    models = {name: EvaluationAdapter(registry.load(name, 'torch', device='cuda', precision='fp16')) for name in names}
    dataset = ROOT/'native/training-data/v3-hfa2k'
    evaluator = FixedEvaluator(dataset, dataset/'codec-validation', per_domain=args.per_domain)
    preview_indices = set(evaluator.selected[:3])
    preview_dir = args.output/'comparisons'
    preview_dir.mkdir(exist_ok=True)
    def save_comparison(case, index, target, outputs):
        if index not in preview_indices:
            return
        values = {'reference': target, **outputs}
        width, height = target.shape[-1], target.shape[-2]
        sheet = Image.new('RGB', (width*3, (height+24)*2), 'white')
        draw = ImageDraw.Draw(sheet)
        for cell, (name, value) in enumerate(values.items()):
            pixels = value[0].detach().float().clamp(0,1).cpu().permute(1,2,0).numpy()
            image = Image.fromarray(np.rint(pixels*255).astype(np.uint8))
            image.save(preview_dir/f'{case}-{index:04d}-{name}.png')
            left, top = (cell%3)*width, (cell//3)*(height+24)
            draw.text((left+4, top+4), name, fill='black')
            sheet.paste(image, (left, top+24))
        sheet.save(preview_dir/f'{case}-{index:04d}-comparison.png')
    report = evaluator.run(models, on_outputs=save_comparison)
    temporal = evaluate_synthetic_temporal(models, load_patches(size=224, dataset=dataset), dtype=torch.float16)
    (args.output/'temporal_screening.json').write_text(json.dumps(temporal, indent=2), encoding='utf-8')
    report.update(backend='ROCm', precision='fp16', gpu=torch.cuda.get_device_name(),
                  targetScale=2, manifestSha256=hashlib.sha256((dataset/'manifest.jsonl').read_bytes()).hexdigest(),
                  models={name: {'nativeScale': registry.get(name).native_scale,
                                 'sha256': hashlib.sha256(Path(registry.get(name).weights_path).read_bytes()).hexdigest()}
                          for name in names},
                  limitations=['Validation crops only; not full-resolution quality or video validation.',
                               'Native 4x output resized to common 2x target using bilinear.',
                               'No training approval or model promotion follows automatically.'])
    (args.output/'quality_screening.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    baseline_path = args.output/'baseline_report.json'
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
        baseline['quality'] = report
        baseline['syntheticTemporal'] = temporal
        baseline['zenkaiTrainingApproved'] = False
        baseline['pending'] = ['Visual review', 'Official ESPAN architecture verification']
        baseline_path.write_text(json.dumps(baseline, indent=2), encoding='utf-8')
    print('Quality screening complete; no training or promotion started.', flush=True)


if __name__ == '__main__':
    main()
