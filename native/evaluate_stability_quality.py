"""Evaluate additional production sources; never select/promote checkpoints."""
import argparse
import json
from pathlib import Path
import torch

from model_registry import build_registry
from prepare_video_degradation import ROOT, file_hash
from evaluate_real_video import evaluate, verified_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    if args.device != 'cpu':
        raise ValueError('This paired evaluator uses CPU FP32; GPU/player timings are measured separately')
    torch.set_num_threads(2)
    for manifest in args.manifest:
        verified_pairs(manifest)
        if 'stability-' not in str(manifest.parent.parent):
            raise ValueError('Only the newly prepared independent stability sources are accepted')
    registry = build_registry(ROOT)
    names = ['adore-lite-v3', 'zenkai-animevideo-30k']
    models = {name: registry.load(name, device='cpu', precision='fp32') for name in names}
    artifacts = {name: model.artifact for name, model in models.items()}
    try:
        report = evaluate(args.manifest, models, args.output, artifacts, save_all_frames=True)
    finally:
        for name in names:
            registry.unload(name)
    report.update({
        'scope': 'Additional independent production sources, aligned decoded-reference/codec-LR pairs; evaluation only',
        'sourceProvenance': [{'path': str(manifest.parent.parent / 'provenance.json'), 'sha256': file_hash(manifest.parent.parent / 'provenance.json')} for manifest in args.manifest],
        'humanOrBlindReview': 'NOT EXECUTED',
        'limitations': [
            'Reference-compensated temporal deltas do not use optical flow.',
            'Compressed source and resized decoded reference; not a pristine studio master.',
            'Independent styles and short excerpts do not establish broad modern-anime generalization.',
            'CPU FP32 quality is separate from ROCm FP16/browser performance.',
            'No checkpoint promotion, training or final-hidden access.',
        ],
    })
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'state': report['state'], 'sequences': len(report['sequences']), 'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
