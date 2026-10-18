"""Paired full-frame real-footage screening; no optical-flow or player FPS claim."""
import argparse
import json
import re
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from prepare_video_degradation import ROOT, file_hash


def paired_temporal_metrics(output, previous, reference, old_reference):
    values = (output, previous, reference, old_reference)
    if any(v.shape != reference.shape or not torch.isfinite(v).all() for v in values):
        raise ValueError('Temporal tensors must have matching finite shapes')
    delta = reference - old_reference
    return {'referenceCompensatedDeltaMae': float(((output - previous) - delta).abs().mean()),
            'referenceDeltaMae': float(delta.abs().mean())}


def verified_pairs(manifest):
    manifest = Path(manifest).resolve()
    report = json.loads(manifest.read_text(encoding='utf-8'))
    rows = report.get('frames', [])
    if report.get('syntheticSource') is not False or len(rows) < 2 or report.get('frameCount') != len(rows):
        raise ValueError('Expected a complete, explicitly nonsynthetic sequence')
    pairs = []
    for index, row in enumerate(rows):
        if row.get('index') != index:
            raise ValueError('Frame order mismatch')
        paths = {}
        for kind in ('hr', 'lr'):
            path = (manifest.parent / row[kind]).resolve()
            if not path.is_relative_to(manifest.parent) or not path.is_file():
                raise ValueError('Missing or escaping frame path')
            if file_hash(path) != row.get(kind + 'Sha256'):
                raise ValueError('Frame hash mismatch')
            with Image.open(path) as image:
                if list(image.size) != report.get(kind + 'Size'):
                    raise ValueError('Frame dimension mismatch')
            paths[kind] = path
        pairs.append(paths)
    return report, pairs


def tensor(path):
    with Image.open(path) as image:
        pixels = np.asarray(image.convert('RGB')).copy()
    return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).float() / 255


def candidate_identity(metadata):
    config=metadata.get('config',{})
    run=config.get('run','')
    if run.startswith('zenkai-ablation-'):
        if not re.fullmatch(r'zenkai-ablation-[a-z0-9-]+',run):raise ValueError('Invalid ablation identity')
        return run
    if run=='zenkai-lite-v1-c-codec50-5k':
        if config.get('family')!='ZenkaiIdentitySkipProbe' or config.get('depth')!=5:
            raise ValueError('Wrong C identity-skip family/depth')
        return run
    if config.get('family')=='ZenkaiIdentitySkipProbe' or run.startswith('zenkai-lite-v1-c-'):
        raise ValueError('Unknown probe identity; do not infer original A/B from depth')
    depth = metadata.get('config', {}).get('depth')
    if depth not in (5, 6):
        raise ValueError('Unknown matched A/B candidate depth')
    return 'zenkai-a' if depth == 5 else 'zenkai-b'


def evaluate(manifests, models, output, artifacts, cut_threshold=.15, save_all_frames=False):
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        raise ValueError('Output must be a new directory inside the workspace')
    if not 0 < cut_threshold <= 1:
        raise ValueError('Invalid cut threshold')
    prepared = [(Path(p).resolve(), *verified_pairs(p)) for p in manifests]
    output.mkdir(parents=True)
    report = {'state': 'running', 'backend': 'torch-cpu', 'precision': 'fp32',
              'previewPolicy':'all frames' if save_all_frames else 'first/middle/last',
              'torch': str(torch.__version__), 'scope': 'Full-frame paired real-video screening',
              'releaseApproved': False, 'trainingUse': False, 'artifacts': artifacts,
              'cutPolicy': {'heuristicReferenceDeltaThreshold': cut_threshold,
                            'meaning': 'Suspected cuts or flashes; manual review required'},
              'limitations': ['Pixel delta error is reference-compensated, not optical-flow warping.',
                              'No certification of ghosting or perceptual flicker.',
                              'CPU execution time is not GPU or player FPS.',
                              'Short selected clips from one compressed episode; not generalization proof.'],
              'sequences': []}
    destination = output / 'report.json'
    def save(): destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
    save()
    try:
        with torch.inference_mode():
            for manifest, metadata, pairs in prepared:
                sequence = {'manifest': str(manifest), 'manifestSha256': file_hash(manifest),
                            'sourceSha256': metadata['sourceSha256'], 'fps': metadata['fps'],
                            'hrSize': metadata['hrSize'], 'lrSize': metadata['lrSize'],
                            'frameCount': len(pairs), 'suspectedCutTransitions': [], 'models': {}}
                records = {name: {'mae': [], 'seconds': [], 'temporal': [], 'allTemporal': []}
                           for name in ('bilinear', *models)}
                previous = None
                preview = output / manifest.parent.name
                preview.mkdir()
                for index, pair in enumerate(pairs):
                    x, target = tensor(pair['lr']), tensor(pair['hr'])
                    base = F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
                    values = {'bilinear': base}
                    timings = {'bilinear': None}
                    for name, model in models.items():
                        start = time.perf_counter()
                        prediction = model(x).float()
                        if prediction.shape != target.shape or not torch.isfinite(prediction).all():
                            raise ValueError('Invalid full-frame model output')
                        values[name] = prediction.clamp(0, 1)
                        timings[name] = time.perf_counter() - start
                    if previous is not None:
                        old_target, old_values = previous
                        reference_delta = float((target - old_target).abs().mean())
                        suspected_cut = reference_delta > cut_threshold
                        if suspected_cut: sequence['suspectedCutTransitions'].append(index)
                    for name, value in values.items():
                        records[name]['mae'].append(float((value - target).abs().mean()))
                        if timings[name] is not None: records[name]['seconds'].append(timings[name])
                        if previous is not None:
                            score = paired_temporal_metrics(value, old_values[name], target, old_target)
                            records[name]['allTemporal'].append(score['referenceCompensatedDeltaMae'])
                            if not suspected_cut: records[name]['temporal'].append(score['referenceCompensatedDeltaMae'])
                    if save_all_frames or index in (0, len(pairs) // 2, len(pairs) - 1):
                        for name, value in {'reference': target, **values}.items():
                            pixels = np.rint(value[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                            Image.fromarray(pixels).save(preview / f'{index:04d}-{name}.png')
                    previous = target, values
                for name, rows in records.items():
                    sequence['models'][name] = {'mae': float(np.mean(rows['mae'])),
                        'referenceCompensatedTemporalMae': float(np.mean(rows['temporal'])) if rows['temporal'] else None,
                        'allTransitionTemporalMae': float(np.mean(rows['allTemporal'])),
                        'includedTransitions': len(rows['temporal']),
                        'cpuMeanSecondsPerInference': float(np.mean(rows['seconds'])) if rows['seconds'] else None}
                report['sequences'].append(sequence)
                save()
                print(json.dumps({'completedSequence': manifest.parent.name, 'frames': len(pairs)}), flush=True)
        report['state'] = 'complete'
        save()
        return report
    except Exception as error:
        report['state'] = 'failed'
        report['error'] = str(error)
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, action='append', required=True)
    checkpoint_group=parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument('--checkpoint', type=Path)
    checkpoint_group.add_argument('--control-checkpoint', type=Path)
    checkpoint_group.add_argument('--probe-checkpoint', type=Path)
    parser.add_argument('--probe-step',type=int,choices=(1000,3000,5000),default=3000)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--save-all-frames',action='store_true',help='Write every RGB8 frame for motion review; metrics/protocol unchanged')
    args = parser.parse_args()
    torch.set_num_threads(2)
    from model_registry import build_registry
    from export_zenkai import load_checkpoint
    registry = build_registry(ROOT)
    legacy = registry.load('adore-lite-v3', device='cpu', precision='fp32')
    checkpoint=args.checkpoint or args.control_checkpoint or args.probe_checkpoint
    if args.control_checkpoint:
        from zenkai_experiments import load_control_checkpoint
        candidate,metadata=load_control_checkpoint(checkpoint)
        candidate_id='adore-v3-retrained-control'
    elif args.probe_checkpoint:
        from zenkai_probe_evaluation import load_probe
        candidate,metadata=load_probe(checkpoint,args.probe_step)
        candidate_id=candidate_identity(metadata)
        candidate=candidate.deploy_copy().eval()
    else:
        candidate, metadata = load_checkpoint(checkpoint)
        candidate_id = candidate_identity(metadata)
        candidate = candidate.deploy_copy().eval()
    artifacts = {'adore-lite-v3': legacy.artifact,
                 candidate_id: {'path': str(checkpoint.resolve()), 'sha256': file_hash(checkpoint),
                              'step': metadata['step'], 'fusion': 'Exact V3' if args.control_checkpoint else 'deploy_copy FP32'}}
    try:
        evaluate(args.manifest, {'adore-lite-v3': legacy.infer, candidate_id: candidate}, args.output, artifacts,
                 save_all_frames=args.save_all_frames)
    finally:
        registry.unload('adore-lite-v3')


if __name__ == '__main__': main()
