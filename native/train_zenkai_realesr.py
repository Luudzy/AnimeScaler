"""Train the isolated Zenkai-RealESR Lite smoke candidate.

This runner intentionally does not modify the frozen Zenkai training scripts.
It records a complete recoverable state and refuses to overwrite any run.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from lite_online_dataset import OnlineAnimeDataset
from artifact_io import atomic_json_write,atomic_torch_save_new
from zenkai_evaluation import ZenkaiFixedEvaluator
from zenkai_losses import ZenkaiLoss
from zenkai_realesr_lite import ZenkaiRealESRLite, ZenkaiAnimeSRLite
from zenkai_realesr_recipe import RUN_ID, create_run_dir, make_recipe, sha256
from train_zenkai import _LegacyAdapter, learning_rate


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def snapshot(model, optimizer, scaler, step, config, dataset_rng, sampler_indices):
    numpy_state = np.random.get_state()
    return {
        'model': {key: value.detach().cpu().float().clone() for key, value in model.state_dict().items()},
        'optimizer': cpu_tree(optimizer.state_dict()),
        'scaler': scaler.state_dict(),
        'step': step,
        'config': config,
        'rng': {
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all(),
            'python': random.getstate(),
            'numpy': {
                'kind': numpy_state[0], 'keys': torch.from_numpy(numpy_state[1].astype(np.int64)),
                'position': numpy_state[2], 'hasGauss': numpy_state[3], 'cachedGauss': numpy_state[4],
            },
        },
        'datasetRng': dataset_rng,
        'samplerIndices': torch.as_tensor(sampler_indices, dtype=torch.long),
        'nextSampleOffset': step * config['effectiveBatch'],
        'scheduler': {'kind': 'explicit cosine with 300-step warmup', 'totalSteps': 500, 'completedStep': step},
    }


def smoke_composite(cases):
    """Composite using metrics available without optional LPIPS/MS-SSIM wheels."""
    ratios = []
    for values in cases.values():
        candidate = values['candidate']; previous = values['adore-lite-v3']
        ratios.append(candidate['mae'] / max(previous['mae'], 1e-8)
                      + .3 * candidate['edgeMae'] / max(previous['edgeMae'], 1e-8)
                      + .1 * candidate['laplacianMae'] / max(previous['laplacianMae'], 1e-8)
                      + .1 * (1 - candidate['ssim']) / max(1 - previous['ssim'], 1e-6))
    return {'composite': float(np.mean(ratios)),
            'components': ['mae', 'edgeMae', 'laplacianMae', 'ssim'],
            'lpipsSqueeze': 'NOT MEASURED', 'msSsim': 'NOT MEASURED'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--interval', type=int, default=500)
    parser.add_argument('--seed', type=int, default=929)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--accumulation', type=int, default=2)
    parser.add_argument('--patch-lr', type=int, default=128, choices=(96, 128))
    parser.add_argument('--codec-fraction', type=float, default=.25)
    parser.add_argument('--run', default=RUN_ID)
    parser.add_argument('--loss', choices=('zenkai_frequency', 'zenkai_baseline'), default='zenkai_frequency')
    parser.add_argument('--architecture', choices=('realesr', 'animesr-multiscale'), default='realesr')
    parser.add_argument('--degradation-profile', choices=('legacy', 'anime-real'), default='legacy')
    parser.add_argument('--train-dataset', type=Path, default=None)
    args = parser.parse_args()
    if args.steps != 500 or args.interval < 1 or args.interval > args.steps:
        parser.error('The smoke is exactly 500 steps and interval must be within the run')
    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError('Configured ROCm GPU required')

    source_path = ROOT / 'models/zenkai/experiments/zenkai-lite-v1-a-5k/config.json'
    source = json.loads(source_path.read_text(encoding='utf-8'))
    dataset = args.train_dataset.resolve() if args.train_dataset else ROOT / 'native/training-data/zenkai'
    fixed = dataset / 'validation-fixed.json'
    if args.train_dataset:
        fixed = ROOT / 'native/training-data/zenkai/validation-fixed.json'
    codec_cache = ROOT / 'native/training-data/v31/codec-full'
    if not (codec_cache / 'complete.json').is_file():
        codec_cache = None
    if not (dataset / 'manifest.jsonl').is_file() or not fixed.is_file():
        raise RuntimeError('Validated Zenkai manifests are missing')
    if args.codec_fraction and codec_cache is None:
        raise RuntimeError('Configured codec cache is missing')
    if str(torch.__version__) != source['torch'] or torch.version.hip != source['hip']:
        raise RuntimeError('Training stack differs from reviewed Zenkai protocol')
    if not args.train_dataset and sha256(dataset / 'manifest.jsonl') != source['datasetManifestSha256']:
        raise RuntimeError('Training manifest changed')

    config = make_recipe({
        'seed': args.seed, 'optimizer': 'AdamW', 'learningRate': [2e-4, 1e-5],
        'betas': [.9, .99], 'weightDecay': 1e-5, 'batch': args.batch,
        'accumulation': args.accumulation, 'effectiveBatch': args.batch * args.accumulation,
        'patchLR': args.patch_lr, 'patchHR': args.patch_lr * 2,
        'codecFraction': args.codec_fraction if codec_cache else 0,
        'codecCache': str(codec_cache) if codec_cache else None,
        'loss': args.loss, 'datasetManifestSha256': sha256(dataset / 'manifest.jsonl'),
        'degradationProfile': args.degradation_profile,
        'fixedValidationSha256': sha256(fixed), 'legacyV3Sha256': source['legacyV3Sha256'],
        'gpu': torch.cuda.get_device_name(), 'hip': torch.version.hip,
        'torch': str(torch.__version__), 'gitCommit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    }, run=args.run)
    run_dir = create_run_dir(ROOT, config)
    atomic_json_write(source,run_dir / 'source-config.json')
    log_path = run_dir / 'training.log'
    try:
        torch.set_num_threads(2)
        torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
        torch.backends.cudnn.benchmark = False
        model = (ZenkaiAnimeSRLite() if args.architecture == 'animesr-multiscale' else ZenkaiRealESRLite()).cuda()
        config['architecture'] = model.config['architecture']
        config['parameters'] = sum(parameter.numel() for parameter in model.parameters())
        config['sourceFilesSha256'] = {name: sha256(HERE / name) for name in (
            'zenkai_realesr_lite.py', 'zenkai_realesr_recipe.py', 'train_zenkai_realesr.py',
            'lite_online_dataset.py', 'lite_degradations.py', 'zenkai_losses.py', 'zenkai_evaluation.py')}
        atomic_json_write(config,run_dir / 'config.json')
        data = OnlineAnimeDataset(dataset, patch_lr=args.patch_lr, seed=args.seed,
                                  codec_fraction=config['codecFraction'], codec_cache=codec_cache,
                                  degradation_profile=args.degradation_profile)
        sampler = RandomSampler(data, replacement=True,
                                num_samples=args.steps * args.batch * args.accumulation,
                                generator=torch.Generator().manual_seed(args.seed))
        indices = list(sampler)
        batches = iter(DataLoader(data, batch_size=args.batch, sampler=indices, num_workers=0))
        evaluator = ZenkaiFixedEvaluator(fixed, per_case=32)
        if evaluator.sha256 != source['fixedValidationSha256']:
            raise RuntimeError('Validation manifest changed')
        from model_registry import build_registry
        registry = build_registry(ROOT)
        legacy = registry.load('adore-lite-v3', device='cuda', precision='fp16')
        if legacy.artifact['sha256'] != source['legacyV3Sha256']:
            raise RuntimeError('Historical V3 changed')
        criterion = ZenkaiLoss(args.loss).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(.9, .99), weight_decay=1e-5)
        scaler = torch.amp.GradScaler('cuda')
        initial = snapshot(model, optimizer, scaler, 0, config, data.rng.getstate(), indices)
        atomic_torch_save_new(initial, run_dir / 'initial-training-state.pt')
        del initial
        started = time.perf_counter(); history = []
        with log_path.open('w', encoding='utf-8') as log:
            for step in range(1, args.steps + 1):
                model.train(); optimizer.zero_grad(set_to_none=True); losses = []
                lr = learning_rate(step, args.steps)
                for group in optimizer.param_groups: group['lr'] = lr
                for _ in range(args.accumulation):
                    low, target = next(batches); low = low.cuda(); target = target.cuda()
                    with torch.autocast('cuda', dtype=torch.float16): prediction = model(low)
                    loss, terms = criterion(prediction, target)
                    if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite loss at {step}')
                    losses.append(float(loss.detach())); scaler.scale(loss / args.accumulation).backward()
                scaler.unscale_(optimizer)
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not torch.isfinite(grad): raise RuntimeError(f'Nonfinite gradient at {step}')
                old_scale = scaler.get_scale(); scaler.step(optimizer); scaler.update()
                if scaler.get_scale() < old_scale: raise RuntimeError(f'AMP update skipped at {step}')
                if step % 100 == 0:
                    entry = {'step': step, 'loss': float(np.mean(losses)), 'gradientNorm': float(grad),
                             'lr': lr, 'samples': data.samples, 'codecSamples': data.codec_samples}
                    print(json.dumps(entry), file=log, flush=True); print(json.dumps(entry), flush=True)
                if step % args.interval == 0 or step == args.steps:
                    model.eval(); deploy = model.deploy_copy().cuda().half().eval()
                    with torch.inference_mode():
                        scores = evaluator.evaluate({'candidate': lambda value: deploy(value.half()).float(),
                                                     'adore-lite-v3': _LegacyAdapter(legacy)},
                                                    device='cuda', perceptual=None)
                    summary = smoke_composite(scores['cases'])
                    current = snapshot(model, optimizer, scaler, step, config, data.rng.getstate(), indices)
                    atomic_torch_save_new(current, run_dir / f'step-{step}-training-state.pt')
                    master = {'model': current['model'], 'config': config, 'step': step}
                    atomic_torch_save_new(master, run_dir / f'step-{step}-master-fp32.pt')
                    history.append({'step': step, 'screening': scores, 'summary': summary,
                                    'trainingLoss': float(np.mean(losses))})
                    report = {**config, 'stepsExecuted': step, 'optimizerUpdates': step,
                              'trainingSeconds': time.perf_counter() - started, 'history': history,
                              'best': {'step': step, 'composite': summary['composite']}, 'promoted': False}
                    atomic_json_write(report,run_dir / 'report.json')
                    print(json.dumps({'validationStep': step, 'composite': summary['composite']}), file=log, flush=True)
                    print(json.dumps({'validationStep': step, 'composite': summary['composite']}), flush=True)
                    del deploy, current
        registry.unload('adore-lite-v3')
        print(json.dumps({'finished': True, 'stepsExecuted': args.steps, 'output': str(run_dir), 'releaseApproved': False}))
    except Exception as error:
        atomic_json_write({'status': 'FAIL', 'type': type(error).__name__, 'message': str(error)},
                          run_dir / 'failure.json')
        raise


if __name__ == '__main__':
    main()
