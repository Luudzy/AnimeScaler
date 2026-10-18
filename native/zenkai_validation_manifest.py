"""Freeze replayable still-image validation records without altering source splits.

This opt-in builder uses the existing image degradations. It does not migrate
legacy validation caches and does not represent codec or real-video validation.
"""
import argparse
import hashlib
import io
import json
import platform
import random
from pathlib import Path

import numpy as np
import PIL
from PIL import Image

from artifact_io import atomic_json_create
import lite_degradations


APPROVED_ROOTS = (Path('F:/treino'),)
CASES = ('clean', 'heavy')


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _seed(*parts):
    encoded = json.dumps(parts, separators=(',', ':'), ensure_ascii=True).encode('utf-8')
    return int(_digest(encoded)[:16], 16)


def _approved_path(path, allowed_roots):
    resolved = Path(path).resolve()
    if not any(resolved.is_relative_to(Path(root).resolve()) for root in allowed_roots):
        raise ValueError(f'Source outside approved roots: {resolved}')
    return resolved


def _read_source(path, expected_sha):
    data = path.read_bytes()
    if _digest(data) != expected_sha:
        raise ValueError(f'Source checksum mismatch: {path}')
    with Image.open(io.BytesIO(data)) as image:
        if getattr(image, 'n_frames', 1) != 1:
            raise ValueError('Only static validation images are supported')
        return image.convert('RGB')


def _render(sample, source):
    if sample['split'] != 'validation':
        raise ValueError('These records support validation only')
    if sample['scale'] != 2 or sample['case'] not in CASES:
        raise ValueError('Only native x2 clean/heavy image validation is supported')
    if list(source.size) != sample['sourceSize']:
        raise ValueError('Source dimensions mismatch')
    crop = sample['cropHR']
    left, top, width, height = (crop[key] for key in ('left', 'top', 'width', 'height'))
    if (any(type(value) is not int for value in (left, top, width, height))
            or min(left, top) < 0 or min(width, height) < 2
            or width % 2 or height % 2
            or left + width > source.width or top + height > source.height):
        raise ValueError('Invalid or out-of-bounds HR crop')
    high = source.crop((left, top, left + width, top + height))
    low = lite_degradations.degrade(high, random.Random(sample['degradationSeed']), sample['case'])
    return low, high


def render_validation_sample(sample, *, allowed_roots=APPROVED_ROOTS):
    """Replay a saved record and fail if source pixels or degradation output drift."""
    if sample['split'] != 'validation':
        raise ValueError('These records support validation only')
    path = _approved_path(sample['sourceFile'], allowed_roots)
    source = _read_source(path, sample['sourceSha256'])
    low, high = _render(sample, source)
    if _digest(high.tobytes()) != sample['hrPixelsSha256']:
        raise ValueError('HR crop pixels mismatch')
    if _digest(low.tobytes()) != sample['lrPixelsSha256']:
        raise ValueError('Degradation output changed; check recorded implementation and library versions')
    return low, high


def build_validation_manifest(dataset, *, patch_lr=128, seed=193, per_image=1,
                              domain='anime', allowed_roots=APPROVED_ROOTS):
    """Build records in memory; only open the selected validation source images.

    Crop and degradation seeds depend on content identity, never row position.
    The original dataset's manifest, ordering and assignments are left intact.
    """
    if type(patch_lr) is not int or patch_lr < 1:
        raise ValueError('patch_lr must be a positive integer')
    if type(per_image) is not int or per_image < 1:
        raise ValueError('per_image must be a positive integer')
    if type(seed) is not int:
        raise ValueError('seed must be an integer')
    manifest_path = Path(dataset).resolve() / 'manifest.jsonl'
    manifest_bytes = manifest_path.read_bytes()
    rows = [json.loads(line) for line in manifest_bytes.decode('utf-8-sig').splitlines() if line.strip()]
    rows = [row for row in rows if row['split'] == 'validation'
            and (domain is None or row.get('domain', 'illustration') == domain)]
    if not rows:
        raise ValueError('No validation sources match the requested domain')
    samples = []
    size = patch_lr * 2
    for row in rows:
        source_path = Path(row['file'])
        if not source_path.is_absolute():
            source_path = manifest_path.parent / source_path
        source_path = _approved_path(source_path, allowed_roots)
        source = _read_source(source_path, row['sha256'])
        if min(source.size) < size:
            raise ValueError(f'Validation image too small for {size}px native crop: {source_path}')
        for crop_index in range(per_image):
            crop_seed = _seed(seed, row['sha256'], crop_index, size, 'crop')
            rng = random.Random(crop_seed)
            crop = {'left': rng.randrange(source.width - size + 1),
                    'top': rng.randrange(source.height - size + 1), 'width': size, 'height': size}
            for case in CASES:
                sample = {
                    'sourceFile': str(source_path), 'sourceSha256': row['sha256'],
                    'sourceSize': list(source.size), 'split': 'validation',
                    'domain': row.get('domain', 'illustration'), 'cluster': row.get('cluster'),
                    'cropIndex': crop_index, 'cropHR': crop, 'cropSeed': crop_seed,
                    'scale': 2, 'case': case,
                    'degradationSeed': _seed(seed, row['sha256'], crop_index, size, case),
                }
                low, high = _render(sample, source)
                sample.update(hrPixelsSha256=_digest(high.tobytes()), lrPixelsSha256=_digest(low.tobytes()))
                samples.append(sample)
    samples.sort(key=lambda sample: (sample['sourceSha256'], sample['sourceFile'],
                                     sample['cropIndex'], sample['case']))
    return {
        'schemaVersion': 1, 'scope': 'Fixed still-image validation; clean/heavy only; no final-hidden',
        'sourceManifest': str(manifest_path), 'sourceManifestSha256': _digest(manifest_bytes),
        'domain': domain, 'seed': seed, 'patchLR': patch_lr, 'patchHR': size,
        'perImage': per_image, 'cases': list(CASES),
        'allowedSourceRoots': [str(Path(root).resolve()) for root in allowed_roots],
        'provenance': {
            'builderSha256': _digest(Path(__file__).read_bytes()),
            'degradationSha256': _digest(Path(lite_degradations.__file__).read_bytes()),
            'python': platform.python_version(), 'pillow': PIL.__version__, 'numpy': np.__version__,
        },
        'realVideoValidation': 'REAL VIDEO VALIDATION NOT EXECUTED', 'samples': samples,
    }


def write_validation_manifest(dataset, output, **options):
    """Write a new sidecar file; never replace an existing file or dataset."""
    output = Path(output)
    report = build_validation_manifest(dataset, **options)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_create(report, output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--patch-lr', type=int, choices=(96, 128), default=128)
    parser.add_argument('--seed', type=int, default=193)
    parser.add_argument('--per-image', type=int, default=1)
    args = parser.parse_args()
    report = write_validation_manifest(args.dataset, args.output, patch_lr=args.patch_lr,
                                       seed=args.seed, per_image=args.per_image)
    print(json.dumps({'output': str(args.output), 'samples': len(report['samples']),
                      'scope': report['scope']}))


if __name__ == '__main__':
    main()
