"""Immutable codec-intra image validation sidecar; never temporal footage."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import subprocess
import numpy as np
from PIL import Image
from prepare_codec_cache import roundtrip
from prepare_video_degradation import ENCODERS, FFMPEG, ROOT
from zenkai_validation_manifest import APPROVED_ROOTS, render_validation_sample

CRF = {'h264': 24, 'h265': 24, 'av1': 24}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def render_codec_sample(sample, folder, allowed_roots=APPROVED_ROOTS):
    if sample['split'] != 'validation' or sample['base']['case'] != 'clean':
        raise ValueError('Codec cache only supports fixed clean validation sources')
    _, high = render_validation_sample(sample['base'], allowed_roots=allowed_roots)
    folder = Path(folder).resolve()
    path = (folder / sample['lrFile']).resolve()
    if not path.is_relative_to(folder):
        raise ValueError('Codec file escaped cache directory')
    data = path.read_bytes()
    if digest(data) != sample['lrFileSha256']:
        raise ValueError('Codec image bytes changed')
    with Image.open(path) as image:
        low = image.convert('RGB')
    if low.size != (high.width // 2, high.height // 2) or digest(low.tobytes()) != sample['lrPixelsSha256']:
        raise ValueError('Codec pixels or dimensions changed')
    return low, high


def prepare_codec_manifest(fixed, output, count=32, codecs=('h264', 'h265', 'av1'), allowed_roots=APPROVED_ROOTS):
    fixed = Path(fixed).resolve(); output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Refusing to overwrite codec validation cache')
    if not output.is_relative_to(ROOT):
        raise ValueError('Codec validation cache must remain in workspace')
    if type(count) is not int or count < 1 or not codecs or len(set(codecs)) != len(codecs) or any(c not in ENCODERS for c in codecs):
        raise ValueError('Invalid count or codecs')
    raw = fixed.read_bytes(); manifest = json.loads(raw)
    if manifest.get('schemaVersion') != 1 or manifest.get('domain') != 'anime':
        raise ValueError('Expected frozen anime validation manifest')
    if digest(Path(manifest['sourceManifest']).read_bytes()) != manifest['sourceManifestSha256']:
        raise ValueError('Source manifest changed')
    rows = sorted((row for row in manifest['samples'] if row['split'] == 'validation' and row['case'] == 'clean'),
                  key=lambda row: row['sourceSha256'])
    if not rows:
        raise ValueError('No validation sources')
    indices = np.linspace(0, len(rows) - 1, min(count, len(rows)), dtype=int)
    selected = [rows[int(index)] for index in indices]
    low_frames = [np.asarray(render_validation_sample(row, allowed_roots=allowed_roots)[0]) for row in selected]
    output.mkdir(parents=True)
    samples = []
    for codec in codecs:
        decoded = roundtrip(low_frames, ENCODERS[codec], CRF[codec])
        for index, (base, pixels) in enumerate(zip(selected, decoded)):
            path = output / f'{codec}-{index:04d}.png'
            Image.fromarray(pixels).save(path)
            samples.append({'split': 'validation', 'case': f'codec-intra-{codec}',
                            'sourceSha256': base['sourceSha256'], 'base': base,
                            'lrFile': path.name, 'lrFileSha256': digest(path.read_bytes()),
                            'lrPixelsSha256': digest(pixels.tobytes()),
                            'codec': codec, 'crf': CRF[codec], 'gop': 1, 'pixelFormat': 'yuv420p'})
    report = {'schemaVersion': 1, 'scope': 'Fixed still-image crops with actual intra-only codec roundtrip; not video',
              'fixedManifestSha256': digest(raw), 'fixedManifest': str(fixed), 'samples': samples,
              'countPerCodec': len(selected), 'cases': [f'codec-intra-{codec}' for codec in codecs],
              'realVideoValidation': 'REAL VIDEO VALIDATION NOT EXECUTED',
              'codecImplementationSha256': digest(Path(__file__).with_name('prepare_codec_cache.py').read_bytes()),
              'ffmpegVersion': subprocess.run([str(FFMPEG), '-version'], capture_output=True, text=True, check=True).stdout.splitlines()[0]}
    (output / 'complete.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixed', type=Path, default=ROOT / 'native/training-data/zenkai/validation-fixed.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'native/training-data/zenkai/codec-validation-fixed')
    parser.add_argument('--count', type=int, default=32)
    args = parser.parse_args()
    report = prepare_codec_manifest(args.fixed, args.output, args.count)
    print(json.dumps({'samples': len(report['samples']), 'scope': report['scope'], 'output': str(args.output)}))


if __name__ == '__main__':
    main()
