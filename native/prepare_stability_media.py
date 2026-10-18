"""Prepare new evaluation-only clips without touching training or final-hidden."""
import argparse
import json
from pathlib import Path
import subprocess
from prepare_video_degradation import ROOT, FFMPEG, file_hash, prepare_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=['hero', 'namakura'], default='hero')
    args = parser.parse_args()
    historical = args.source == 'namakura'
    starts, geometry = ((60, 180), '960:752') if historical else ((25, 80, 150), '1280:536')
    source = ROOT / f'diagnostics/stability-media/{args.source}-original.webm'
    if not source.is_file():
        raise FileNotFoundError('Download the source documented in docs/reproducibility-20261011.md first')
    destination = ROOT / f'diagnostics/video-sequences/stability-{args.source}-20261011'
    if destination.exists():
        raise FileExistsError('Preserve previous evaluation data')
    destination.mkdir(parents=True)
    provenance = {
        'title': 'Hero', 'author': 'Blender Foundation', 'year': 2018,
        'source': 'https://upload.wikimedia.org/wikipedia/commons/a/a9/HERO_-_Blender_Open_Movie-full_movie.webm',
        'licensePage': 'https://commons.wikimedia.org/wiki/File:HERO_-_Blender_Open_Movie-full_movie.webm',
        'license': 'CC BY 4.0', 'licenseUrl': 'https://creativecommons.org/licenses/by/4.0/',
        'sourceSha256': file_hash(source), 'sourceBytes': source.stat().st_size,
        'evaluationOnly': True, 'trainingUse': False, 'finalHiddenAccessed': False,
        'sourceSeparation': 'Different production from ReZERO; acquired after existing checkpoints were completed. No training/source file was replaced.',
        'limitations': ['Independent 2D animation, not another modern Japanese anime title.', 'Downloaded source is compressed; resized decoded source is the reference, not a pristine studio master.'],
        'changes': f'Software-decoded excerpts at {starts} seconds; Lanczos resized to {geometry}, lossless FFV1 reference; x2 LR H264/H265/AV1 CRF24 GOP24.',
        'manifests': [],
    }
    if historical:
        provenance.update({
            'title': 'Namakura Gatana', 'author': "Jun'ichi Kouchi", 'year': 1917,
            'source': 'https://upload.wikimedia.org/wikipedia/commons/b/bf/Kouichi_Jun%27ichi_-_Namakura_Gatana_%281917%29_-_4-minute_restored_version.webm',
            'licensePage': 'https://commons.wikimedia.org/wiki/File:Kouichi_Jun%27ichi_-_Namakura_Gatana_(1917)_-_4-minute_restored_version.webm',
            'license': 'Public domain per source file page', 'licenseUrl': None,
            'limitations': ['Historical monochrome anime, not evidence of modern-anime generalization.', 'Compressed/restored source; decoded resized reference rather than pristine master.'],
        })
    else:
        provenance['limitations'].append('The 150-second excerpts contain credits; report separately from animation shots at 25/80 seconds.')
    for start in starts:
        clip = ROOT / f'diagnostics/stability-media/{args.source}-{start}s-reference.mkv'
        if clip.exists():
            raise FileExistsError(clip)
        command = [str(FFMPEG), '-nostdin', '-hide_banner', '-loglevel', 'error', '-ss', str(start), '-i', str(source), '-t', '2.2', '-an', '-vf', f'scale={geometry}:flags=lanczos', '-c:v', 'ffv1', '-level', '3', '-threads', '2', str(clip)]
        subprocess.run(command, check=True, timeout=120)
        for codec in ('h264', 'h265', 'av1'):
            target = destination / f'{args.source}-{start}s-{codec}'
            prepare_video(clip, target, codec=codec, crf=24, gop=24, max_frames=48)
            provenance['manifests'].append({'path': (target / 'complete.json').relative_to(ROOT).as_posix(), 'sha256': file_hash(target / 'complete.json'), 'clipSeconds': start, 'codec': codec})
            (destination / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
            print(json.dumps({'prepared': target.name, 'frames': 48}), flush=True)
    print('Prepared independent evaluation clips', flush=True)


if __name__ == '__main__':
    main()
