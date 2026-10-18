"""Export labeled fixed crops and lossless moving panels from completed evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'native'))
from prepare_video_degradation import FFMPEG, file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--region', choices=['left', 'center', 'right'], default='center')
    args = parser.parse_args()
    report = json.loads((args.evaluation / 'report.json').read_text(encoding='utf-8'))
    if report['state'] != 'complete' or args.output.exists():
        raise ValueError('Require completed evaluation and a new review directory')
    args.output.mkdir(parents=True)
    provenance = {'scope': f'Labeled static {args.region} crops and lossless motion panels for manual review', 'humanOrBlindReview': 'NOT EXECUTED',
                  'sourceReportSha256': file_hash(args.evaluation / 'report.json'), 'outputs': []}
    names = ['reference', 'bilinear', 'adore-lite-v3', 'zenkai-animevideo-30k']
    for sequence in report['sequences']:
        source = args.evaluation / Path(sequence['manifest']).parent.name
        output = args.output / source.name; output.mkdir()
        with Image.open(source / '0000-reference.png') as image:
            width, height = image.size
        cw, ch = min(320, width), min(240, height)
        left = {'left': 0, 'center': (width - cw) // 2, 'right': width - cw}[args.region]
        top = (height - ch) // 2
        roi = (left, top, left + cw, top + ch)
        count = sequence['frameCount']
        for index in range(count):
            panel = Image.new('RGB', (cw * 4, ch + 24), '#111822')
            draw = ImageDraw.Draw(panel)
            for column, name in enumerate(names):
                with Image.open(source / f'{index:04d}-{name}.png') as image:
                    panel.paste(image.convert('RGB').crop(roi), (column * cw, 24))
                draw.text((column * cw + 6, 6), name, fill='white')
            panel.save(output / f'{index:08d}.png')
        rate = sequence['fps']
        movie = output / f'{args.region}-motion.mkv'
        command = [str(FFMPEG), '-nostdin', '-hide_banner', '-loglevel', 'error', '-framerate', rate, '-i', str(output / '%08d.png'), '-frames:v', str(count), '-c:v', 'ffv1', '-level', '3', '-threads', '2', str(movie)]
        subprocess.run(command, check=True, timeout=120)
        provenance['outputs'].append({'sequence': source.name, 'roi': list(roi), 'frames': count, 'fps': rate, 'movie': str(movie.relative_to(args.output)), 'sha256': file_hash(movie),
                                      'stills': [f'{source.name}/{index:08d}.png' for index in (0, count // 2, count - 1)]})
        (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(json.dumps({'sequences': len(provenance['outputs']), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
