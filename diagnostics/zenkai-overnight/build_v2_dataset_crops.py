"""Matched immutable dataset-only V2/anime1k stills; not a moving preference study."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RUN = 'zenkai-ablation-skip-v2-dataset-1k'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    report = read(ROOT/'diagnostics/zenkai-v2-dataset-1k/comparison.json')
    old_report_path = ROOT/'diagnostics/zenkai-real-video-skip-frequency-1k/report.json'
    old_op = {'state':'PASS','result':{'path':str(old_report_path)}}
    new_op = report['operations']['video']
    if old_op['state'] != 'PASS' or new_op['state'] != 'PASS':
        raise ValueError('Both complete video operations required')
    old_folder = Path(old_op['result']['path']).parent
    new_folder = Path(new_op['result']['path']).parent
    old_video, new_video = read(old_report_path), report['video']
    if old_video['precision'] != new_video['precision'] or old_video['backend'] != new_video['backend']:
        raise ValueError('Video precision/backend changed')
    dest = ROOT/'diagnostics/zenkai-v2-dataset-1k/crops'
    if dest.exists():
        raise FileExistsError('Preserve previous panels')
    dest.mkdir()
    roi = (448, 100, 832, 484)
    records = []
    if len(new_video['sequences']) != 3:
        raise ValueError('Three complete sequences required')
    for old, new in zip(old_video['sequences'], new_video['sequences']):
        if old['manifestSha256'] != new['manifestSha256'] or old['frameCount'] != 48 or new['frameCount'] != 48:
            raise ValueError('Frozen video identity/count changed')
        clip = Path(new['manifest']).parent.name
        for frame in (0, 24, 47):
            # The references and baselines must also be byte-identical.
            for tag in ('reference', 'bilinear', 'adore-lite-v3'):
                name = f'{frame:04d}-{tag}.png'
                if sha(old_folder/clip/name) != sha(new_folder/clip/name):
                    raise ValueError('Reference/baseline RGB8 output changed')
            entries = [
                ('Reference', new_folder/clip/f'{frame:04d}-reference.png'),
                ('Bilinear', new_folder/clip/f'{frame:04d}-bilinear.png'),
                ('Historical V3', new_folder/clip/f'{frame:04d}-adore-lite-v3.png'),
                ('Anime 1k', old_folder/clip/f'{frame:04d}-zenkai-ablation-spab-identity-skip-1k.png'),
                ('Anime + V2 1k', new_folder/clip/f'{frame:04d}-{RUN}.png'),
            ]
            fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), dpi=100)
            for axis, (label, path) in zip(axes, entries):
                with Image.open(path) as im:
                    axis.imshow(im.crop(roi), interpolation='nearest')
                axis.set_title(label)
                axis.axis('off')
            fig.suptitle(f'{clip}, frame {frame}, fixed ROI {roi}; identical RGB8 crops')
            fig.tight_layout()
            output = dest/f'{clip}-{frame:04d}.png'
            fig.savefig(output)
            plt.close(fig)
            records.append({'clip': clip, 'frame': frame, 'roi': roi,
                'manifestSha256': new['manifestSha256'],
                'inputs': {label: {'path': str(path), 'sha256': sha(path)} for label, path in entries},
                'output': str(output), 'outputSha256': sha(output)})
    (dest/'provenance.json').write_text(json.dumps({
        'scope': 'Nine fixed still panels, not blind or moving perceptual evaluation',
        'comparisons': records}, indent=2), encoding='utf-8')
    print('Nine matched V2 dataset panels saved')


if __name__ == '__main__':
    main()
