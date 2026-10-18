"""Prepare lossless fixed-ROI motion panels from already completed RGB8 outputs."""
import hashlib
import json
from pathlib import Path
import subprocess
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
FFMPEG = ROOT/'native/.tools/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe'
RUN = 'zenkai-lite-v1-c-codec50-5k'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    report = read(ROOT/'diagnostics/zenkai-c5k/comparison.json')
    operation = read(ROOT/'diagnostics/zenkai-c5k/motion-frames/operation.json')
    if report['operations']['video']['state'] != 'PASS' or operation['state'] != 'PASS':
        raise ValueError('Complete selected screening and full motion-review operation required')
    motion = read(operation['result']['path'])
    if motion['previewPolicy'] != 'all frames':
        raise ValueError('Full motion material required')
    new_folder = Path(operation['result']['path']).parent
    dest = ROOT/'diagnostics/zenkai-c5k/motion-review-v2'
    if dest.exists():
        raise FileExistsError('Preserve existing motion panels')
    if not FFMPEG.is_file():
        raise FileNotFoundError('Existing bundled ffmpeg required; no install')
    dest.mkdir()
    outputs = []
    for old, new in zip(report['video']['sequences'], motion['sequences']):
        if old['manifestSha256'] != new['manifestSha256'] or old['frameCount'] != 48 or new['frameCount'] != 48:
            raise ValueError('Frozen sequence mismatch')
        clip = Path(new['manifest']).parent.name
        fps = read(new['manifest'])['fps']
        entries = [('Reference', new_folder/clip, 'reference'),
            ('Bilinear', new_folder/clip, 'bilinear'),
            ('HistoricalV3', new_folder/clip, 'adore-lite-v3'),
            ('C5k', new_folder/clip, RUN)]
        records = []
        args = [str(FFMPEG), '-hide_banner', '-loglevel', 'error', '-nostdin', '-n', '-threads', '2']
        for label, folder, tag in entries:
            paths = [folder/f'{n:04d}-{tag}.png' for n in range(48)]
            if any(not p.is_file() for p in paths):
                raise ValueError('Missing output frame')
            records.append({'column': label, 'inputFrameHashes': [sha(p) for p in paths]})
            args += ['-framerate', fps, '-start_number', '0', '-i', str(folder/f'%04d-{tag}.png')]
        # Preserve the same384px ROI used in the still comparisons, without scaling.
        graph = ';'.join(f'[{i}:v]crop=384:384:448:100[c{i}]' for i in range(4))
        graph += ';'+''.join(f'[c{i}]' for i in range(4))+'hstack=inputs=4:shortest=1[out]'
        output = dest/(clip+'.mkv')
        args += ['-filter_complex_threads', '1', '-filter_complex', graph, '-map', '[out]',
            '-frames:v', '48', '-an', '-c:v', 'ffv1', '-level', '3', '-pix_fmt', 'bgr0', '-threads', '2', str(output)]
        result = subprocess.run(args, capture_output=True, text=True, timeout=120)
        (dest/(clip+'.stderr.log')).write_text(result.stderr, encoding='utf-8')
        if result.returncode != 0:
            raise RuntimeError('Motion review encoding failed; preserve partial artifact/log')
        decoded = subprocess.run([str(FFMPEG), '-hide_banner', '-loglevel', 'error', '-nostdin',
            '-threads', '2', '-i', str(output), '-map', '0:v:0', '-pix_fmt', 'rgb24',
            '-frames:v', '48', '-f', 'framemd5', '-'], capture_output=True, text=True, timeout=120, check=True)
        actual = [line.split(',')[-1].strip() for line in decoded.stdout.splitlines() if line and not line.startswith('#')]
        expected = []
        for frame in range(48):
            joined = Image.new('RGB', (1536,384))
            for column, (_, folder, tag) in enumerate(entries):
                with Image.open(folder/f'{frame:04d}-{tag}.png') as image:
                    joined.paste(image.convert('RGB').crop((448,100,832,484)), (column*384,0))
            expected.append(hashlib.md5(joined.tobytes()).hexdigest())
        if actual != expected:
            raise ValueError('Review encoding did not preserve exact RGB pixels/frame order')
        (dest/(clip+'.framemd5')).write_text(decoded.stdout, encoding='utf-8')
        outputs.append({'clip': clip, 'output': str(output), 'sha256': sha(output),
            'manifestSha256': new['manifestSha256'], 'frames': 48, 'nominalPlaybackFps': fps,
            'roi': [448,100,832,484], 'codec': 'FFV1 lossless BGR0',
            'decodedRgbFrameMd5Verified':True, 'columns': records})
    (dest/'provenance.json').write_text(json.dumps({'state':'PREPARED NOT PERCEPTUALLY REVIEWED',
        'scope':'Three nominal-rate lossless384px ROI panels from exact RGB8 outputs; no interpolation or new model inference. Includes original scene transition. No blind preference/full-frame/player/flicker certification.',
        'panels':outputs},indent=2),encoding='utf-8')
    (dest/'README.md').write_text('# Motion review panels\n\nLeft to right: Reference | Bilinear | Historical V3 | C5k. Each column is the same unscaled384px ROI used in the still panels. Each clip contains48 distinct frames at the nominal source rate24000/1001, roughly2 seconds; no looping or interpolation. FFV1/BGR0 keeps the stacked RGB pixels lossless, verified by decoding every stacked RGB frame.\n\nThese files prepare manual motion review; no human/blind perceptual verdict or player performance result has been collected. The600s scene transition is retained. A separate full-PNG CPU pass matched the selected full-frame metrics; review encoding itself did not run inference or supply the reported metrics.\n',encoding='utf-8')
    print('Three lossless matched motion panels prepared; perceptual review still pending')


if __name__ == '__main__':
    main()
