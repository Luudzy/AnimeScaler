"""Full PNG motion-review material, independent of the selected screening metrics."""
import json
from pathlib import Path
import sys
from hip_operation import run_operation
from frozen_video_protocol import validate_protocol
from prepare_video_degradation import file_hash

ROOT = Path(__file__).resolve().parents[1]
RUN = 'zenkai-lite-v1-c-codec50-5k'


def main():
    base = ROOT/'diagnostics/zenkai-c5k'
    original = json.loads((base/'comparison.json').read_text())['video']
    validate_protocol(original)
    checkpoint = ROOT/'models/zenkai/experiments'/RUN/'step-5000-master-fp32.pt'
    pin = file_hash(checkpoint)
    if original['artifacts'][RUN]['sha256'] != pin:
        raise ValueError('Selected C artifact changed')
    def command(attempt, out):
        args = [sys.executable,ROOT/'native/evaluate_real_video.py','--probe-checkpoint',checkpoint,
            '--probe-step','5000','--save-all-frames','--output',out/'video']
        for sequence in original['sequences']:
            args += ['--manifest',sequence['manifest']]
        return args
    def validate(out):
        path = out/'video/report.json'
        current = json.loads(path.read_text())
        validate_protocol(current)
        if current['previewPolicy'] != 'all frames' or current['artifacts'][RUN]['sha256'] != pin:
            raise ValueError('Wrong motion-review identity/policy')
        if current['cutPolicy'] != original['cutPolicy']:
            raise ValueError('Cut protocol changed')
        for old, new in zip(original['sequences'],current['sequences']):
            if old['manifestSha256'] != new['manifestSha256'] or old['suspectedCutTransitions'] != new['suspectedCutTransitions']:
                raise ValueError('Frozen sequence identity changed')
            for model, metrics in new['models'].items():
                for name in ('mae','referenceCompensatedTemporalMae','allTransitionTemporalMae','includedTransitions'):
                    if abs(metrics[name]-old['models'][model][name]) > 1e-9:
                        raise ValueError('Saving all frames changed selected quality metrics')
            clip = Path(new['manifest']).parent.name
            for tag in ('reference','bilinear','adore-lite-v3',RUN):
                if any(not (out/'video'/clip/f'{frame:04d}-{tag}.png').is_file() for frame in range(48)):
                    raise ValueError('Incomplete full-frame review output')
        return {'path':str(path),'sha256':file_hash(path),
            'scope':'Full motion-review outputs; matches selected metrics, not a new player/perceptual result'}
    result = run_operation(base/'motion-frames',command,validate,timeout=1200)
    if result['state'] != 'PASS':
        raise RuntimeError('Motion material incomplete; preserve selected screening and diagnostics')


if __name__ == '__main__':
    main()
