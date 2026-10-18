"""Repeat CPU video after a proven labeling defect; preserve all original operations."""
import json
from pathlib import Path
import sys
from hip_operation import run_operation, save_json
from frozen_video_protocol import validate_protocol
from prepare_video_degradation import file_hash

ROOT = Path(__file__).resolve().parents[1]
RUN = 'zenkai-lite-v1-c-codec50-5k'


def main():
    base = ROOT/'diagnostics/zenkai-c5k'
    path = base/'pipeline.json'
    original = json.loads(path.read_text())
    archive = base/'pipeline-before-video-identity-recovery1.json'
    if archive.exists() or original['state'] != 'COMPLETE':
        raise ValueError('Preserve recovery; original operations must have finished')
    video = original['results']['video']
    frozen = json.loads(Path(video['result']['path']).read_text())
    validate_protocol(frozen)
    checkpoint = ROOT/'models/zenkai/experiments'/RUN/'step-5000-master-fp32.pt'
    pin = file_hash(checkpoint)
    if video['state'] != 'PASS' or frozen['artifacts'].get('zenkai-a', {}).get('sha256') != pin or RUN in frozen['artifacts']:
        raise ValueError('Not the proven correct-checkpoint/wrong-label state')
    save_json(archive, original)
    reason = 'C checkpoint SHA is correct, but old depth-based candidate_identity labeled it zenkai-a. Original numeric measurements retained; a new complete CPU run supplies corrected identity, with no partial merging or model invalidation.'
    original.update(state='RUNNING', phase='video-identity-recovery1', videoIdentityRepair=reason)
    save_json(path, original)
    def command(attempt, out):
        args = [sys.executable, ROOT/'native/evaluate_real_video.py', '--probe-checkpoint', checkpoint,
            '--probe-step', '5000', '--output', out/'video']
        for sequence in frozen['sequences']:
            args += ['--manifest', sequence['manifest']]
        return args
    def validate(out):
        report_path = out/'video/report.json'
        report = json.loads(report_path.read_text())
        validate_protocol(report)
        if report['artifacts'].get(RUN, {}).get('sha256') != pin or any(RUN not in s['models'] for s in report['sequences']):
            raise ValueError('Wrong C video identity/checkpoint')
        if report['precision'] != frozen['precision'] or report['backend'] != frozen['backend'] or report['cutPolicy'] != frozen['cutPolicy']:
            raise ValueError('Video protocol changed')
        for old, new in zip(frozen['sequences'], report['sequences']):
            if old['manifestSha256'] != new['manifestSha256'] or old['suspectedCutTransitions'] != new['suspectedCutTransitions']:
                raise ValueError('Video/cut identity changed')
        return {'path':str(report_path), 'sha256':file_hash(report_path)}
    original['results']['video'] = run_operation(base/'video-identity-recovery1', command, validate, timeout=1200)
    original['operationRecords'] = {'video':str(base/'video-identity-recovery1/operation.json')}
    original.update(state='COMPLETE', phase='completed', releaseApproved=False)
    save_json(path, original)


if __name__ == '__main__':
    main()
