"""Evaluation-only recovery after coordinator disappearance; prior PASS results retained."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from hip_operation import run_operation,save_json
from frozen_video_protocol import validate_protocol
ROOT=Path(__file__).resolve().parents[1]
RUN='zenkai-ablation-skip-factor-codec50-3k-continuation'


def main():
    base=ROOT/'diagnostics/zenkai-codec-pair-3k';path=base/'pipeline.json'
    old=json.loads(path.read_text());archive=base/'pipeline-before-video-recovery1.json'
    if archive.exists():raise FileExistsError('Previous recovery preserved')
    if old['phase']!='video-codec50' or any(old['results'][n]['state']!='PASS' for n in
             ('quality','gradients','benchmark-codec25','benchmark-codec50')):
        raise ValueError('Not the completed-measurements/incomplete-video state')
    save_json(archive,old)
    frozen=json.loads((ROOT/'diagnostics/zenkai-real-video-b-5k-complete/report.json').read_text())
    def command(a,out):
        args=[sys.executable,ROOT/'native/evaluate_real_video.py','--probe-checkpoint',
              ROOT/'models/zenkai/experiments'/RUN/'step-3000-master-fp32.pt','--probe-step','3000','--output',out/'video']
        for s in frozen['sequences']:args+=['--manifest',s['manifest']]
        return args
    def validate(out):
        report=out/'video/report.json';r=json.loads(report.read_text());validate_protocol(r)
        if r['state']!='complete' or sum(s['frameCount'] for s in r['sequences'])!=144:raise ValueError('Incomplete video')
        from export_zenkai import sha256
        return {'path':str(report),'sha256':sha256(report)}
    old.update(state='RUNNING',phase='video-codec50-recovery1',
        interruptionNote='Previous coordinator/child absent with no captured exit; cause NOT ESTABLISHED. Partial video discarded; all previous PASS operations retained.')
    save_json(path,old)
    operation=run_operation(base/'video-codec50-recovery1',command,validate,timeout=1200)
    old['results']['video-codec50']=operation
    old['operationRecords']={'video-codec50':str(base/'video-codec50-recovery1/operation.json')}
    old.update(state='COMPLETE',phase='completed',releaseApproved=False);save_json(path,old)


if __name__=='__main__':main()
