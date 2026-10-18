"""Serial postprocessing; each GPU operation gets at most three clean attempts."""
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from hip_operation import run_operation,save_json

ROOT=Path(__file__).resolve().parents[1]
GPU='F:/comfyui-rocm-071926/python_env/python.exe'
CPU=sys.executable
CODEC25='zenkai-ablation-spab-identity-skip-3k-continuation'
CODEC50='zenkai-ablation-skip-factor-codec50-3k-continuation'


def main():
    base=ROOT/'diagnostics/zenkai-codec-pair-3k'
    if base.exists():raise FileExistsError('Preserve prior evaluation')
    base.mkdir();status=base/'pipeline.json';results={}
    save_json(status,{'state':'WAITING','phase':'training','results':results,'releaseApproved':False})
    deadline=time.monotonic()+5400
    training=ROOT/'diagnostics/zenkai-training/codec50-3k-training/operation.json'
    while True:
        value=json.loads(training.read_text())
        if value['state']=='PASS':break
        if value['state'] in ('FAIL','NOT_MEASURED'):
            save_json(status,{'state':'STOPPED','phase':'training','reason':value.get('reason'),'results':results});return
        if time.monotonic()>deadline:raise TimeoutError('Training wait expired')
        time.sleep(50)
    def stage(name,command,validator,timeout=600):
        save_json(status,{'state':'RUNNING','phase':name,'results':results,'releaseApproved':False})
        result=run_operation(base/name,command,validator,timeout=timeout)
        results[name]=result
        save_json(status,{'state':'RUNNING','phase':name,'results':results,'releaseApproved':False})
    def validate_quality(out):
        folder=out/'quality';q=json.loads((folder/'quality.json').read_text());p=json.loads((folder/'parity.json').read_text())
        if q['state']!='complete' or sum(q['sampleCounts'].values())!=160 or p['state']!='complete':raise ValueError('Incomplete fixed/parity operation')
        if not all(m['parityApproved'] for m in p['models'].values()):raise ValueError('Numerical parity gate failed')
        old=json.loads((ROOT/'diagnostics/zenkai-ablations/probes-3k-quality/quality.json').read_text())
        if q['indices']!=old['indices'] or q['validationManifestSha256']!=old['validationManifestSha256']:raise ValueError('Fixed inputs changed')
        return {'quality':str(folder/'quality.json'),'parity':str(folder/'parity.json')}
    stage('quality',lambda a,out:[GPU,ROOT/'native/zenkai_probe_evaluation.py','--stage','codec-pair-3k','--output',out/'quality'],validate_quality)
    def validate_gradient(out):
        path=out/'gradients.json';r=json.loads(path.read_text())
        if r['state']!='PASS' or set(r['runs'])!={CODEC25,CODEC50}:raise ValueError('Incomplete gradient operation')
        return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    stage('gradients',lambda a,out:[GPU,ROOT/'diagnostics/zenkai-overnight/probe_codec3k_gradients.py','--output',out/'gradients.json'],validate_gradient)
    for label,run in (('codec25',CODEC25),('codec50',CODEC50)):
        def command(attempt,out,run=run,label=label):
            args=[GPU,ROOT/'native/benchmark_sr_models.py','--probe-checkpoint',ROOT/'models/zenkai/experiments'/run/'step-3000-master-fp32.pt',
                  '--probe-step','3000','--model',run,'--output',out/'benchmark','--warmup','50','--iterations','500']
            if label=='codec25':args+=['--model','adore-lite-v3']
            return args
        def validate(out,label=label):
            path=out/'benchmark/benchmark_results.json';r=json.loads(path.read_text())
            if not r['complete'] or len(r['cases'])!=(4 if label=='codec25' else 2):raise ValueError('Incomplete benchmark')
            if any(c['iterations']!=500 or c['warmup']!=50 for c in r['cases']):raise ValueError('Benchmark protocol changed')
            return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        stage('benchmark-'+label,command,validate)
    frozen=json.loads((ROOT/'diagnostics/zenkai-real-video-b-5k-complete/report.json').read_text())
    def video_command(a,out):
        args=[CPU,ROOT/'native/evaluate_real_video.py','--probe-checkpoint',ROOT/'models/zenkai/experiments'/CODEC50/'step-3000-master-fp32.pt',
              '--probe-step','3000','--output',out/'video']
        for sequence in frozen['sequences']:args+=['--manifest',sequence['manifest']]
        return args
    def validate_video(out):
        from frozen_video_protocol import validate_protocol
        path=out/'video/report.json';r=json.loads(path.read_text());validate_protocol(r)
        if r['state']!='complete' or sum(s['frameCount'] for s in r['sequences'])!=144:raise ValueError('Incomplete video')
        return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    stage('video-codec50',video_command,validate_video,timeout=1200)
    save_json(status,{'state':'COMPLETE','results':results,'releaseApproved':False})


if __name__=='__main__':main()
