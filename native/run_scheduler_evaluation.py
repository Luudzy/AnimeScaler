"""Serial fixed/video/performance/gradient evaluation for new scheduler smoke."""
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from hip_operation import run_operation,save_json
ROOT=Path(__file__).resolve().parents[1]
GPU='F:/comfyui-rocm-071926/python_env/python.exe'
RUN='zenkai-ablation-skip-factor-scheduler-constant-v2-1k'
REFERENCE='zenkai-ablation-spab-identity-skip-1k'


def main():
    base=ROOT/'diagnostics/zenkai-scheduler-v2-1k'
    if base.exists():raise FileExistsError('Preserve prior evaluation')
    base.mkdir();status=base/'pipeline.json';results={}
    save_json(status,{'state':'WAITING','phase':'training','results':results,'releaseApproved':False})
    deadline=time.monotonic()+3600
    while True:
        r=json.loads((ROOT/'diagnostics/zenkai-training/scheduler-v2-1k-training/operation.json').read_text())
        if r['state']=='PASS':break
        if r['state'] in ('FAIL','NOT_MEASURED'):
            save_json(status,{'state':'STOPPED','phase':'training','results':results,'reason':r.get('reason')});return
        if time.monotonic()>deadline:raise TimeoutError('Scheduler wait expired')
        time.sleep(40)
    checkpoint=ROOT/'models/zenkai/experiments'/RUN/'step-1000-master-fp32.pt'
    def stage(name,command,validate,timeout=600):
        save_json(status,{'state':'RUNNING','phase':name,'results':results,'releaseApproved':False})
        results[name]=run_operation(base/name,command,validate,timeout=timeout)
        save_json(status,{'state':'RUNNING','phase':name,'results':results,'releaseApproved':False})
    def quality(out):
        q=out/'quality/quality.json';p=out/'quality/parity.json'
        r=json.loads(q.read_text());parity=json.loads(p.read_text())
        if r['state']!='complete' or sum(r['sampleCounts'].values())!=160 or not all(m['parityApproved'] for m in parity['models'].values()):raise ValueError('Fixed/parity incomplete')
        old=json.loads((ROOT/'diagnostics/zenkai-ablations/skip-loss-1k-quality/quality.json').read_text())
        for key in ('indices','validationManifestSha256','sourceManifestSha256','codecManifestSha256'):
            if r[key]!=old[key]:raise ValueError('Fixed inputs changed')
        return {'quality':str(q),'parity':str(p)}
    stage('quality',lambda a,out:[GPU,ROOT/'native/zenkai_probe_evaluation.py','--stage','scheduler-v2-1k','--output',out/'quality'],quality)
    def gradients(out):
        path=out/'gradients.json';r=json.loads(path.read_text())
        if r['state']!='PASS' or set(r['runs'])!={RUN,REFERENCE}:raise ValueError('Gradient diagnostic incomplete')
        return {'path':str(path)}
    stage('gradients',lambda a,out:[GPU,ROOT/'diagnostics/zenkai-overnight/probe_scheduler_gradients.py','--output',out/'gradients.json'],gradients)
    def benchmark(out):
        path=out/'benchmark/benchmark_results.json';r=json.loads(path.read_text())
        if not r['complete'] or len(r['cases'])!=4 or any(c['warmup']!=50 or c['iterations']!=500 for c in r['cases']):raise ValueError('Benchmark incomplete')
        return {'path':str(path)}
    stage('benchmark',lambda a,out:[GPU,ROOT/'native/benchmark_sr_models.py','--probe-checkpoint',checkpoint,
          '--probe-step','1000','--model',RUN,'--model','adore-lite-v3','--output',out/'benchmark','--warmup','50','--iterations','500'],benchmark)
    frozen=json.loads((ROOT/'diagnostics/zenkai-real-video-skip-frequency-1k/report.json').read_text())
    def video_command(a,out):
        args=[sys.executable,ROOT/'native/evaluate_real_video.py','--probe-checkpoint',checkpoint,'--probe-step','1000','--output',out/'video']
        for sequence in frozen['sequences']:args+=['--manifest',sequence['manifest']]
        return args
    def video(out):
        from frozen_video_protocol import validate_protocol
        path=out/'video/report.json';r=json.loads(path.read_text());validate_protocol(r)
        if r['state']!='complete' or sum(s['frameCount'] for s in r['sequences'])!=144:raise ValueError('Video incomplete')
        return {'path':str(path)}
    stage('video',video_command,video,1200)
    save_json(status,{'state':'COMPLETE','results':results,'releaseApproved':False})


if __name__=='__main__':main()
