"""One subprocess at a time; HIP retries never reuse a failed context or timings."""
import argparse,datetime,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
PYTHON=Path('F:/comfyui-rocm-071926/python_env/python.exe')
p=argparse.ArgumentParser()
p.add_argument('--stages',nargs='+',choices=['quality','performance','renderer'],default=['quality','performance','renderer'])
p.add_argument('--attempt-start',type=int,default=1)
args=p.parse_args()
results=[]
for stage in args.stages:
    final=None
    for attempt in range(args.attempt_start,args.attempt_start+3):
        out=HERE/f'{stage}-attempt-{attempt}.json'
        log=HERE/f'{stage}-attempt-{attempt}.log'
        if out.exists() or log.exists():raise FileExistsError('Preserve previous measurements: '+str(log))
        cmd=[str(PYTHON),str(HERE/'measure.py'),stage,'--output',str(out)]
        start=datetime.datetime.now(datetime.timezone.utc).isoformat()
        with log.open('w',encoding='utf-8') as stream:
            run=subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        # MIOpen writes native CP1252 bytes even when Python stdout is UTF-8.
        # HIP failure signatures are ASCII; retain raw logs and decode safely.
        content=log.read_text(encoding='utf-8',errors='replace')
        hip='unspecified launch failure' in content.lower()
        final={'stage':stage,'attempt':attempt,'exitCode':run.returncode,'started':start,
               'ended':datetime.datetime.now(datetime.timezone.utc).isoformat(),'report':str(out.relative_to(ROOT)),
               'log':str(log.relative_to(ROOT)),'status':'PASS' if run.returncode==0 else ('HIP_INFRA_FAILURE' if hip else 'FAIL')}
        results.append(final)
        (HERE/'pipeline.json').write_text(json.dumps({'status':'RUNNING','attempts':results},indent=2),encoding='utf-8')
        print(json.dumps(final),flush=True)
        if run.returncode==0 or not hip:break
    if final['status']=='FAIL':break  # no activation after a quality/protocol failure
(HERE/'pipeline.json').write_text(json.dumps({'status':'COMPLETE' if len([r for r in results if r['status']=='PASS'])==len(args.stages) else 'INCONCLUSIVE',
                                               'attempts':results},indent=2),encoding='utf-8')
if len([r for r in results if r['status']=='PASS'])!=len(args.stages):
    raise SystemExit(1)
