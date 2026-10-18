"""Queue exact-family CPU exports and serial ROCm reloaded-artifact parity."""
import json
import argparse
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from hip_operation import run_operation,save_json
ROOT=Path(__file__).resolve().parents[1]
GPU='F:/comfyui-rocm-071926/python_env/python.exe'
RUNS={'codec25':'zenkai-ablation-spab-identity-skip-3k-continuation',
      'codec50':'zenkai-ablation-skip-factor-codec50-3k-continuation'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--recovery1',action='store_true');args=parser.parse_args()
    base=ROOT/('diagnostics/zenkai-codec-pair-3k-exports-recovery1' if args.recovery1 else 'diagnostics/zenkai-codec-pair-3k-exports')
    if base.exists():raise FileExistsError('Preserve previous export operations')
    base.mkdir();results={};status=base/'pipeline.json'
    save_json(status,{'state':'WAITING','phase':'evaluation','results':results,'releaseApproved':False})
    deadline=time.monotonic()+7200
    while True:
        p=json.loads((ROOT/'diagnostics/zenkai-codec-pair-3k/pipeline.json').read_text())
        if p['state']=='COMPLETE':break
        if p['state']=='STOPPED':
            save_json(status,{'state':'STOPPED','phase':'evaluation','results':results});return
        if time.monotonic()>deadline:raise TimeoutError('Evaluation wait expired')
        time.sleep(40)
    if p['results']['quality']['state']!='PASS':
        save_json(status,{'state':'STOPPED','reason':'No valid fixed/numeric quality operation','results':results});return
    for label,run in RUNS.items():
        name=label+'-export';save_json(status,{'state':'RUNNING','phase':name,'results':results,'releaseApproved':False})
        def command(a,out,run=run):
            return [sys.executable,ROOT/'native/export_zenkai_skip.py','--checkpoint',
                    ROOT/'models/zenkai/experiments'/run/'step-3000-master-fp32.pt',
                    '--output',ROOT/'models/zenkai/exports'/(run+'-export-attempt-'+str(a))]
        def validate(out,run=run):
            folder=ROOT/'models/zenkai/exports'/(run+'-export-'+out.name)
            r=json.loads((folder/'export-report.json').read_text())
            if r['state']!='complete' or not r['parityApproved'] or r['family']!='ZenkaiIdentitySkipProbe' or len(r['cases'])!=6:
                raise ValueError('Incomplete exact-family export')
            return {'folder':str(folder),'report':str(folder/'export-report.json')}
        result=run_operation(base/name,command,validate,timeout=900);results[name]=result
        if result['state']!='PASS':continue
        folder=result['result']['folder'];name=label+'-runtime'
        save_json(status,{'state':'RUNNING','phase':name,'results':results,'releaseApproved':False})
        def validate_runtime(out):
            path=out/'runtime.json';r=json.loads(path.read_text())
            if r['state']!='complete' or not r['parityApproved'] or len(r['cases'])!=6:raise ValueError('Runtime parity incomplete')
            return {'path':str(path)}
        result=run_operation(base/name,lambda a,out:[GPU,ROOT/'native/export_zenkai_skip.py',
                    '--validate-runtime',folder,'--output',out/'runtime.json'],validate_runtime)
        results[name]=result
    save_json(status,{'state':'COMPLETE','results':results,'releaseApproved':False})


if __name__=='__main__':main()
