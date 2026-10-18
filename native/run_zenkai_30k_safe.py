"""Crash-resilient segmented continuation for the Zenkai AnimeVideo recipe."""
import argparse, json, os, subprocess, time
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--source-run',type=Path,required=True); p.add_argument('--dataset',type=Path,required=True); p.add_argument('--target-steps',type=int,default=30000); p.add_argument('--segment',type=int,default=1000); p.add_argument('--root',type=Path,required=True); p.add_argument('--python',type=Path,required=True); a=p.parse_args()
    status=a.root/'safe-30k-status.json'; a.root.mkdir(parents=True,exist_ok=True)
    source=a.source_run.resolve(); entries=[]
    while True:
        states=list(source.glob('step-*-training-state.pt'))
        if not states: raise RuntimeError(f'No intact training state in {source}')
        current=max(int(x.stem.split('-')[1]) for x in states)
        if current>=a.target_steps: break
        end=min(a.target_steps,current+a.segment); output=a.root/f'segment-{current+1:05d}-{end:05d}'
        env=os.environ.copy(); env['MIOPEN_DISABLE_CACHE']='1'; env['MIOPEN_FIND_MODE']='FAST'; env['PYTHONUNBUFFERED']='1'
        cmd=[str(a.python),'-u','native/continue_real_pairs.py','--source-run',str(source),'--dataset',str(a.dataset),'--total-steps',str(end),'--output',str(output),'--augment-paired','--degradation-profile','anime-balanced','--loss','zenkai_frequency_edge','--batch-size','2','--accumulation','4']
        started=time.time(); result=subprocess.run(cmd,env=env,cwd=Path(__file__).resolve().parents[1]); row={'from':current,'to':end,'output':str(output),'exitCode':result.returncode,'durationSeconds':time.time()-started}; entries.append(row)
        status.write_text(json.dumps({'state':'running' if result.returncode==0 else 'HIP_INFRA_FAILURE','targetSteps':a.target_steps,'segments':entries},indent=2),encoding='utf-8')
        if result.returncode!=0: raise SystemExit(result.returncode)
        source=output
    status.write_text(json.dumps({'state':'complete','targetSteps':a.target_steps,'finalRun':str(source),'segments':entries},indent=2),encoding='utf-8'); print(json.dumps({'state':'complete','finalRun':str(source)}))
if __name__=='__main__': main()
