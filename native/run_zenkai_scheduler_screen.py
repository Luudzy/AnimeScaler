"""Fresh scheduler screening coordinator; bounded faithful recovery only."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from hip_operation import run_operation
from zenkai_scheduler_screen import ROOT,RUN,SOURCE,validate_snapshot
from export_zenkai import sha256
GPU='F:/comfyui-rocm-071926/python_env/python.exe'


def main():
    exports=json.loads((ROOT/'diagnostics/zenkai-codec-pair-3k-exports-recovery1/pipeline.json').read_text())
    pair=json.loads((ROOT/'diagnostics/zenkai-codec-pair-3k/pipeline.json').read_text())
    if exports['state']!='COMPLETE' or pair['state']!='COMPLETE':raise ValueError('Prior serial operations still active')
    folder=ROOT/'models/zenkai/experiments'/RUN
    if folder.exists():raise FileExistsError('Preserve existing new run')
    initial=ROOT/'models/zenkai/experiments'/SOURCE/'initial-training-state.pt'
    pins=json.loads((initial.parent/'metadata.json').read_text())['checkpointsSha256']
    if sha256(initial)!=pins[initial.name]:raise ValueError('Initial hash changed')
    validate_snapshot(torch.load(initial,map_location='cpu',weights_only=True))
    choices=[]
    def command(a,out):
        path=initial
        if folder.exists():
            index=json.loads((folder/'checkpoint-index.json').read_text())
            if index['lastCompleteStep']:
                path=Path(index['lastCheckpoint'])
                if sha256(path)!=index['checkpoints'][str(path.relative_to(folder))]:raise ValueError('Recovery SHA mismatch')
        state=torch.load(path,map_location='cpu',weights_only=True);validate_snapshot(state)
        choices.append({'attempt':a,'step':state['step'],'sha256':sha256(path),'checkpoint':str(path)})
        return [GPU,ROOT/'native/zenkai_scheduler_screen.py','--checkpoint',path,'--attempt',str(a)]
    def validate(out):
        state=folder/'step-1000-training-state.pt';master=folder/'step-1000-master-fp32.pt'
        s=torch.load(state,map_location='cpu',weights_only=True);validate_snapshot(s)
        m=torch.load(master,map_location='cpu',weights_only=True);r=json.loads((folder/'report.json').read_text())
        if r['stepsExecuted']!=r['optimizerUpdates'] or r['stepsExecuted']!=1000 or m['step']!=1000 or any(
                not torch.equal(v,s['model'][k]) for k,v in m['model'].items()):raise ValueError('Incomplete final scheduler state')
        return {'run':RUN,'step':1000,'checkpoint':str(master),'sha256':sha256(master),'stateSha256':sha256(state)}
    base=ROOT/'diagnostics/zenkai-training/scheduler-v2-1k-training'
    result=run_operation(base,command,validate,timeout=1800)
    result['resumeSelections']=choices
    from hip_operation import save_json
    save_json(base/'operation.json',result)
    print(json.dumps({'state':result['state'],'run':RUN}))


if __name__=='__main__':main()
