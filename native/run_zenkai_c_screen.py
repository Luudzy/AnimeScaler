"""C5k coordinator, evidence-gated faithful continuation with bounded HIP recovery."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from hip_operation import run_operation,save_json
from zenkai_c_screen import ROOT,RUN,SOURCE,validate_c_resume
from export_zenkai import sha256
GPU='F:/comfyui-rocm-071926/python_env/python.exe'


def main():
    scheduler=json.loads((ROOT/'diagnostics/zenkai-scheduler-v2-1k/pipeline.json').read_text())
    gate=json.loads((ROOT/'docs/zenkai-scheduler-decision.json').read_text())
    if scheduler['state']!='COMPLETE' or not gate.get('C5kAuthorizedByEvidence'):raise ValueError('Unfinished scheduler/recipe gate')
    folder=ROOT/'models/zenkai/experiments'/RUN
    if folder.exists():raise FileExistsError('Preserve existing C run')
    source=ROOT/'models/zenkai/experiments'/SOURCE/'step-3000-training-state.pt'
    pins=json.loads((source.parent/'metadata.json').read_text())['checkpointsSha256']
    if sha256(source)!=pins[source.name]:raise ValueError('Actual3k source hash changed')
    s=torch.load(source,map_location='cpu',weights_only=True);validate_c_resume(s,5000)
    master=torch.load(source.parent/'step-3000-master-fp32.pt',map_location='cpu',weights_only=True)
    if any(not torch.equal(v,s['model'][k]) for k,v in master['model'].items()):raise ValueError('3k master/state mismatch')
    choices=[]
    def command(a,out):
        path=source
        if folder.exists():
            index=json.loads((folder/'checkpoint-index.json').read_text())
            if index['lastCompleteStep']>3000:
                path=Path(index['lastCheckpoint'])
                if sha256(path)!=index['checkpoints'][str(path.relative_to(folder))]:raise ValueError('Recovery SHA mismatch')
        state=torch.load(path,map_location='cpu',weights_only=True);validate_c_resume(state,5000)
        choices.append({'attempt':a,'step':state['step'],'checkpoint':str(path),'sha256':sha256(path)})
        return [GPU,ROOT/'native/zenkai_c_screen.py','--checkpoint',path,'--attempt',str(a),'--target','5000']
    def validate(out):
        s=torch.load(folder/'step-5000-training-state.pt',map_location='cpu',weights_only=True);validate_c_resume(s,5000)
        m=torch.load(folder/'step-5000-master-fp32.pt',map_location='cpu',weights_only=True);r=json.loads((folder/'report.json').read_text())
        if r['stepsExecuted']!=5000 or r['optimizerUpdates']!=5000 or m['step']!=5000 or any(not torch.equal(v,s['model'][k]) for k,v in m['model'].items()):raise ValueError('Incomplete C5k')
        return {'run':RUN,'step':5000,'checkpoint':str(folder/'step-5000-master-fp32.pt'),
                'sha256':sha256(folder/'step-5000-master-fp32.pt'),'stateSha256':sha256(folder/'step-5000-training-state.pt')}
    base=ROOT/'diagnostics/zenkai-training/c5k-training'
    result=run_operation(base,command,validate,timeout=2400);result['resumeSelections']=choices;save_json(base/'operation.json',result)
    print(json.dumps({'state':result['state'],'run':RUN}))


if __name__=='__main__':main()
