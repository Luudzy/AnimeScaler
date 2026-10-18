"""Single-GPU V2 coordinator; up to two clean HIP retries from intact real states only."""
import json
from pathlib import Path
import sys
import torch
from hip_operation import run_operation,save_json
from zenkai_v2_dataset_protocol import ROOT,RUN,SOURCE,validate_snapshot
from export_zenkai import sha256

GPU='F:/comfyui-rocm-071926/python_env/python.exe'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    if read(ROOT/'diagnostics/zenkai-c5k/pipeline.json')['state']!='COMPLETE':raise ValueError('Previous GPU operations still active')
    folder=ROOT/'models/zenkai/experiments'/RUN
    if folder.exists():raise FileExistsError('Preserve existing V2 experiment')
    initial=ROOT/'models/zenkai/experiments'/SOURCE/'initial-training-state.pt'
    pin=read(initial.parent/'metadata.json')['checkpointsSha256'][initial.name]
    if sha256(initial)!=pin:raise ValueError('Original initialization SHA changed')
    validate_snapshot(torch.load(initial,map_location='cpu',weights_only=True))
    choices=[]
    def command(attempt,out):
        path=initial
        if folder.exists():
            index=read(folder/'checkpoint-index.json')
            path=Path(index['lastCheckpoint']) if index['lastCompleteStep'] else folder/'initial-training-state.pt'
            expected=index['checkpoints'][str(path.relative_to(folder))] if index['lastCompleteStep'] else index['initialSha256']
            if sha256(path)!=expected:raise ValueError('Recovery SHA mismatch')
        state=torch.load(path,map_location='cpu',weights_only=True);validate_snapshot(state)
        choices.append({'attempt':attempt,'step':state['step'],'checkpoint':str(path),'sha256':sha256(path)})
        return [GPU,ROOT/'native/zenkai_v2_dataset_screen.py','--checkpoint',path,'--attempt',str(attempt)]
    def validate(out):
        full=folder/'step-1000-training-state.pt';master=folder/'step-1000-master-fp32.pt'
        state=torch.load(full,map_location='cpu',weights_only=True);validate_snapshot(state)
        saved=torch.load(master,map_location='cpu',weights_only=True);report=read(folder/'report.json')
        if report['stepsExecuted']!=1000 or report['optimizerUpdates']!=1000 or state['step']!=1000 or saved['step']!=1000:
            raise ValueError('Incomplete V2 budget')
        if any(not torch.equal(v,state['model'][k]) for k,v in saved['model'].items()):raise ValueError('Master/full state differ')
        if sum(report['sampledTrainDomainCounts'].values())!=8000 or report['samples']!=8000:raise ValueError('Sample count mismatch')
        return {'run':RUN,'step':1000,'checkpoint':str(master),'sha256':sha256(master),'stateSha256':sha256(full)}
    base=ROOT/'diagnostics/zenkai-training/v2-dataset-1k-training'
    result=run_operation(base,command,validate,timeout=1800);result['resumeSelections']=choices
    save_json(base/'operation.json',result)
    print(json.dumps({'state':result['state'],'run':RUN}))


if __name__=='__main__':main()
