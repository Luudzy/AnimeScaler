"""Single serial training operation with hash-checked faithful clean-process HIP recovery."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from hip_operation import run_operation,save_json
from zenkai_codec_resume import ROOT,SOURCE,RUN,validate_codec_resume,_sha
import torch


def main():
    base=ROOT/'diagnostics/zenkai-training/codec50-3k-training'
    output=ROOT/'models/zenkai/experiments'/RUN
    origin=ROOT/'models/zenkai/experiments'/SOURCE
    if output.exists():raise FileExistsError('Reserved continuation already exists; no blind restart')
    if base.exists():raise FileExistsError('Preserve prior training attempts')
    initial=origin/'step-1000-training-state.pt'
    pin=json.loads((origin/'metadata.json').read_text())['checkpointsSha256'][initial.name]
    if _sha(initial)!=pin:raise ValueError('Original state changed')
    validate_codec_resume(torch.load(initial,map_location='cpu',weights_only=True),RUN,3000)
    selected=[]
    def command(attempt,out):
        path=initial;step=1000
        if attempt>1:
            index=json.loads((output/'checkpoint-index.json').read_text()) if output.exists() else None
            if index and index.get('lastCheckpoint'):
                path=Path(index['lastCheckpoint']);digest=index['checkpoints'][str(path.relative_to(output))]
                if _sha(path)!=digest:raise ValueError('Recovery checkpoint corrupt; no resume')
                saved=torch.load(path,map_location='cpu',weights_only=True)
                validate_codec_resume(saved,RUN,3000);step=saved['step']
            # An intact initial1k state is a real checkpoint, not a fabricated reset.
        selected.append({'attempt':attempt,'resumeCheckpoint':str(path),'resumeStep':step,'sha256':_sha(path)})
        return ['F:/comfyui-rocm-071926/python_env/python.exe',ROOT/'native/zenkai_codec_resume.py',
                '--checkpoint',path,'--attempt',str(attempt),'--target','3000']
    def validate(out):
        report=json.loads((output/'report.json').read_text())
        if report['stepsExecuted']!=3000 or report['optimizerUpdates']!=3000:raise ValueError('Training incomplete')
        state=output/'step-3000-training-state.pt';saved=torch.load(state,map_location='cpu',weights_only=True)
        validate_codec_resume(saved,RUN,3000)
        master=output/'step-3000-master-fp32.pt'
        checkpoint=torch.load(master,map_location='cpu',weights_only=True)
        if any(not torch.equal(v,saved['model'][k]) for k,v in checkpoint['model'].items()):raise ValueError('Final master/state differ')
        return {'step':3000,'checkpoint':str(master),'sha256':_sha(master),'stateSha256':_sha(state)}
    result=run_operation(base,command,validate,timeout=3600)
    result['resumeSelections']=selected
    result['sourceCheckpointSha256']=pin
    save_json(base/'operation.json',result)
    print(json.dumps({'state':result['state'],'run':RUN,'selectedAttempt':result['selectedAttempt']}))


if __name__=='__main__':main()
