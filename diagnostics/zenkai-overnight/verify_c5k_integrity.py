"""CPU-only immutable full-state and preservation checks after C training."""
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'native'))
import torch
from zenkai_c_screen import validate_c_resume


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    output = ROOT/'diagnostics/zenkai-c5k/integrity.json'
    if output.exists():
        raise FileExistsError('Preserve previous audit')
    op = read(ROOT/'diagnostics/zenkai-training/c5k-training/operation.json')
    if op['state'] != 'PASS':
        raise ValueError('Training did not pass')
    root = ROOT/'models/zenkai/experiments/zenkai-lite-v1-c-codec50-5k'
    config = read(root/'config.json')
    index = read(root/'checkpoint-index.json')
    if index['lastCompleteStep'] != 5000 or len(index['checkpoints']) != 20:
        raise ValueError('Incomplete100-step recovery history')
    checked = {}
    for name, pin in index['checkpoints'].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or sha(path) != pin:
            raise ValueError('Recovery state hash mismatch')
        checked[name] = pin
    master = root/'step-5000-master-fp32.pt'
    full = root/'step-5000-training-state.pt'
    if sha(master) != op['result']['sha256'] or sha(full) != op['result']['stateSha256']:
        raise ValueError('Final hashes differ from complete operation')
    state = torch.load(full, map_location='cpu', weights_only=True)
    validate_c_resume(state, 5000)
    saved = torch.load(master, map_location='cpu', weights_only=True)
    best = torch.load(root/'best-composite-master-fp32.pt', map_location='cpu', weights_only=True)
    if state['config'] != config or saved['config'] != config or saved['step'] != 5000:
        raise ValueError('Final config/step changed')
    if best['step'] != state['resumeProgress']['best']['step']:
        raise ValueError('Best checkpoint does not match recorded selection')
    if set(saved['model']) != set(state['model']) or any(not torch.equal(v, state['model'][k]) for k,v in saved['model'].items()):
        raise ValueError('Master and complete state disagree')
    if best['step'] == 5000 and any(not torch.equal(v, best['model'][k]) for k,v in saved['model'].items()):
        raise ValueError('Best/final tensors disagree')
    source = ROOT/'models/zenkai/experiments'/config['cSourceRun']/'step-3000-training-state.pt'
    if sha(source) != config['cSourceSha256']:
        raise ValueError('Original3k state changed')
    for name, pin in config['sourceFilesSha256'].items():
        if sha(root/'source'/name) != pin:
            raise ValueError('Archived mathematical source changed')
    if sha(root/'source/zenkai_c_screen.py') != config['cResumeSourceSha256']:
        raise ValueError('Archived C source changed')
    original = ROOT/'native/training-data/zenkai/manifest.jsonl'
    if sha(original) != config['datasetManifestSha256']:
        raise ValueError('Original TRAIN manifest changed')
    output.write_text(json.dumps({'state':'PASS', 'finalStep':5000, 'optimizerSteps':5000,
        'recoveryStatesVerified':checked, 'masterSha256':sha(master), 'fullStateSha256':sha(full),
        'source3kPreserved':True, 'source3kSha256':sha(source), 'bestStep':best['step'],
        'bestMatchesFinalTensors':best['step']==5000, 'archivedSourceHashesVerified':True,
        'originalDatasetPreserved':True, 'finalHiddenOpened':False},indent=2),encoding='utf-8')
    print('PASS:C5000 full state,20 recovery hashes,source/config/master/best preservation')


if __name__ == '__main__':
    main()
