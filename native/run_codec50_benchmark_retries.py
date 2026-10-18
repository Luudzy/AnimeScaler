"""Continue the existing failed Codec50 benchmark with at most two clean attempts."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from hip_operation import run_operation,save_json

ROOT=Path(__file__).resolve().parents[1]
GPU=Path('F:/comfyui-rocm-071926/python_env/python.exe')
RUN='zenkai-ablation-skip-factor-codec50-1k'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--final-attempt',action='store_true')
    args=parser.parse_args()
    base=ROOT/'diagnostics/zenkai-ablations'/('codec50-1k-clean-benchmark-final' if args.final_attempt else 'codec50-1k-clean-benchmark')
    prior=None
    if args.final_attempt:
        prior=json.loads((ROOT/'diagnostics/zenkai-ablations/codec50-1k-clean-benchmark/operation.json').read_text())
        if [a['attempt'] for a in prior['attempts']]!=[2] or prior['state']!='FAIL':raise ValueError('Only final third attempt remains')
    original=ROOT/'diagnostics/zenkai-training/factor-performance-1k-codec50.exit.json'
    original_error=original.with_name('factor-performance-1k-codec50.stderr.log')
    raw=original_error.read_bytes()
    text=raw.decode('utf-16' if raw.startswith(b'\xff\xfe') else 'utf-8-sig',errors='replace')
    if json.loads(original.read_text(encoding='utf-8-sig'))['exitCode']==0 or 'unspecified launch failure' not in text:
        raise ValueError('Expected original HIP infrastructure failure')
    checkpoint=ROOT/'models/zenkai/experiments'/RUN/'step-1000-master-fp32.pt'
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    pins=json.loads((checkpoint.parent/'metadata.json').read_text())['checkpointsSha256']
    if digest!=pins[checkpoint.name]:raise ValueError('Checkpoint hash changed')
    def command(attempt,out):
        return [GPU,ROOT/'native/benchmark_sr_models.py','--probe-checkpoint',checkpoint,'--probe-step','1000',
                '--model',RUN,'--output',out/'benchmark','--warmup','50','--iterations','500']
    def validate(out):
        path=out/'benchmark/benchmark_results.json';report=json.loads(path.read_text())
        if not report.get('complete') or len(report['cases'])!=2:raise ValueError('Incomplete benchmark')
        expected=[(640,360,1280,720),(1280,720,1920,1080)]
        for row,shape in zip(report['cases'],expected):
            if tuple(row[k] for k in ('width','height','outputWidth','outputHeight'))!=shape:raise ValueError('Shape changed')
            if row['model']!=RUN or row['artifactSha256']!=digest or row['iterations']!=500 or row['warmup']!=50:
                raise ValueError('Protocol/artifact changed')
            for key in ('meanMs','medianMs','p50Ms','p95Ms','p99Ms','minMs','maxMs'):
                if row[key]<=0:raise ValueError('Invalid timing')
        if report['precision']!='fp16' or report['provider']!='ROCm':raise ValueError('Backend changed')
        return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'caseCount':2}
    result=run_operation(base,command,validate,start_attempt=3 if args.final_attempt else 2)
    if prior is not None:result['previousAttemptRecords']=prior['attempts']
    result['originalAttempt']={'attempt':1,'state':'HIP_INFRA_FAILURE','exitCode':1,'path':str(original),
                               'stderrSha256':hashlib.sha256(raw).hexdigest()}
    save_json(base/'operation.json',result)
    print(json.dumps({'state':result['state'],'selectedAttempt':result['selectedAttempt'],'scope':result['scope']}))


if __name__=='__main__':main()
