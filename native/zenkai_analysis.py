"""Matched A/B analysis: retain regressions and scope Pareto to LPIPS/latency."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

ROOT=Path(__file__).resolve().parents[1]
MODELS=('zenkai-a','zenkai-b','adore-lite-v3','realesr-anime','hfa2k-cugan')
CASES=('clean','heavy','codec-intra-h264','codec-intra-h265','codec-intra-av1')
LOWER=('mae','edgeMae','laplacianMae','hfError','lpipsSqueeze')
HIGHER=('ssim','msSsim')


def validate_matched_configs(a,b):
    if (a.get('family')!='Zenkai' or b.get('family')!='Zenkai' or
            a.get('channels')!=32 or b.get('channels')!=32 or
            a.get('nativeScale')!=2 or b.get('nativeScale')!=2 or
            a.get('depth')!=5 or b.get('depth')!=6):
        raise ValueError('Expected paired Zenkai A/depth5 and B/depth6')
    if set(a)!=set(b):raise ValueError('A/B configuration fields differ')
    for key in a:
        if key not in ('run','depth','parameters') and a[key]!=b[key]:
            raise ValueError(f'Unmatched A/B field: {key}')
    return True


def finite(value,positive=False):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError('Non-finite or nonnumeric result')
    if value<0 or (positive and value==0):raise ValueError('Invalid negative/zero result')
    return value


def pareto_dominators(points):
    for point in points.values():finite(point['latency'],positive=True);finite(point['error'])
    return {name:sorted(other for other,item in points.items() if other!=name and
            item['latency']<=point['latency'] and item['error']<=point['error'] and
            (item['latency']<point['latency'] or item['error']<point['error']))
            for name,point in points.items()}


def analyze_results(quality,performance):
    if quality.get('state')!='complete' or performance.get('complete') is not True:
        raise ValueError('Quality/performance report is incomplete')
    if (quality.get('inferenceBackend')!='torch' or performance.get('backend')!='torch' or
            quality.get('backend')!=performance.get('provider') or
            quality.get('precision')!=performance.get('precision') or
            not quality.get('gpu') or quality['gpu']!=performance.get('gpu') or
            quality.get('hip')!=performance.get('hip') or quality.get('torch')!=performance.get('torch')):
        raise ValueError('Quality and latency used different or unrecorded execution stacks')
    if set(quality.get('models',{}))!=set(MODELS) or set(quality.get('cases',{}))!=set(CASES):
        raise ValueError('Expected five matched models and five quality cases')
    result={'schemaVersion':1,'decision':'REVIEW_REQUIRED','releaseApproved':False,
            'realVideoValidation':'REAL VIDEO VALIDATION NOT EXECUTED',
            'rankingBasis':'Equal-case mean LPIPS is a proxy for review order, not a release score',
            'scope':'Measured Torch model latency plus resize; not player FPS. Quality uses native x2 validation crops.',
            'models':{},'workloads':{}}
    for case in CASES:
        if type(quality.get('sampleCounts',{}).get(case)) is not int or quality['sampleCounts'][case]<1:
            raise ValueError('Missing quality sample count')
        for name in MODELS:
            scores=quality['cases'][case].get(name,{})
            for key in ('mae','edgeMae','lpipsSqueeze'):
                if key not in scores:raise ValueError('Incomplete quality metrics')
            for key in LOWER+HIGHER:
                if key in scores:finite(scores[key])
    for name in MODELS:
        regressions=[]
        for case in CASES:
            scores=quality['cases'][case][name];baseline=quality['cases'][case]['adore-lite-v3']
            for key in LOWER+HIGHER:
                if key not in scores or key not in baseline:continue
                delta=scores[key]-baseline[key]
                worse=delta if key in LOWER else -delta
                if worse>1e-9:
                    regressions.append({'case':case,'metric':key,'value':scores[key],
                        'baseline':baseline[key],'absoluteDelta':delta,
                        'relativeWorseningPct':100*worse/abs(baseline[key]) if abs(baseline[key])>1e-12 else None})
        result['models'][name]={'balancedMeanLpips':mean(quality['cases'][case][name]['lpipsSqueeze'] for case in CASES),
                               'qualityByCase':{case:quality['cases'][case][name] for case in CASES},
                               'regressionsVsV3':regressions}
    grouped={}
    for row in performance.get('cases',[]):
        name=row.get('model')
        if name not in MODELS:raise ValueError('Unexpected benchmark model')
        if row.get('artifactSha256')!=quality['models'][name].get('sha256'):
            raise ValueError(f'Quality/latency artifact mismatch: {name}')
        shape=tuple(row.get(key) for key in ('width','height','outputWidth','outputHeight'))
        if any(type(value) is not int or value<1 for value in shape):raise ValueError('Invalid benchmark shape')
        if name in grouped.setdefault(shape,{}):raise ValueError('Duplicated benchmark case')
        for key in ('medianMs','meanMs','p95Ms','p99Ms'):finite(row.get(key),positive=True)
        if row.get('warmup',0)<50 or row.get('iterations',0)<500:raise ValueError('Benchmark protocol is incomplete')
        grouped[shape][name]=row
    if not grouped:raise ValueError('No benchmark workloads')
    for shape,rows in grouped.items():
        if set(rows)!=set(MODELS):raise ValueError('Benchmark resolution is not paired across all models')
        points={name:{'latency':row['medianMs'],'error':result['models'][name]['balancedMeanLpips']} for name,row in rows.items()}
        dominated=pareto_dominators(points)
        result['workloads']['%dx%d-to-%dx%d'%shape]={name:{
            'medianMs':row['medianMs'],'meanMs':row['meanMs'],'p95Ms':row['p95Ms'],'p99Ms':row['p99Ms'],
            'modelOnlyMeanFps':1000/row['meanMs'],'peakAllocatedMiB':row.get('peakAllocatedMiB'),
            'balancedMeanLpips':points[name]['error'],'proxyDominatedBy':dominated[name]} for name,row in rows.items()}
    result['candidateReviewOrder']=sorted(('zenkai-a','zenkai-b'),key=lambda name:result['models'][name]['balancedMeanLpips'])
    result['pending']=['Blind visual review, halos and lineart','Real video temporal validation',
                       'End-to-end player throughput','Decision on 15k budget; no automatic training or promotion']
    return result


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_artifacts(report,output):
    output=Path(output).resolve()
    if not output.is_relative_to(ROOT) or output.exists():raise ValueError('Analysis output must be new and inside workspace')
    output.mkdir(parents=True)
    (output/'analysis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines=['# Zenkai A/B: quality and latency','','Decision: REVIEW REQUIRED. No model promoted.',
           '',report['scope'],'','LPIPS/latency dominance is a numeric proxy; every metric regression remains in analysis.json.']
    for workload,rows in report['workloads'].items():
        lines+=['',f'## {workload}','','| Model | Mean LPIPS | Median ms | P95 ms | Proxy dominated by |',
                '|---|---:|---:|---:|---|']
        lines.extend(f"| {name} | {row['balancedMeanLpips']:.5f} | {row['medianMs']:.2f} | {row['p95Ms']:.2f} | {', '.join(row['proxyDominatedBy']) or 'none'} |" for name,row in rows.items())
    lines+=['','## Regressions against V3','','| Model | Case | Metric | Value | V3 |','|---|---|---|---:|---:|']
    for name,model in report['models'].items():
        lines.extend(f"| {name} | {row['case']} | {row['metric']} | {row['value']:.6f} | {row['baseline']:.6f} |" for row in model['regressionsVsV3'])
    lines+=['','REAL VIDEO VALIDATION NOT EXECUTED.']
    (output/'analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    for workload,rows in report['workloads'].items():
        figure,axis=plt.subplots(figsize=(8,5))
        for index,(name,row) in enumerate(rows.items()):
            axis.scatter(row['medianMs'],row['balancedMeanLpips'],s=65,marker='x' if row['proxyDominatedBy'] else 'o')
            axis.annotate(name,(row['medianMs'],row['balancedMeanLpips']),xytext=(6,6+index%2*8),textcoords='offset points',fontsize=8)
        axis.set_xscale('log');axis.margins(.3)
        axis.set_xlabel('Model + resize median latency (ms; lower is better)')
        axis.set_ylabel('Equal-case mean LPIPS (proxy; lower is better)')
        axis.set_title(workload+'\nMatched ROCm FP16; no player FPS or release decision',fontsize=11)
        axis.grid(True,alpha=.25);figure.tight_layout()
        figure.savefig(output/(workload+'.png'),dpi=160);figure.savefig(output/(workload+'.svg'))
        plt.close(figure)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'diagnostics/zenkai-ab-5k')
    parser.add_argument('--output',type=Path,default=ROOT/'diagnostics/zenkai-ab-5k/analysis')
    parser.add_argument('--runtime-a',type=Path,help='Explicit approved retry report; original failure stays preserved')
    args=parser.parse_args()
    quality_path=args.input/'quality/quality.json';performance_path=args.input/'performance/benchmark_results.json'
    quality=json.loads(quality_path.read_text(encoding='utf-8'));performance=json.loads(performance_path.read_text(encoding='utf-8'))
    configs=[];exports={}
    for letter in ('a','b'):
        folder=ROOT/f'models/zenkai/experiments/zenkai-lite-v1-{letter}-5k'
        config=json.loads((folder/'config.json').read_text(encoding='utf-8'));configs.append(config)
        trained=json.loads((folder/'report.json').read_text(encoding='utf-8'))
        if trained['stepsExecuted']!=5000:raise ValueError('A/B training is incomplete')
        export_path=ROOT/f'models/zenkai/exports/{letter}-5k/export-report.json'
        exported=json.loads(export_path.read_text(encoding='utf-8'))
        runtime_path=args.runtime_a if letter=='a' and args.runtime_a else args.input/f'runtime-{letter}.json'
        runtime=json.loads(runtime_path.read_text(encoding='utf-8'))
        if (exported.get('state')!='complete' or exported.get('parityApproved') is not True or exported['training']!=config or
                runtime.get('parityApproved') is not True or runtime.get('state')!='complete' or
                runtime['checkpointSha256']!=exported['checkpointSha256'] or runtime['exportReportSha256']!=_sha(export_path) or
                runtime.get('device')!='cuda' or runtime.get('gpu')!=quality.get('gpu') or
                runtime.get('hip')!=quality.get('hip') or runtime.get('torch')!=quality.get('torch') or
                exported['artifacts']['deploy-fp32.pt']['sha256']!=quality['models'][f'zenkai-{letter}']['sha256'] or
                runtime.get('artifact')!=quality['models'][f'zenkai-{letter}']):
            raise ValueError('Missing or mismatched numeric export/runtime gate')
        if (quality['sourceManifestSha256']!=config['datasetManifestSha256'] or
                quality['validationManifestSha256']!=config['fixedValidationSha256']):
            raise ValueError('Candidate validation input provenance changed')
        exports[letter]={'step':exported['step'],'checkpointSha256':exported['checkpointSha256'],
                         'trainingBudget':5000,'exportReportSha256':_sha(export_path)}
    validate_matched_configs(*configs)
    for name,artifact in quality['models'].items():
        if _sha(artifact['path'])!=artifact['sha256']:raise ValueError(f'Quality artifact changed: {name}')
    report=analyze_results(quality,performance)
    report.update(exports=exports,qualityReportSha256=_sha(quality_path),performanceReportSha256=_sha(performance_path),
                  sampleCounts=quality['sampleCounts'],codecManifestSha256=quality.get('codecManifestSha256'))
    temporal_path=args.input/'synthetic-temporal.json'
    if temporal_path.is_file():
        temporal=json.loads(temporal_path.read_text(encoding='utf-8'))
        if temporal.get('models')!=quality['models'] or temporal.get('fixedManifestSha256')!=quality['validationManifestSha256']:
            raise ValueError('Synthetic temporal provenance does not match quality')
        report['syntheticTemporal']=temporal
    write_artifacts(report,args.output)
    print(json.dumps({'decision':report['decision'],'output':str(args.output),'releaseApproved':False}))


if __name__=='__main__':main()
