"""Report only selected complete attempts, retaining absent measurements explicitly."""
import datetime
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'native'))
from frozen_video_protocol import validate_protocol

BASE=ROOT/'diagnostics/zenkai-codec-pair-3k'
RUNS={'codec25':'zenkai-ablation-spab-identity-skip-3k-continuation',
      'codec50':'zenkai-ablation-skip-factor-codec50-3k-continuation'}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pct(value,reference):return 'UNDEFINED' if reference==0 else f'{100*(value/reference-1):+.5f}'
def table(headers,rows):
    return ['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+[
        '| '+' | '.join(map(str,row))+' |' for row in rows]


def selected(operation):
    if operation['state']!='PASS':return None
    attempt=operation['selectedAttempt']
    rows=[r for r in operation['attempts'] if r['attempt']==attempt]
    if len(rows)!=1 or rows[0]['state']!='PASS' or rows[0]['exitCode']!=0:
        raise ValueError('Selected attempt is not a complete successful process')
    out=Path(rows[0]['output']).resolve()
    result=operation['result']
    paths=[Path(v).resolve() for k,v in result.items() if k in ('path','quality','parity')]
    if not paths or any(not p.is_relative_to(out) or not p.is_file() for p in paths):
        raise ValueError('Result does not belong to selected attempt')
    if 'sha256' in result and sha(result['path'])!=result['sha256']:raise ValueError('Selected result mutated')
    return result


def main():
    pipeline=read(BASE/'pipeline.json');training=read(ROOT/'diagnostics/zenkai-training/codec50-3k-training/operation.json')
    if pipeline['state']!='COMPLETE' or training['state']!='PASS':raise ValueError('Training/postprocessing unfinished')
    results={}
    for name,operation in pipeline['results'].items():
        record=pipeline.get('operationRecords',{}).get(name,BASE/name/'operation.json')
        if operation!=read(record):raise ValueError('Operation record changed')
        results[name]=selected(operation)
    report={'generatedUTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'state':'COMPLETE','training':training,'operations':pipeline['results'],
            'runs':RUNS,'quality':None,'performance':{},'video':{},'gradients':None,
            'releaseApproved':False,'C5kStarted':False,'Stage2Started':False}
    lines=['# Zenkai Codec25 versus Codec50 at 3,000 total steps','',
           'Generated UTC: '+report['generatedUTC'],'',
           'Same seed929, LR128 crops, batch4×2, frequency loss and original5k cosine timeline. Codec50 faithfully continued its intact1k state; Codec25 is the preserved3k continuation. Only codec fraction differs (.25/.50). No combined recipe, hidden holdout, product promotion, C5k or Stage2.','',
           'Operation PASS means a complete valid measurement. Quality advancement is a separate criterion. HIP_INFRA_FAILURE invalidates only the interrupted operation; incomplete attempts supply no table entries.','',
           '## Operations','']
    lines+=table(['Operation','State','Selected attempt','Attempt states'],[
        [name,op['state'],op['selectedAttempt'],', '.join(str(r['attempt'])+':'+r['state'] for r in op['attempts'])]
        for name,op in pipeline['results'].items()])
    rows=[]
    for label,run in RUNS.items():
        folder=ROOT/'models/zenkai/experiments'/run;r=read(folder/'report.json');c=read(folder/'config.json')
        if r['stepsExecuted']!=3000 or r['optimizerUpdates']!=3000:raise ValueError('Wrong budget or skipped updates')
        archive=folder/'source';sources={}
        for name,pin in c['sourceFilesSha256'].items():
            direct=archive/name
            if direct.is_file():
                if sha(direct)!=pin:raise ValueError('Archived source changed')
                sources[name]={'state':'HASH VERIFIED','path':str(direct),'sha256':pin}
            else:
                candidates=list((ROOT/'models/zenkai/experiments').glob('*/source/'+name))+[ROOT/'native'/name]
                matches=[p for p in candidates if p.is_file() and sha(p)==pin]
                sources[name]={'state':'HASH VERIFIED EXISTING COPY' if matches else 'ARCHIVE NOT AVAILABLE',
                               'path':str(matches[0]) if matches else None,'sha256':pin}
        record={'config':c,'training':r,'sourceIntegrity':sources,
                'checkpointsSha256':{p.name:sha(p) for p in folder.glob('*.pt')}}
        report[label]=record
        rows.append([label,3000,c['codecFraction'],r['trainingSeconds'],r['best'],c['gitCommit']])
    lines+=['','## Training','']+table(['Candidate','Steps/updates','Codec fraction','Training seconds','Best validation','Git'],rows)
    for label in RUNS:
        missing=[name for name,r in report[label]['sourceIntegrity'].items() if r['state']=='ARCHIVE NOT AVAILABLE']
        if missing:lines+=['',label+' source limitation: original hash recorded but matching archived bytes unavailable for '+', '.join(missing)+'. No source reconstruction or claim of full archived-source reproducibility. Checkpoints and saved full training state remain independently hash verified.']
    old=read(ROOT/'diagnostics/zenkai-ablations/factor-screens-1k-partial-quality/quality.json')
    if results['quality']:
        q=read(results['quality']['quality']);p=read(results['quality']['parity'])
        if q['state']!='complete' or sum(q['sampleCounts'].values())!=160:raise ValueError('Wrong fixed count')
        for key in ('indices','validationManifestSha256','sourceManifestSha256','codecManifestSha256'):
            if q[key]!=old[key]:raise ValueError('Fixed protocol differs: '+key)
        if not all(v['parityApproved'] for v in p['models'].values()):raise ValueError('Parity failed')
        report['quality']=q;report['parity']=p;rows=[];budget=[]
        for domain,models in q['cases'].items():
            for metric,v3 in models['adore-lite-v3'].items():
                a=models['skip-codec25-3k'][metric];b=models['skip-codec50-3k'][metric]
                rows.append([domain,metric,f'{v3:.9f}',f'{a:.9f}',f'{b:.9f}',pct(a,v3),pct(b,v3),pct(b,a)])
                for label,key in (('codec25','skip-frequency-1k'),('codec50','factor-codec50-1k')):
                    start=old['cases'][domain][key][metric];end=a if label=='codec25' else b
                    budget.append([label,domain,metric,f'{start:.9f}',f'{end:.9f}',pct(end,start)])
        lines+=['','## Fixed validation: every individual metric','',
                'MAE/MSE/LPIPS/Edge/Laplacian/HF/DCT lower is better; PSNR/SSIM/MS-SSIM higher is better. Percent changes are signed arithmetic differences; PSNR is in dB. No composite determines the winner. Domains are not separately labeled anime/illustration by this manifest.','']
        lines+=table(['Domain','Metric','HistoricalV3','Codec25','Codec50','25/V3 %','50/V3 %','50/25 %'],rows)
        lines+=['','## 1k to 3k trajectories','',
                'Budget and scheduled learning rate advance together; this is not a pure budget-only causal test.','']
        lines+=table(['Candidate','Domain','Metric','1k','3k','3k/1k %'],budget)
        rows=[]
        for model,r in p['models'].items():
            for comparison in ('fusionFp32','deployFp16'):
                rows.append([model,comparison,max(c[comparison]['max_abs_error'] for c in r['cases']),
                             max(c[comparison]['mean_abs_error'] for c in r['cases'])])
        lines+=['','## ROCm numerical parity','']+table(['Model','Comparison','Worst max_abs_error','Worst mean_abs_error'],rows)
    else:lines+=['','Fixed quality/parity: NOT MEASURED; see operation record.']
    perf=[]
    for label in RUNS:
        result=results['benchmark-'+label]
        if not result:continue
        r=read(result['path'])
        if not r['complete'] or r['provider']!='ROCm' or r['precision']!='fp16':raise ValueError('Benchmark protocol changed')
        report['performance'][label]=r
        for c in r['cases']:
            if c['warmup']!=50 or c['iterations']!=500 or sha(c['artifactPath'])!=c['artifactSha256']:raise ValueError('Incomplete/mutated benchmark')
            if (c['width'],c['height'],c['outputWidth'],c['outputHeight']) not in ((640,360,1280,720),(1280,720,1920,1080)):raise ValueError('Benchmark shapes changed')
            for component,values in c['components'].items():
                shape=f"{c['width']}×{c['height']}→{c['outputWidth']}×{c['outputHeight']}"
                if 'medianMs' not in values:
                    perf.append([c['model'],shape,component,'NOT APPLICABLE']+['—']*8);continue
                perf.append([c['model'],shape,component,*[f'{values[k]:.6f}' for k in
                    ('meanMs','medianMs','p50Ms','p95Ms','p99Ms','minMs','maxMs','fps')],f"{c['peakAllocatedMiB']:.3f}"])
    lines+=['','## Performance','',
            'ROCm FP16,50 warmups and500 distinct inputs. Model inference, post-resize and total are independently measured complete passes. FPS equivalent is1000/mean ms; this is not player FPS. Peak memory is Torch allocated memory, not whole-system VRAM. Missing operations remain NOT MEASURED.','']
    lines+=table(['Model','Input→output','Component','Mean ms','Median ms','P50','P95','P99','Min','Max','FPS equivalent','Peak MiB'],perf)
    for label in RUNS:
        if not results['benchmark-'+label]:lines+=['',label+' latency: NOT MEASURED — '+pipeline['results']['benchmark-'+label].get('reason','see operation record')]
    a=read(ROOT/'diagnostics/zenkai-real-video-skip-3k-complete/report.json');validate_protocol(a)
    report['video']['codec25']=a;video=[]
    if results['video-codec50']:
        b=read(results['video-codec50']['path']);validate_protocol(b)
        if a['backend']!=b['backend'] or a['precision']!=b['precision'] or a['cutPolicy']!=b['cutPolicy']:raise ValueError('Video protocol changed')
        report['video']['codec50']=b
        for x,y in zip(a['sequences'],b['sequences']):
            for field in ('manifestSha256','frameCount','suspectedCutTransitions'):
                if x[field]!=y[field]:raise ValueError('Video inputs changed')
            for metric in ('mae','referenceCompensatedTemporalMae','allTransitionTemporalMae'):
                v3=x['models']['adore-lite-v3'][metric]
                if abs(v3-y['models']['adore-lite-v3'][metric])>1e-9:raise ValueError('V3 video baseline changed')
                va=x['models'][RUNS['codec25']][metric];vb=y['models'][RUNS['codec50']][metric]
                video.append([Path(x['manifest']).parent.name,metric,f'{v3:.9f}',f'{va:.9f}',f'{vb:.9f}',pct(va,v3),pct(vb,v3),pct(vb,va)])
        for label,r in report['video'].items():
            if r['artifacts'][RUNS[label]]['sha256']!=report[label]['checkpointsSha256']['step-3000-master-fp32.pt']:raise ValueError('Video artifact changed')
    lines+=['','## Same 144 real frames','',
            'Three48-frame360→720 clips, CPU FP32 deploy, same compressed WebRip reference and cut policy. These pixel-delta metrics do not certify perceptual flicker or motion-compensated consistency. All-transition values preserve the excluded suspected cut.','']
    lines+=table(['Clip','Metric (lower better)','V3','Codec25','Codec50','25/V3 %','50/V3 %','50/25 %'],video)
    if results['gradients']:
        g=read(results['gradients']['path']);report['gradients']=g
        lines+=['','## Gradient diagnostics','',g['scope'],'',
                'Full block/case FP32 and scaled AMP gradient norms, accumulated weight delta and ratios are in the selected gradients.json. A small norm alone does not prove a dead block. No checkpoint weights were modified by this diagnostic.']
    else:lines+=['','Gradients: NOT MEASURED; see operation record.']
    lines+=['','## Decision','',
            'Read the separate decision report after inspecting these individual metrics and matched crops. No C5k/Stage2/default selection is performed by this builder. HistoricalV3 has a different training budget; this is a product baseline comparison, not a matched-budget proof of architecture superiority.']
    (BASE/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (ROOT/'docs/zenkai-codec3k-results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'state':'COMPLETE','report':str(BASE/'comparison.json')}))


if __name__=='__main__':main()
