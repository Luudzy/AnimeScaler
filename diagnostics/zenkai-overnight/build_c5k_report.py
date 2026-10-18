"""Consolidate complete C operations and preserve every individual regression."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'native'))
from frozen_video_protocol import validate_protocol
from build_codec3k_report import selected,table,pct
BASE=ROOT/'diagnostics/zenkai-c5k';RUN='zenkai-lite-v1-c-codec50-5k'


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    pipeline=read(BASE/'pipeline.json');training=read(ROOT/'diagnostics/zenkai-training/c5k-training/operation.json')
    if pipeline['state']!='COMPLETE' or training['state']!='PASS':raise ValueError('C screening unfinished')
    results={k:selected(v) for k,v in pipeline['results'].items() if k!='export'}
    export=pipeline['results'].get('export');results['export']=None
    if export and export['state']=='PASS':
        attempts=[a for a in export['attempts'] if a['attempt']==export['selectedAttempt']]
        expected=ROOT/'models/zenkai/exports'/(RUN+'-export-attempt-'+str(export['selectedAttempt']))
        if len(attempts)!=1 or attempts[0]['state']!='PASS' or attempts[0]['exitCode']!=0 or Path(export['result']['folder'])!=expected:
            raise ValueError('Export is not the selected complete attempt')
        results['export']=export['result']
    folder=ROOT/'models/zenkai/experiments'/RUN;cfg=read(folder/'config.json');train=read(folder/'report.json')
    if train['stepsExecuted']!=5000 or train['optimizerUpdates']!=5000:raise ValueError('Incomplete actual C budget')
    for name,pin in cfg['sourceFilesSha256'].items():
        if sha(folder/'source'/name)!=pin:raise ValueError('Archived source changed')
    if sha(folder/'source/zenkai_c_screen.py')!=cfg['cResumeSourceSha256']:raise ValueError('Archived C continuation changed')
    pins={p.name:sha(p) for p in folder.glob('*.pt')}
    old=read(ROOT/'diagnostics/zenkai-codec-pair-3k/comparison.json')
    report={'state':'COMPLETE','run':RUN,'trainingOperation':training,'config':cfg,'training':train,
        'checkpointsSha256':pins,'operations':pipeline['results'],'quality':None,'video':None,'performance':None,
        'gradients':None,'export':None,'releaseApproved':False,'promoted':False,'Stage2Started':False,
        'scope':'Exploratory exact5000-total-step screening; no player FPS or hidden holdout',
        'videoIdentityRepair':pipeline.get('videoIdentityRepair')}
    lines=['# Zenkai C5k: individual measured results','',
           'Frozen Skip+Frequency+Codec50+original cosine. Faithful actual3000→5000 continuation in a new run, complete optimizer/scaler/RNG/datasetRNG/sampler/scheduler. No mixed factor, hidden holdout or automatic15k.','',
           f"Final5000 updates, selected training attempt{training['selectedAttempt']}, cumulative training seconds{train['trainingSeconds']:.3f}, best{train['best']}. Final training loss{train['history'][-1]['trainingLoss']:.9f}. Full config/history/hashes are in the run metadata.",'']
    lines+=table(['Operation','State','Selected attempt'],[[k,v['state'],v['selectedAttempt']] for k,v in pipeline['results'].items()])
    if pipeline.get('videoIdentityRepair'):
        lines+=['','Video identity repair: '+pipeline['videoIdentityRepair'],
                'Original pipeline/report/logs remain archived; tables below use only the new complete selected CPU video operation.','']
    if results['quality']:
        q=read(results['quality']['quality']);p=read(results['quality']['parity']);report.update(quality=q,parity=p);rows=[]
        for key in ('indices','validationManifestSha256','sourceManifestSha256','codecManifestSha256'):
            if q[key]!=old['quality'][key]:raise ValueError('Fixed inputs changed')
        for domain,m in q['cases'].items():
            for metric,v3 in m['adore-lite-v3'].items():
                a=old['quality']['cases'][domain]['skip-codec50-3k'][metric];b=m['zenkai-c-5k'][metric]
                rows.append([domain,metric,f'{v3:.9f}',f'{a:.9f}',f'{b:.9f}',pct(b,a),pct(b,v3)])
        lines+=['','## Same160 fixed samples','',
                'MAE/MSE/Edge/Laplacian/HF/DCT/LPIPS lower is better; PSNR/SSIM/MS-SSIM higher is better. PSNR is in dB. No composite hides regressions. Budget and learning rate advance together, not a pure budget-only causal attribution.','']
        lines+=table(['Domain','Metric','HistoricalV3','Codec50 3k','C5k','C/3k %','C/V3 %'],rows)
        rows=[]
        for name,r in p['models'].items():
            for key in ('fusionFp32','deployFp16'):
                rows.append([name,key,max(c[key]['max_abs_error'] for c in r['cases']),max(c[key]['mean_abs_error'] for c in r['cases'])])
        lines+=['','## ROCm model fusion/FP16 parity','']+table(['Model','Comparison','Worst max_abs_error','Worst mean_abs_error'],rows)
    else:lines+=['','Fixed/parity: NOT MEASURED.']
    if results['video']:
        a=old['video']['codec50'];b=read(results['video']['path']);validate_protocol(a);validate_protocol(b);report['video']=b;rows=[]
        if a['cutPolicy']!=b['cutPolicy'] or a['backend']!=b['backend'] or a['precision']!=b['precision']:raise ValueError('Video protocol changed')
        for x,y in zip(a['sequences'],b['sequences']):
            if x['manifestSha256']!=y['manifestSha256'] or x['suspectedCutTransitions']!=y['suspectedCutTransitions']:raise ValueError('Frozen video changed')
            for metric in ('mae','referenceCompensatedTemporalMae','allTransitionTemporalMae'):
                v3=x['models']['adore-lite-v3'][metric];a3=x['models'][old['runs']['codec50']][metric];b5=y['models'][RUN][metric]
                if abs(v3-y['models']['adore-lite-v3'][metric])>1e-9:raise ValueError('V3 video baseline changed')
                rows.append([Path(x['manifest']).parent.name,metric,f'{v3:.9f}',f'{a3:.9f}',f'{b5:.9f}',pct(b5,a3),pct(b5,v3)])
        if b['artifacts'][RUN]['sha256']!=pins['step-5000-master-fp32.pt']:raise ValueError('Video used wrong C checkpoint')
        lines+=['','## Same144 real frames','',
                'Three48-frame360→720 sequences. CPU FP32 deploy, compressed WebRip reference, frozen suspected-cut policy. Pixel temporal delta does not certify absence of flicker, halos or detail popping.','']
        lines+=table(['Clip','Metric (lower better)','V3','Codec50 3k','C5k','C/3k %','C/V3 %'],rows)
    else:lines+=['','Real video: NOT MEASURED.']
    if results['benchmark']:
        perf=read(results['benchmark']['path']);report['performance']=perf;rows=[]
        if not perf['complete'] or perf['precision']!='fp16' or perf['provider']!='ROCm':raise ValueError('Benchmark backend/protocol changed')
        for c in perf['cases']:
            if sha(c['artifactPath'])!=c['artifactSha256'] or c['warmup']!=50 or c['iterations']!=500:raise ValueError('Benchmark incomplete')
            for component,r in c['components'].items():
                if 'medianMs' not in r:continue
                rows.append([c['model'],f"{c['width']}×{c['height']}→{c['outputWidth']}×{c['outputHeight']}",component,
                             *[r[k] for k in ('meanMs','medianMs','p50Ms','p95Ms','p99Ms','minMs','maxMs','fps')],c['peakAllocatedMiB']])
        lines+=['','## Performance','',
                'ROCm FP16,50 warmups/500 distinct inputs, independently measured inference/post-resize/total. Complete selected attempt only. FPS equivalent is not player FPS; Torch allocated memory is not complete system VRAM.','']
        lines+=table(['Model','Shape','Component','Mean ms','Median ms','P50','P95','P99','Min','Max','FPS equivalent','Peak MiB'],rows)
    else:lines+=['','Latency: NOT MEASURED.']
    if results['gradients']:report['gradients']=read(results['gradients']['path'])
    if results.get('export'):
        e=read(results['export']['report']);report['export']=e;rows=[]
        for c in e['cases']:
            for k,v in c.items():
                if isinstance(v,dict) and 'max_abs_error' in v:rows.append([c['name'],k,v['max_abs_error'],v['mean_abs_error']])
        lines+=['','## CPU synthetic export parity','']+table(['Case','Comparison','max_abs_error','mean_abs_error'],rows)
    if results.get('runtime'):
        r=read(results['runtime']['path']);report['runtime']=r;rows=[]
        for c in r['cases']:
            for key in ('fusionFp32','deployFp16'):rows.append([c['name'],key,c[key]['max_abs_error'],c[key]['mean_abs_error']])
        lines+=['','## Reloaded deployment ROCm parity','']+table(['Case','Comparison','max_abs_error','mean_abs_error'],rows)
    lines+=['','ONNX GPU/backend parity: NOT MEASURED. Export numeric correctness is separate from product quality advancement. Gradient diagnostics did not update weights. Final-hidden unopened.','',
            'Screening/Stage2 decision is recorded separately after checking individual fixed/video metrics and matched crops. No automatic candidate/default promotion by this builder.']
    (BASE/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (ROOT/'docs/zenkai-c5k-results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (folder/'metadata.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('C5k report, metadata and individual metric tables saved')


if __name__=='__main__':main()
