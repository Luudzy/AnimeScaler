"""Complete one-variable scheduler smoke report; no automatic recipe selection."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'native'))
from frozen_video_protocol import validate_protocol
from build_codec3k_report import selected,table,pct
BASE=ROOT/'diagnostics/zenkai-scheduler-v2-1k'
RUN='zenkai-ablation-skip-factor-scheduler-constant-v2-1k'
REFERENCE='zenkai-ablation-spab-identity-skip-1k'


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    pipeline=read(BASE/'pipeline.json');training=read(ROOT/'diagnostics/zenkai-training/scheduler-v2-1k-training/operation.json')
    if pipeline['state']!='COMPLETE' or training['state']!='PASS':raise ValueError('Scheduler operations unfinished')
    results={k:selected(v) for k,v in pipeline['results'].items()}
    folder=ROOT/'models/zenkai/experiments'/RUN;config=read(folder/'config.json');train=read(folder/'report.json')
    if train['stepsExecuted']!=train['optimizerUpdates'] or train['stepsExecuted']!=1000:raise ValueError('Wrong scheduler screening budget')
    for name,pin in config['sourceFilesSha256'].items():
        if sha(folder/'source'/name)!=pin:raise ValueError('Original scheduler source snapshot changed')
    report={'state':'COMPLETE','run':RUN,'training':training,'config':config,'history':train['history'],
            'trainingSeconds':train['trainingSeconds'],'best':train['best'],'operations':pipeline['results'],
            'quality':None,'video':None,'performance':None,'gradients':None,'releaseApproved':False}
    lines=['# Zenkai scheduler: new paired1k screening','',
           'Same warmup300, then constant2e-4 versus original5000-step cosine. Same identity skip, codec.25, patch128, batch4×2, seed929, frequency loss and frozen inputs. Actual preserved step-zero full initialization was paired; this is a new run, not a restart/resume of the failed old step200 experiment.','',
           f"Training1000/1000, exit0, duration{train['trainingSeconds']:.3f}s, best{train['best']}. Sources copied and hash verified. All recovery snapshots saved every100; no hidden or product changes.",'']
    lines+=table(['Operation','State','Selected attempt'],[[k,v['state'],v['selectedAttempt']] for k,v in pipeline['results'].items()])
    if results['quality']:
        q=read(results['quality']['quality']);p=read(results['quality']['parity']);report.update(quality=q,parity=p);rows=[]
        for domain,m in q['cases'].items():
            for metric,v3 in m['adore-lite-v3'].items():
                a=m['skip-frequency-1k'][metric];b=m['factor-scheduler-constant-v2-1k'][metric]
                rows.append([domain,metric,f'{v3:.9f}',f'{a:.9f}',f'{b:.9f}',pct(b,a),pct(b,v3)])
        lines+=['','## Fixed160 samples','',
                'Lower MAE/MSE/Edge/Laplacian/HF/DCT/LPIPS is better; higher PSNR/SSIM/MS-SSIM is better. PSNR is in dB. Individual metrics take priority over any composite.','']
        lines+=table(['Domain','Metric','HistoricalV3','Cosine1k','Constant1k','Constant/Cosine %','Constant/V3 %'],rows)
        rows=[]
        for name,r in p['models'].items():
            for key in ('fusionFp32','deployFp16'):
                rows.append([name,key,max(c[key]['max_abs_error'] for c in r['cases']),max(c[key]['mean_abs_error'] for c in r['cases'])])
        lines+=['','## Numerical parity','']+table(['Model','Comparison','Worst max_abs_error','Worst mean_abs_error'],rows)
    else:lines+=['','Fixed/parity: NOT MEASURED.']
    if results['video']:
        a=read(ROOT/'diagnostics/zenkai-real-video-skip-frequency-1k/report.json');b=read(results['video']['path'])
        validate_protocol(a);validate_protocol(b);report['video']=b;rows=[]
        if a['cutPolicy']!=b['cutPolicy'] or a['backend']!=b['backend'] or a['precision']!=b['precision']:raise ValueError('Video protocol changed')
        for x,y in zip(a['sequences'],b['sequences']):
            if x['manifestSha256']!=y['manifestSha256'] or x['suspectedCutTransitions']!=y['suspectedCutTransitions']:raise ValueError('Video sequence changed')
            for metric in ('mae','referenceCompensatedTemporalMae','allTransitionTemporalMae'):
                v3=x['models']['adore-lite-v3'][metric];ref=x['models'][REFERENCE][metric];new=y['models'][RUN][metric]
                if abs(v3-y['models']['adore-lite-v3'][metric])>1e-9:raise ValueError('V3 video changed')
                rows.append([Path(x['manifest']).parent.name,metric,f'{v3:.9f}',f'{ref:.9f}',f'{new:.9f}',pct(new,ref),pct(new,v3)])
        lines+=['','## Same144 frames','',
                'Three48-frame360→720 clips, CPU FP32 deploy. Pixel temporal delta with frozen reference-cut exclusions; all-transition metrics retained. Not blind/moving perceptual flicker certification.','']
        lines+=table(['Clip','Metric (lower better)','V3','Cosine1k','Constant1k','Constant/Cosine %','Constant/V3 %'],rows)
    else:lines+=['','Real video: NOT MEASURED.']
    if results['benchmark']:
        perf=read(results['benchmark']['path']);report['performance']=perf;rows=[]
        if not perf['complete'] or perf['precision']!='fp16' or perf['provider']!='ROCm':raise ValueError('Invalid benchmark')
        for c in perf['cases']:
            if sha(c['artifactPath'])!=c['artifactSha256'] or c['warmup']!=50 or c['iterations']!=500:raise ValueError('Incomplete benchmark')
            for key,r in c['components'].items():
                if 'medianMs' not in r:continue
                rows.append([c['model'],f"{c['width']}×{c['height']}→{c['outputWidth']}×{c['outputHeight']}",key,
                             *[r[k] for k in ('meanMs','medianMs','p50Ms','p95Ms','p99Ms','minMs','maxMs','fps')],c['peakAllocatedMiB']])
        lines+=['','## Performance','',
                'ROCm FP16,50 warmups/500 distinct inputs per component. Separate inference/post-resize/total, no timing mixing. FPS equivalent is not playerFPS. Identical model geometry means no scheduler-caused speed advantage can be inferred from run ordering.','']
        lines+=table(['Model','Shape','Component','Mean','Median','P50','P95','P99','Min','Max','FPS equivalent','Peak allocated MiB'],rows)
    else:lines+=['','Latency: NOT MEASURED.']
    if results['gradients']:report['gradients']=read(results['gradients']['path'])
    lines+=['','Gradient block norms/zero fractions and cumulative init→1k changes remain in the selected raw diagnostic. No optimizer update by diagnostics; no automatic dead-block classification or model promotion.','',
            'Recipe/advancement decision is recorded separately after reviewing these measurements. No C or15k is automatically started by this builder.']
    (BASE/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (ROOT/'docs/zenkai-scheduler-results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    metadata={'run':RUN,'config':config,'trainingExitCode':0,'stepFinal':1000,'trainingSeconds':train['trainingSeconds'],
              'best':train['best'],'validationHistory':train['history'],'trainingOperation':training,
              'checkpointsSha256':{p.name:sha(p) for p in folder.glob('*.pt')},
              'evaluationReport':str(BASE/'comparison.json'),'evaluationReportSha256':sha(BASE/'comparison.json'),
              'releaseApproved':False,'promoted':False}
    (folder/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print('New scheduler report and metadata saved')


if __name__=='__main__':main()
