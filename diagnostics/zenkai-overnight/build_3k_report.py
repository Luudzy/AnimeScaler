"""Append the matched3k evidence only after every stage has actually completed."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8-sig'))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def table(headers,rows):
    return ['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows]
def replace_section(path,heading,lines):
    p=ROOT/path;s=p.read_text(encoding='utf-8');s=s.split('\n'+heading)[0]
    p.write_text(s+'\n'+heading+'\n\n'+'\n'.join(lines)+'\n',encoding='utf-8')

def main():
    assert read('diagnostics/zenkai-training/ablations3k-recovery1.status.json')['state']=='complete'
    assert read('diagnostics/zenkai-training/ablations3k-evaluation-recovery5.status.json')['state']=='complete'
    q=read('diagnostics/zenkai-ablations/probes-3k-quality/quality.json')
    p=read('diagnostics/zenkai-ablations/performance-3k/benchmark_results.json')
    assert q['state']=='complete' and p['complete']
    runs=['zenkai-ablation-frequency-3k-continuation','zenkai-ablation-spab-identity-skip-3k-continuation']
    names=['zenkai-frequency-3k','zenkai-identity-skip-3k']
    videos=[read('diagnostics/zenkai-real-video-frequency-3k/report.json'),read('diagnostics/zenkai-real-video-skip-3k-complete/report.json')]
    assert all(v['state']=='complete' and sum(s['frameCount'] for s in v['sequences'])==144 for v in videos)
    lines=['Matched frequency/identity-skip1k ->3k TOTAL. One variable, same seed/sampler/loss/schedule; source snapshots preserved. This is a controlled ablation, not Stage2. No product promotion. Errors lower; PSNR/SSIM higher. All metrics retained.','']
    rows=[]
    for domain,models in q['cases'].items():
        for metric,ref in models[names[0]].items():
            skip=models[names[1]][metric];v3=models['adore-lite-v3'][metric]
            rows.append([domain,metric,f'{v3:.9f}',f'{ref:.9f}',f'{skip:.9f}',f'{(skip/ref-1)*100:+.4f}',f'{(skip/v3-1)*100:+.4f}'])
    lines+=table(['Domain','Metric','HistoricalV3','Frequency3k','Skip3k','Skip/ref %','Skip/V3 %'],rows)
    lines+=['','## Matched performance3k','','ROCm FP16,50warmup/500 full distinct frames; preparation excluded. Model plus requested post-resize, not player FPS. Components measured separately in raw JSON; no median subtraction. Driver utilization/temperature NOT MEASURED.','']
    rows=[]
    for c in p['cases']:
        rows.append([c['model'],f"{c['height']}p ->{c['outputHeight']}p"]+[f'{c[k]:.5f}' for k in ('meanMs','medianMs','p50Ms','p95Ms','p99Ms','minMs','maxMs','fps','peakAllocatedMiB')])
    lines+=table(['Run','Resolution','Mean ms','Median','P50','P95','P99','Min','Max','Equivalent FPS','Peak allocator MiB'],rows)
    lines+=['','## Frozen144-frame video3k','','CPU FP32, three identical48-frame clips and pinned manifests. Bilinear and HistoricalV3 rerun for each candidate. Reference-compensated temporal MAE is not optical-flow compensation or a perceptual flicker certificate.','']
    rows=[]
    for a,b in zip(videos[0]['sequences'],videos[1]['sequences']):
        assert a['manifestSha256']==b['manifestSha256']
        for metric in ('mae','referenceCompensatedTemporalMae','allTransitionTemporalMae'):
            v3=a['models']['adore-lite-v3'][metric];ref=a['models'][runs[0]][metric];skip=b['models'][runs[1]][metric]
            assert abs(v3-b['models']['adore-lite-v3'][metric])<1e-9
            rows.append([Path(a['manifest']).parent.name,metric,f"{a['models']['bilinear'][metric]:.9f}",f'{v3:.9f}',f'{ref:.9f}',f'{skip:.9f}',f'{(skip/ref-1)*100:+.4f}',f'{(skip/v3-1)*100:+.4f}'])
    lines+=table(['Clip','Metric','Bilinear','V3','Frequency3k','Skip3k','Skip/ref %','Skip/V3 %'],rows)
    lines+=['','These measurements do not automatically authorize15k. Screening A/B remain FAIL for advancement versus HistoricalV3. No5k smoke extension,30k, final-hidden access or release performed. Final choice must consider individual heavy/codec/video regressions, latency and visual review.','',
        'Sources:probes-3k-quality/quality.json and parity.json; performance-3k/benchmark_results.json; real-video-frequency-3k/report.json; real-video-skip-3k-complete/report.json.']
    gradient=ROOT/'diagnostics/zenkai-ablation-gradient-analysis-3k.json'
    lines+=['','3k checkpoint-local gradients: '+('MEASURED; diagnostics/zenkai-ablation-gradient-analysis-3k.json, SHA256 '+digest(gradient) if gradient.exists() else 'NOT MEASURED; separate diagnostic pending')+'. No checkpoint edited.']
    replace_section('docs/zenkai-ablations.md','## Completed matched3k evidence',lines)
    replace_section('docs/zenkai-overnight-report.md','## Additional controlled3k evidence',lines)
    replace_section('docs/zenkai-real-video.md','## Additional matched3k video',lines[lines.index('## Frozen144-frame video3k')+1:])
    for run,video in zip(runs,videos):
        folder=ROOT/'models/zenkai/experiments'/run;r=read(folder/'report.json')
        assert r['stepsExecuted']==3000 and not r['promoted']
        reports={'quality-report.json':q,'performance-report.json':p,'real-video-report.json':video,'training-history.json':r['history']}
        for name,value in reports.items():(folder/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
        meta={'run':run,'stepFinal':3000,'best':r['best'],'trainingExitCode':0 if 'skip' in run else 'NOT RECORDED; see frequency-3k-completion-audit.json','trainingSeconds':r['trainingSeconds'],
            'resumeTrainingSeconds':r['resumeTrainingSeconds'],'lastTrainingLoss':r['history'][-1]['trainingLoss'],
            'config':read(folder/'config.json'),'validationHistory':r['history'],'releaseApproved':False,
            'checkpointsSha256':{f.name:digest(f) for f in folder.glob('*.pt')},
            'evaluationArtifactsSha256':{n:digest(folder/n) for n in reports},
            'resumeScope':'Matched1k ->3k only; actual optimizer/scaler/RNG/sampler retained; original5k schedule'}
        (folder/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('Complete matched3k measurements saved; no release decision')

if __name__=='__main__':main()
