"""Record the controlled historical-loss probe from complete measured artifacts."""
import datetime
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN='zenkai-ablation-legacy-v3-loss-1k'
REFERENCE='zenkai-ablation-frequency-1k'
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def table(headers,rows):return ['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows]

def main():
    status=read('diagnostics/zenkai-training/recipe-loss-1k.status.json')
    assert status['state']=='complete'
    folder=ROOT/'models/zenkai/experiments'/RUN
    train=read(folder/'report.json');config=read(folder/'config.json');pair=read(folder/'initial-pairing.json')
    assert train['stepsExecuted']==1000 and train['optimizerUpdates']==1000 and pair['state']=='matched'
    exits={}
    for name in ('train','export','parity','fixed-quality','video-gate','video-frequency','video-legacy-v3-loss'):
        path='diagnostics/zenkai-training/recipe-loss-1k-'+name+'.exit.json'
        value=read(path)
        assert value['exitCode']==0,(path,value)
        exits[name]={'path':path,'sha256':sha(path),'exitCode':0}
    frozen=read('models/legacy/adore-lite-v3/frozen-20260930/report.json')
    assert config['lossWeights']==frozen['loss']
    q=read('diagnostics/zenkai-ablations/legacy-v3-loss-1k-quality/quality.json')
    assert q['state']=='complete' and sum(q['sampleCounts'].values())==160
    videos=[read(f'diagnostics/zenkai-real-video-{label}-1k/report.json') for label in ('frequency','legacy-v3-loss')]
    import sys
    sys.path.insert(0,str(ROOT/'native'))
    from frozen_video_protocol import validate_protocol
    for video in videos:validate_protocol(video)
    lines=['# Zenkai: historical-loss recipe ablation1k','',f"Measured UTC:{datetime.datetime.now(datetime.timezone.utc).isoformat()}",'',
      'Original plan items50–54. One changed factor:the complete loss profile. This does not isolate individual Edge/Laplacian/SSIM terms. HistoricalV3 used a selected25k checkpoint in a cumulative30k run; this1000-step probe cannot measure the entire budget gap. No final-hidden, promotion or Stage2.', '',
      'Candidate:32 channels/depth5 x2, seed929, LR128 crop, batch4/accumulation2, codec.25, AdamW and first1000 steps of the same5k schedule. Loss:Charbonnier1+Edge.2+Laplacian.1+SSIM.05, exact legacy implementation. Reference:frequency1k with Charbonnier1+HF1+DCT.001. Full initialized model/optimizer/scaler/RNG/sampler/dataRNG equality passed before first update; source snapshot SHA:'+pair['referenceSha256'], '',
      f"Training exit0; final step1000; duration{train['trainingSeconds']:.3f}s; best:{train['best']}. Objective loss values are recorded but are not directly comparable across different recipes.", '',
      '## Numeric equivalence','','Independent synthetic-case numeric gates; no quality claim. Worst per-case max/mean errors, not pooled mean:','']
    export=read('models/zenkai/exports/'+RUN+'/export-report.json')
    runtime=read('diagnostics/zenkai-ablations/legacy-v3-loss-1k-runtime.json')
    rows=[]
    for scope,report,keys in [('CPU',export,('fusionFp32','deployFp16','onnxFP32','onnxFP16')),('ROCm',runtime,('fusionFp32','deployFp16'))]:
        assert report['state']=='complete' and report['parityApproved']
        for key in keys:rows.append([scope,key,f"{max(c[key]['max_abs_error'] for c in report['cases']):.9g}",f"{max(c[key]['mean_abs_error'] for c in report['cases']):.9g}"])
    lines+=table(['Backend','Comparison','max_abs_error','mean_abs_error'],rows)
    lines+=['','ONNX GPU:NOT MEASURED. New-checkpoint latency:NOT MEASURED. The architecture is unchanged, but previous checkpoint timings are not relabeled as this run’s benchmark.','', '## Fixed160-sample comparison','','Errors/LPIPS lower; PSNR/SSIM/MS-SSIM higher. All metrics retained, no composite-only winner.','']
    rows=[]
    for domain,models in q['cases'].items():
        for metric,ref in models['zenkai-frequency-1k'].items():
            candidate=models['zenkai-legacy-v3-loss-1k'][metric];v3=models['adore-lite-v3'][metric]
            rows.append([domain,metric,f'{v3:.9f}',f'{ref:.9f}',f'{candidate:.9f}',f'{(candidate/ref-1)*100:+.4f}',f'{(candidate/v3-1)*100:+.4f}'])
    lines+=table(['Domain','Metric','HistoricalV3','Frequency1k','LegacyLoss1k','New/ref %','New/V3 %'],rows)
    lines+=['','## Frozen144-frame video','','Same three48-frame sequences, CPU FP32. These are reference-compensated temporal pixel errors, not optical-flow or perceptual flicker certification.','']
    rows=[]
    for a,b in zip(videos[0]['sequences'],videos[1]['sequences']):
        for field in ('manifestSha256','frameCount','suspectedCutTransitions'):assert a[field]==b[field]
        for metric in ('mae','referenceCompensatedTemporalMae','allTransitionTemporalMae'):
            v3=a['models']['adore-lite-v3'][metric];ref=a['models'][REFERENCE][metric];candidate=b['models'][RUN][metric]
            assert abs(v3-b['models']['adore-lite-v3'][metric])<1e-9
            rows.append([Path(a['manifest']).parent.name,metric,f'{v3:.9f}',f'{ref:.9f}',f'{candidate:.9f}',f'{(candidate/ref-1)*100:+.4f}',f'{(candidate/v3-1)*100:+.4f}'])
    lines+=table(['Clip','Metric','V3','Frequency1k','LegacyLoss1k','New/ref %','New/V3 %'],rows)
    lines+=['','Blind panels were generated with a separate key. Human blind preference study:NOT EXECUTED. Manual still review and experiment decision are recorded below after inspection. No automatic extension.','',
      'Sources:diagnostics/zenkai-ablations/legacy-v3-loss-1k-quality/quality.json; diagnostics/zenkai-real-video-frequency-1k/report.json; diagnostics/zenkai-real-video-legacy-v3-loss-1k/report.json.']
    lines += ['','## Decision and still review','',
      'Loss superiority at1000 steps:INCONCLUSIVE. Advancement against HistoricalV3:FAIL. All160 fixed comparisons and144 video frames per candidate completed with explicit exit0. There were1000 optimizer updates for1000 steps.', '',
      'Compared with frequency1k, clean MAE worsened0.4683%; heavy/H264/H265/AV1 MAE improved only0.0368/0.1386/0.0486/0.0704%. Clean metrics all regressed; H265 andAV1 edge/Laplacian/LPIPS results are mixed. This is not coherent fixed-image evidence for changing the chosen loss.', '',
      'Video visual MAE improved0.5310/0.1909/4.7908% relative to frequency1k. Temporal MAE worsened0.0468/0.0750% in clips60/600 and improved1.7995% in clip1200. Against HistoricalV3, visual and temporal errors remain worse in clips60/600 and better in clip1200. The large relative difference on clip1200 starts from a small absolute error; all absolute values remain in the tables.', '',
      'Nine still panels use unchanged frames0/24/47 and ROI(448,100,832,484); provenance and source hashes are in diagnostics/zenkai-real-video-comparison-crops-recipe-1k/provenance.json. Manual inspection was performed on clip60/frame0,clip600/frame24,clip1200/frame47. Both1k neural outputs are visibly sharper than bilinear, but their differences are subtle. Fine eye/hair/finger/background details remain lost relative to the reference. No clear consistent visual preference for the legacy recipe was established in those three stills. This is an unblinded still inspection, not a moving-video or human blind study.', '',
      'Interpretation:swapping the complete loss profile alone did not explain or close the gap at1000 steps. This does not rule out a later training-budget effect or a loss interaction with the identity-skip architecture. The original near-dead-block gradient problem was not changed in this probe. One seed and one training budget cannot establish statistical equivalence of recipes.', '',
      'Next informative experiment:if loss testing is extended, use a paired frequency/legacy comparison on the same learning-enabled identity-skip model, with identical initialized state and1000-step budget. Change only the loss profile within that pair. Do not extrapolate this run into15k or treat an interaction between two changed factors as a single-factor result. This follow-up has not been executed.', '',
      'Stage2:NOT STARTED. No candidate promoted; historical baselines and prior runs retained. New-checkpoint performance and ONNX GPU remain NOT MEASURED. Final-hidden remains unopened.']
    (ROOT/'docs/zenkai-loss-recipe.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for name,value in {'quality-report.json':q,'real-video-report.json':videos[1],'training-history.json':train['history']}.items():
        (folder/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    meta={'run':RUN,'stepFinal':1000,'trainingExitCode':0,'trainingSeconds':train['trainingSeconds'],'config':config,
      'best':train['best'],'lastTrainingLoss':train['history'][-1]['trainingLoss'],'validationHistory':train['history'],
      'initialPairing':pair,'stageExitRecords':exits,'realVideoEvaluation':'complete:144 frozen frames','screeningDecision':{'lossSuperiority':'INCONCLUSIVE','advancementAgainstHistoricalV3':'FAIL'},'releaseApproved':False,'performance':'NOT MEASURED','humanBlindPreferenceStudy':'NOT EXECUTED',
      'checkpointsSha256':{p.name:sha(p) for p in folder.glob('*.pt')},
      'evaluationArtifactsSha256':{p.name:sha(p) for p in folder.glob('*report.json')},
      'sourceSnapshotSha256':{p.name:sha(p) for p in (folder/'source').glob('*.py')},
      'exportArtifacts':{str(p):sha(p) for p in (ROOT/'models/zenkai/exports'/RUN).iterdir() if p.is_file()}}
    (folder/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('Recipe-loss measured report and metadata saved; no automatic promotion')

if __name__=='__main__':main()
