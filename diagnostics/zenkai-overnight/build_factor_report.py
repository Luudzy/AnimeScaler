"""Consolidate four independent factor screens only after every measured gate."""
import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'native'))
from frozen_video_protocol import validate_protocol

FACTORS = ('dataset-sampling', 'crop96', 'codec50', 'scheduler-constant')
REFERENCE = 'zenkai-ablation-spab-identity-skip-1k'


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def table(headers, rows):
    return ['| ' + ' | '.join(headers) + ' |',
            '| ' + ' | '.join(['---'] * len(headers)) + ' |'] + [
                '| ' + ' | '.join(map(str, row)) + ' |' for row in rows]


def percent(value, reference):
    if reference == 0:
        return 'UNDEFINED'
    return f'{100 * (value / reference - 1):+.5f}'


def main():
    global FACTORS
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partial',action='store_true',help='Report three completed factors and the failed scheduler')
    parser.add_argument('--with-performance',action='store_true',help='Require the completed matched new-checkpoint benchmark')
    parser.add_argument('--partial-performance',action='store_true',help='Record completed timings and the stopped HIP launch failure')
    args=parser.parse_args()
    args.with_performance=args.with_performance or args.partial_performance
    performance_labels=('reference',*FACTORS,'reference-repeat')
    recovery='factor-partial-evaluation-recovery1'
    if args.partial:
        FACTORS=FACTORS[:3]
        assert read('diagnostics/zenkai-training/factor-screens-1k.status.json')['state']=='failed'
        assert read('diagnostics/zenkai-training/factor-screens-1k-scheduler-constant.exit.json')['exitCode']==1
    for prefix in ((recovery,) if args.partial else ('factor-screens-1k', 'factor-video-1k')):
        if read(f'diagnostics/zenkai-training/{prefix}.status.json')['state'] != 'complete':
            raise RuntimeError('All training and video gates must complete before report')
    stages = ['factor-screens-1k-' + f for f in FACTORS] + [
        'factor-screens-1k-fixed-quality', 'factor-screens-1k-gradients', 'factor-video-1k-gate'] + [
        'factor-video-1k-' + f for f in FACTORS]
    if args.partial:
        stages=['factor-screens-1k-'+f for f in FACTORS]+[recovery+'-'+name for name in (
            'fixed-quality','gradients','video-gate',*('video-'+f for f in FACTORS))]
    if args.with_performance:
        if args.partial_performance:
            assert read('diagnostics/zenkai-training/factor-performance-1k.status.json')['state']=='failed'
            assert read('diagnostics/zenkai-training/factor-performance-1k-codec50.exit.json')['exitCode']==1
            performance_labels=('reference','dataset-sampling','crop96')
            stages += ['factor-performance-1k-'+label for label in performance_labels]
        else:
            assert read('diagnostics/zenkai-training/factor-performance-1k.status.json')['state']=='complete'
            assert read('diagnostics/zenkai-training/factor-performance-1k-reference-repeat.status.json')['state']=='complete'
            stages += ['factor-performance-1k-'+label for label in ('reference',*FACTORS)]
            stages += ['factor-performance-1k-reference-repeat']
    exits = {}
    for stage in stages:
        path = f'diagnostics/zenkai-training/{stage}.exit.json'
        if read(path)['exitCode'] != 0:
            raise RuntimeError('Nonzero or unrecorded exit: ' + stage)
        exits[stage] = {'exitCode': 0, 'path': path, 'sha256': sha(path)}

    quality_folder='factor-screens-1k-partial-quality' if args.partial else 'factor-screens-1k-quality'
    quality = read('diagnostics/zenkai-ablations/'+quality_folder+'/quality.json')
    old = read('diagnostics/zenkai-ablations/skip-loss-1k-quality/quality.json')
    assert quality['state'] == 'complete' and list(quality['sampleCounts'].values()) == [32] * 5
    for field in ('indices', 'validationManifestSha256', 'sourceManifestSha256', 'codecManifestSha256'):
        assert quality[field] == old[field], field
    parity = read('diagnostics/zenkai-ablations/'+quality_folder+'/parity.json')
    assert parity['state'] == 'complete' and all(x['parityApproved'] for x in parity['models'].values())
    gradient = read('diagnostics/zenkai-factor-partial-gradient-analysis.json' if args.partial else 'diagnostics/zenkai-factor-gradient-analysis.json')
    video_reference = read('diagnostics/zenkai-real-video-skip-frequency-1k/report.json')
    validate_protocol(video_reference)
    videos = {}
    training_rows = []
    metadata = {}
    for factor in FACTORS:
        run = 'zenkai-ablation-skip-factor-' + factor + '-1k'
        folder = ROOT / 'models/zenkai/experiments' / run
        config = read(folder / 'config.json')
        training = read(folder / 'report.json')
        pairing = read(folder / 'initial-pairing.json')
        assert training['stepsExecuted'] == training['optimizerUpdates'] == 1000
        assert pairing['state'] == 'matched' and pairing['factor'] == factor
        assert config['comparisonRun'] == REFERENCE and config['loss'] == 'zenkai_frequency'
        for name, digest in config['sourceFilesSha256'].items():
            assert sha(folder / 'source' / name) == digest, (run, name)
        video = read(f'diagnostics/zenkai-real-video-factor-{factor}-1k/report.json')
        validate_protocol(video)
        assert video['backend'] == video_reference['backend'] and video['precision'] == video_reference['precision']
        assert video['cutPolicy'] == video_reference['cutPolicy']
        for a, b in zip(video_reference['sequences'], video['sequences']):
            for field in ('manifestSha256', 'frameCount', 'suspectedCutTransitions'):
                assert a[field] == b[field], (factor, field)
            for metric in ('mae', 'referenceCompensatedTemporalMae', 'allTransitionTemporalMae'):
                assert abs(a['models']['adore-lite-v3'][metric] - b['models']['adore-lite-v3'][metric]) < 1e-9
        assert video['artifacts'][run]['sha256'] == sha(folder / 'step-1000-master-fp32.pt')
        g = gradient['runs'][run]
        for field, name in (('checkpointSha256', 'step-1000-master-fp32.pt'),
                            ('initialSha256', 'initial-training-state.pt'),
                            ('trainingStateSha256', 'step-1000-training-state.pt')):
            assert g[field] == sha(folder / name)
        videos[factor] = video
        training_rows.append([factor, 1000, 0, f"{training['trainingSeconds']:.3f}",
                              config['patchLR'], config['codecFraction'], training['best']])
        metadata[factor] = {'run': run, 'config': config, 'stepFinal': 1000, 'trainingExitCode': 0,
            'trainingSeconds': training['trainingSeconds'], 'best': training['best'],
            'validationHistory': training['history'], 'initialPairing': pairing,
            'checkpointsSha256': {p.name: sha(p) for p in folder.glob('*.pt')},
            'sourceSnapshotSha256': config['sourceFilesSha256'], 'stageExitRecords': exits,
            'performance': 'NOT MEASURED', 'onnx': 'NOT MEASURED', 'releaseApproved': False}
        metadata[factor]['screeningDecision']={'advancementAgainstHistoricalV3':'FAIL at measured1k screening',
            'factorAdoption':'PROMISING WITH REGRESSIONS, not selected' if factor=='codec50' else 'NOT SELECTED',
            'C5kStarted':False,'Stage2Started':False}

    lines = ['# Zenkai: independent protocol factor screens', '',
        'Generated UTC: ' + datetime.datetime.now(datetime.timezone.utc).isoformat(), '',
        f'{len(FACTORS)} completed separate one-seed1000-step screens against frozen Skip+Frequency1k. No combined recipe, C5k, Stage2 or product promotion. Every completed variant received the same160 fixed samples and144 real frames; no eligibility cutoff based on results.', '',
        '## Training and integrity', '',
        'Dataset changes group-versus-image sampling on the same1792 TRAIN identities, not dataset membership. Crop changes LR128→96. Codec changes intra-cache sampling .25→.50. Scheduler preserves the exact original first300 warmup steps, then uses constant2e-4. Shared initial model/optimizer/scaler/RNG/dataRNG match; only declared config/state exemptions apply.', '']
    lines += table(['Factor', 'Steps/updates', 'Exit', 'Seconds', 'LR crop', 'Codec fraction', 'Best recorded'], training_rows)
    if args.partial:
        lines += ['', 'Scheduler experiment: FAILED before first1000-step checkpoint. Log last records step200; process exit1 with no traceback, cause NOT ESTABLISHED. Only initial state survives, so optimizer/RNG after the last update cannot be recovered. No blind restart/resume. Final checkpoint, quality, parity, gradients, latency and video: NOT MEASURED. Original failed logs/status/source/init retained. The old CPU waiting runner also disappeared without a final exit/status; its stale waiting status is not completion evidence. A new unique evaluation-only runner processed just the three completed candidates.', '']
    lines += ['', 'All original checkpoints and copied training sources retained. Configs record git commit and source hashes. Best values are within-run validation records, not cross-loss winner declarations.', '',
        '## Fixed quality: individual metrics', '',
        'MAE/MSE/LPIPS/Edge/Laplacian/DCT/HF lower is better; PSNR/SSIM/MS-SSIM higher is better. PSNR is in dB: its percentage is only a relative arithmetic table value. Reference is the same frozen Skip+Frequency1k, not the3k continuation.', '']
    quality_rows = []
    for domain, models in quality['cases'].items():
        for metric, reference in models['skip-frequency-1k'].items():
            v3 = models['adore-lite-v3'][metric]
            for factor in FACTORS:
                value = models['factor-' + factor + '-1k'][metric]
                quality_rows.append([domain, metric, factor, f'{v3:.9f}', f'{reference:.9f}',
                                     f'{value:.9f}', percent(value, reference), percent(value, v3)])
    lines += table(['Domain', 'Metric', 'Factor', 'HistoricalV3', 'SkipFrequency1k', 'Factor value', 'Factor/ref %', 'Factor/V3 %'], quality_rows)
    lines += ['', '## Numerical ROCm equivalence', '', 'Worst per-case errors, not pooled averages. These gates test numerical export/fusion correctness, not perceptual quality.', '']
    rows = []
    for model, result in parity['models'].items():
        for comparison in ('fusionFp32', 'deployFp16'):
            rows.append([model, comparison, f"{max(c[comparison]['max_abs_error'] for c in result['cases']):.9g}",
                         f"{max(c[comparison]['mean_abs_error'] for c in result['cases']):.9g}"])
    lines += table(['Model', 'Comparison', 'max_abs_error', 'mean_abs_error'], rows)
    lines += ['', 'ONNX export/parity: NOT MEASURED. CPU video duration is not GPU/player FPS.', '']
    if args.with_performance:
        performance_rows=[]
        for label in performance_labels:
            perf=read(f'diagnostics/zenkai-ablations/factor-performance-1k/{label}/benchmark_results.json')
            assert perf['complete'] and perf['precision']=='fp16' and perf['provider']=='ROCm'
            for row in perf['cases']:
                assert row['warmup']==50 and row['iterations']==500
                assert row['artifactSha256']==sha(row['artifactPath'])
                assert (row['width'],row['height'],row['outputWidth'],row['outputHeight']) in (
                    (640,360,1280,720),(1280,720,1920,1080))
                for component,result in row['components'].items():
                    if not isinstance(result,dict) or 'medianMs' not in result:
                        performance_rows.append([row['model']+' ['+label+']',f"{row['width']}x{row['height']}→{row['outputWidth']}x{row['outputHeight']}",component,
                                                 'NOT APPLICABLE']+['—']*8)
                        continue
                    assert result['iterations']==500 and result['warmup']==50
                    performance_rows.append([row['model']+' ['+label+']',f"{row['width']}x{row['height']}→{row['outputWidth']}x{row['outputHeight']}",component,
                        *[f"{result[key]:.6f}" for key in ('meanMs','medianMs','p50Ms','p95Ms','p99Ms','minMs','maxMs','fps')],
                        f"{row['peakAllocatedMiB']:.3f}"])
            if label in metadata:metadata[label]['performance']='MEASURED: ROCm FP16 completion-timed model/resize/total'
        lines += ['## Matched new-checkpoint performance','',
            'ROCm FP16,50 warmup and500 distinct full random frames per measured component. Preparation excluded, completion barrier used. Model inference/post-resize/total are separately timed passes; do not subtract medians. FPS equivalent is1000/mean measured milliseconds, not player FPS. Peak allocated memory is the Torch allocator peak, not complete system VRAM. Driver utilization/temperature: NOT MEASURED.','']
        lines += table(['Model','Input→output','Component','Mean ms','Median ms','P50','P95','P99','Min','Max','FPS equivalent','Peak allocated MiB'],performance_rows)
        if args.partial_performance:
            lines += ['', 'Codec50 benchmark failed with HIP unspecified launch failure while synchronizing the first measured case, exit1; no completed case/timing recorded. Codec50 latency: NOT MEASURED. GPU work stopped; queued reference/V3 repeat NOT EXECUTED. Initial reference360→720 median31.97ms versus same-architecture group18.18ms exposes unresolved timing variation; reference720 P95 is also high. All completed values retained, but factor-speed ranking is INCONCLUSIVE. Do not call this a training-factor speed improvement. No GPU retry/reset or driver/config change.']
        else:
            lines += ['', 'Initial reference360→720 median31.97ms versus same-architecture group candidate18.18ms prompted ONE identical reference/V3 repeat after all candidates. Both sets retained with separate labels and exits. Cause of timing variation NOT ESTABLISHED; do not select the fastest reference silently or attribute ordering/environment variation to changed training data.']
    else:
        lines += ['New-checkpoint latency: NOT MEASURED. Prior3k architecture timings are contextual and are not reassigned to these new checkpoints.']
    lines += ['',
        '## Same144 real frames', '',
        'CPU FP32 deploy_copy, three48-frame360→720 clips. Standard reference-compensated temporal MAE excludes the pinned suspected cut; all-transition error retains it. Pixel deltas do not certify absence of perceptual flicker. Original source is a compressed4K WebRip, not proven native4K ground truth.', '']
    video_rows = []
    for factor, video in videos.items():
        run = metadata[factor]['run']
        for a, b in zip(video_reference['sequences'], video['sequences']):
            for metric in ('mae', 'referenceCompensatedTemporalMae', 'allTransitionTemporalMae'):
                reference = a['models'][REFERENCE][metric]
                v3 = a['models']['adore-lite-v3'][metric]
                value = b['models'][run][metric]
                video_rows.append([Path(a['manifest']).parent.name, metric, factor, f'{v3:.9f}',
                    f'{reference:.9f}', f'{value:.9f}', percent(value, reference), percent(value, v3)])
    lines += table(['Clip', 'Metric lower is better', 'Factor', 'V3', 'SkipFrequency1k', 'Factor value', 'Factor/ref %', 'Factor/V3 %'], video_rows)
    lines += ['', '## Checkpoint-local gradient evidence', '',
        'Same clean/heavy samples; both FP32 and scaledAMP using actual checkpoint scales. No optimizer step. Delta is accumulated init→1k weight change, not an individual optimizer update or complete training trajectory.', '']
    rows = []
    for run, result in gradient['runs'].items():
        for case in result['cases']:
            for precision in ('fp32', 'amp'):
                for block, g in case[precision]['groups'].items():
                    delta = result['accumulatedWeightChanges'][block]['deltaL2']
                    weight = result['currentWeightL2'][block]
                    rows.append([run, case['case'], precision, block, f"{g['gradientL2']:.9g}",
                        f"{g['zeroElementsPct']:.6f}", f'{weight:.9g}', f'{delta:.9g}', f'{delta/max(weight,1e-30):.9g}'])
    lines += table(['Run', 'Case', 'Precision', 'Block', 'Grad L2', 'Zero %', 'Current weight L2', 'Init→1k delta L2', 'Delta/current'], rows)
    lines += ['', '## Budget and limits', '',
        'Preserved SkipFrequency1k→3k comparison is in docs/zenkai-budget-analysis.md. MAE improves in all five domains but still loses to HistoricalV3. This trajectory also advances the LR schedule; it is not a pure budget-vs-scheduler identification.', '',
        'All fixed domains and measured regressions remain visible. This one-seed1k smoke cannot establish statistical equivalence, late-budget interaction or generalization. Group labels are heuristic; codec TRAIN cache is all-intra, not dynamic GOP training. Human blind study and moving perceptual review: NOT EXECUTED. Final-hidden remains untouched.', '',
        '## Decision', '',
        'Detailed interpretation and the next training gate are recorded separately after reviewing all measurements. No candidate selection is inferred automatically from a composite. C5k requires a frozen justified recipe; total15k requires C to start consistently exceeding HistoricalV3.']
    (ROOT / 'docs/zenkai-factor-screens.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    for factor, m in metadata.items():
        folder = ROOT / 'models/zenkai/experiments' / m['run']
        reports = {'quality-report.json': quality, 'real-video-report.json': videos[factor],
                   'gradient-report.json': gradient['runs'][m['run']], 'training-history.json': m['validationHistory']}
        if args.with_performance and factor in performance_labels:
            reports['performance-report.json']=read(f'diagnostics/zenkai-ablations/factor-performance-1k/{factor}/benchmark_results.json')
        failures={}
        for name in ('factor-screens-1k-scheduler-constant','factor-performance-1k-codec50'):
            path=f'diagnostics/zenkai-training/{name}.exit.json'
            if (ROOT/path).exists() and read(path)['exitCode']!=0:
                failures[name]={'exitCode':read(path)['exitCode'],'exitPath':path,'exitSha256':sha(path),
                    'logs':{suffix:{'path':f'diagnostics/zenkai-training/{name}.{suffix}.log',
                                   'sha256':sha(f'diagnostics/zenkai-training/{name}.{suffix}.log')}
                            for suffix in ('stdout','stderr')}}
        m['failedStageRecords']=failures
        for name, value in reports.items():
            (folder / name).write_text(json.dumps(value, indent=2), encoding='utf-8')
        m['evaluationArtifactsSha256'] = {name: sha(folder / name) for name in reports}
        (folder / 'metadata.json').write_text(json.dumps(m, indent=2), encoding='utf-8')
    summary = {'state': 'complete-partial' if args.partial else 'complete', 'scope': 'Independent1k screens, no automatic recipe selection',
        'unavailableFactors': {'scheduler-constant':'FAILED before final checkpoint; NOT MEASURED'} if args.partial else {},
        'stageExitRecords': exits, 'qualityRows': quality_rows, 'videoRows': video_rows,
        'reportSha256': sha('docs/zenkai-factor-screens.md'), 'releaseApproved': False}
    if args.partial_performance:
        summary['performanceStatus']={'state':'failed-partial','codec50':'NOT MEASURED: HIP launch failure',
                                     'referenceRepeat':'NOT EXECUTED: GPU safety stop', 'speedRanking':'INCONCLUSIVE'}
    if args.partial:
        folder=ROOT/'models/zenkai/experiments/zenkai-ablation-skip-factor-scheduler-constant-1k'
        failed={'run':folder.name,'config':read(folder/'config.json'),'trainingExitCode':1,
            'stepFinal':'NOT RECORDED','lastLoggedStep':200,'status':'FAILED, cause NOT ESTABLISHED',
            'resumeState':'Initial state only; no optimizer/RNG checkpoint after updates; not resumed/restarted',
            'checkpointsSha256':{p.name:sha(p) for p in folder.glob('*.pt')},
            'sourceSnapshotSha256':read(folder/'config.json')['sourceFilesSha256'],
            'initialPairing':read(folder/'initial-pairing.json'),'failedStageRecords':failures,
            'quality':'NOT MEASURED','performance':'NOT MEASURED','realVideo':'NOT MEASURED','releaseApproved':False}
        (folder/'metadata.json').write_text(json.dumps(failed,indent=2),encoding='utf-8')
    (ROOT / 'diagnostics/zenkai-factor-screens-results.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('All measured factor results and per-run metadata saved')


if __name__ == '__main__':
    main()
