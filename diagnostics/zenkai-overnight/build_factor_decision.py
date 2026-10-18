"""Interpret completed independent screens without manufacturing a combined winner."""
from pathlib import Path
from build_factor_report import read,table,percent
ROOT=Path(__file__).resolve().parents[2]
FACTORS=('dataset-sampling','crop96','codec50')
REFERENCE='zenkai-ablation-spab-identity-skip-1k'


def main():
    assert read('diagnostics/zenkai-factor-screens-results.json')['state']=='complete-partial'
    assert read('diagnostics/zenkai-training/factor-performance-1k.status.json')['state'] in ('complete','failed')
    q=read('diagnostics/zenkai-ablations/factor-screens-1k-partial-quality/quality.json')
    reference=read('diagnostics/zenkai-real-video-skip-frequency-1k/report.json')
    lines=['# Zenkai: factor decisions and next gate','',
        'Measured independent1000-step screens, seed929. Decisions concern these checkpoints and this screening budget; they do not prove the architecture cannot improve at longer budget. HistoricalV3 selected25k versus this1k screen is a product baseline, not a matched-budget architecture attribution.','',
        '## MAE summary','', 'Lower is better; percentages are relative to frozen SkipFrequency1k. All individual metrics and absolute values are retained in [the complete measured tables](zenkai-factor-screens.md).','']
    rows=[]
    for domain,models in q['cases'].items():
        rows.append([domain,*[percent(models['factor-'+factor+'-1k']['mae'],models['skip-frequency-1k']['mae']) for factor in FACTORS]])
    lines+=table(['Domain','Group sampler/ref %','Crop96/ref %','Codec50/ref %'],rows)
    lines+=['','## Real-video summary','', 'All144 pinned frames per candidate. Visual MAE/reference-compensated temporal MAE:lower is better. No moving perceptual/blind certification.','']
    rows=[]
    for factor in FACTORS:
        run='zenkai-ablation-skip-factor-'+factor+'-1k'
        video=read(f'diagnostics/zenkai-real-video-factor-{factor}-1k/report.json')
        for a,b in zip(reference['sequences'],video['sequences']):
            m=b['models'][run];r=a['models'][REFERENCE];v=a['models']['adore-lite-v3']
            rows.append([factor,Path(a['manifest']).parent.name,percent(m['mae'],r['mae']),
                percent(m['referenceCompensatedTemporalMae'],r['referenceCompensatedTemporalMae']),
                percent(m['mae'],v['mae']),percent(m['referenceCompensatedTemporalMae'],v['referenceCompensatedTemporalMae'])])
    lines+=table(['Factor','Clip','Visual/ref %','Temporal/ref %','Visual/V3 %','Temporal/V3 %'],rows)
    lines+=['','## Classification','',
        '- Group sampler: do not adopt. Some fixed structural metrics improve, but real-video visualMAE regresses in all three clips; two temporal improvements do not erase the third regression. Uniform group coverage is not demonstrated quality superiority.',
        '- Crop96: do not adopt. All ten clean metrics improve; heavy/H264 all ten regress. Real-video visual and temporal errors regress in all three clips. A clean-only score would conceal this failure.',
        '- Codec50: promising targeted codec hypothesis, not an overall winner. H264/H265/AV1 improve in nine of ten fixed metrics each, but LPIPS regresses in all three; all ten clean metrics regress. Visual and temporal errors improve in every real clip versus SkipFrequency1k, yet still lose to HistoricalV3 in clips60/600. Keep separate experimental artifact, no automatic continuation/promotion.',
        '- Scheduler: INCONCLUSIVE / NOT MEASURED after failed process. Last loggedstep200, exit1, only initial state preserved; no valid optimizer/RNG resume at the interruption. Cause NOT ESTABLISHED. No restart or synthetic result.', '',
        'All three completed candidates: FAIL the advancement gate against HistoricalV3 at this screening budget. No candidate wins HistoricalV3 MAE in any fixed domain, and none gives a coherent real-video win across clips. Loss selection remains INCONCLUSIVE from the prior paired skip-loss experiment. Do not combine factors merely because they help different subsets.', '',
        '## Latency and gradients','',
        'Fresh protocol:ROCm FP16,50 warmups/500 distinct frames, inference/post-resize/total separately measured. Reference,V3,group sampler,crop timings completed and retained with all means/percentiles/allocator peaks. Codec50 failed with HIP unspecified launch failure before any completed timing; latency NOT MEASURED. GPU work stopped and planned reference repeat NOT EXECUTED. First reference360 median31.97ms vs group18.18ms and elevated reference720 P95 expose unresolved variation: factor-speed ranking INCONCLUSIVE. No timing replacement/retry or playerFPS/temperature claim.', '',
        'Identity-skip continues to avoid the near-dead gradient block found in original frequency architecture. All block-level FP32/scaledAMP norms and zero fractions retained; small nonzero zero-element counts in codec50 do not establish a dead block. Init→1k weight changes are cumulative, not per-step optimizer updates.', '',
        '## C5k and Stage2','',
        'C5k started:NO. Stage2 started:NO. No frozen recipe has demonstrated coherent gain against the product baseline, and scheduler evidence is incomplete. Total15k remains gated on C beginning to consistently exceed HistoricalV3; no automatic30k. Existing checkpoints/defaults/presets untouched, final-hidden unopened.', '',
        'Useful evidence for the next controlled round:codec50 is the most informative candidate for a same-protocol longer-budget hypothesis, but its clean/perceptual tradeoffs must remain visible. Before any faithful continuation, implement/review a dedicated exact-state continuation path for this factor family; the existing resume loader intentionally does not accept it. Original full1000-step optimizer/scaler/RNG/master/sampler state is preserved. Do not silently relax that guard or restart under the old run identity.', '',
        'Scheduler interruption requires its own diagnosed infrastructure plan; this report does not authorize inventing a missing step200 checkpoint. Future separate scheduler run would need a new identity and documented start/config, after deciding the failed-run handling.','',
        '## Manual still inspection','',
        'Nine panels generated using the same frames0/24/47 and ROI(448,100,832,484). Manual unblinded inspection:clip60/frame0,clip600/frame24,clip1200/frame47. Neural outputs improve some edges over bilinear, but fine eye/hair/finger and bright-background detail remain reduced versus reference. Differences among factors are subtle at these display crops; no robust overall visual preference established. Codec50 is a measured hypothesis, not a visually certified winner. Still inspection cannot verify flicker/shimmering/detail popping. Provenance/input/output hashes retained with the panels.','',
        '## Five review priorities','',
        '1. Codec50 wins all real clips versus Skip1k but regresses every clean metric and codec LPIPS.',
        '2. Group/crop improvements on fixed clean samples fail to generalize to these real clips.',
        '3. HistoricalV3 remains substantially cheaper and wins two of three clips; budget/protocol effects still matter.',
        '4. Scheduler has no post-initial checkpoint; preserve the failure rather than treating it as a completed ablation.',
        '5. Freeze the next single-factor hypothesis and its faithful state continuation before C5k; reserve final-hidden until the final architecture/loss/protocol.']
    (ROOT/'docs/zenkai-factor-decision.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Evidence-based factor decisions saved; no C/Stage2/promotion')


if __name__=='__main__':main()
