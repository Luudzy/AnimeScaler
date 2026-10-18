"""Regenerate the review report strictly from recorded local measurements."""
import json
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8-sig'))
def table(headers,rows):return ['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows]
def f(value):return f'{value:.8g}'
def main():
 a=read('models/zenkai/experiments/zenkai-lite-v1-a-5k/metadata.json');b=read('models/zenkai/experiments/zenkai-lite-v1-b-5k/metadata.json')
 q=read('diagnostics/zenkai-ab-5k/quality/quality.json');perf=read('diagnostics/zenkai-ab-5k/performance/benchmark_results.json');v=read('diagnostics/zenkai-real-video-comparison-5k.json')
 control=read('models/zenkai/experiments/adore-v3-retrained-control-5k/metadata.json');cv=read('diagnostics/zenkai-real-video-control-5k/report.json');cp=read('diagnostics/zenkai-control-5k/performance/benchmark_results.json')
 assert q['state']=='complete' and v['state']=='complete' and cv['state']=='complete'
 lines=['# ZENKAI OVERNIGHT REPORT','',f'Updated UTC:{datetime.now(timezone.utc).isoformat()}. Saved measurements only. No default/preset changes, final-hidden access or30k run.','', '## 1. B TRAINING','',f"Final step:{b['stepFinal']}. Exit code:{b['status']['exitCode']}. Status:COMPLETE. B was not restarted or altered mid-run.",f"Best checkpoint:best-composite-master-fp32.pt, step5000. Composite:1.7889756511294577. Final recorded training loss:{b['lastTrainingLoss']:.9f}. Duration:{b['trainingSeconds']:.3f}s.",'','Master/best hashes, complete configuration, logged final loss, all validation history and gitCommit recorded in `models/zenkai/experiments/zenkai-lite-v1-b-5k/metadata.json`. A equivalent metadata preserved. Best/final model tensors identical at5k; serialization archive names cause different file hashes.','', '## 2. EXPORT','','FP32 deploy, FP16 and ONNX FP32/FP16 completed for A/B; numeric gates passed within predefined tolerances. ONNX CPUExecutionProvider measured; ONNX GPU:NOT MEASURED. ROCm PyTorch parity measured separately.','', 'Worst error across synthetic cases (maximum of per-case max and mean, not a pooled mean):','']
 rows=[]
 for name in ('a','b'):
  export=read(f'models/zenkai/exports/{name}-5k/export-report.json');runtime=read(f"diagnostics/zenkai-ab-5k/runtime-{'a-retry1' if name=='a' else 'b'}.json")
  assert export['parityApproved'] and runtime['parityApproved']
  for scope,report,keys in [('CPU',export,['fusionFp32','deployFp16','onnxFP32','onnxFP16']),('ROCm',runtime,['fusionFp32','deployFp16'])]:
   for key in keys:rows.append([name,scope,key,f(max(c[key]['max_abs_error'] for c in report['cases'])),f(max(c[key]['mean_abs_error'] for c in report['cases']))])
 lines+=table(['Run','Runtime','Comparison','max_abs_error','mean_abs_error'],rows)
 lines+=['','Tolerance fusionFP32:max2e-5/mean2e-6; FP16:max.004/mean.0005. Hashes and full case values in export-report.json/runtime JSONs. These synthetic numeric gates do not prove image-quality superiority.','', '## 3. PERFORMANCE','','ROCm RX5500XT FP16,50warmup/500 full-distinct-frame measurements, completion barrier. Preparation excluded; no reuse. Values are model plus required post-resize, not playerFPS. Driver/utilization/temperature:NOT MEASURED. Peak allocator memory is recorded, not complete systemVRAM.','']
 rows=[]
 for case in perf['cases']+cp['cases']:
  if case['model'] not in ('zenkai-a','zenkai-b','adore-lite-v3','adore-v3-retrained-control'):continue
  rows.append([case['model'],f"{case['width']}×{case['height']} → {case['outputWidth']}×{case['outputHeight']}",case['parameters']]+[f(case[k]) for k in ('meanMs','medianMs','p50Ms','p95Ms','p99Ms','minMs','maxMs','fps','peakAllocatedMiB')])
 lines+=table(['Model','Input → output','Deploy params','Mean ms','Median ms','P50','P95','P99','Min','Max','FPS equivalent','Peak MiB'],rows)
 lines+=['','Model inference/post-resize/total are separate measured passes, not subtracted medians. Component tables and all percentiles in `docs/zenkai-screening-results.md` and raw performance JSONs. Revised full-frame input protocol was applied to A/B/V3 equally; earlier single-pixel-change benchmarks are not paired evidence.','', '## 4. FIXED VALIDATION','','Every individual metric,32 samples/domain. Anime only; no illustration subdivision available. Errors/LPIPS:lower better. PSNR/SSIM/MS-SSIM:higher better.','']
 rows=[]
 for domain,models in q['cases'].items():
  for metric in models['adore-lite-v3']:
   rows.append([domain,metric]+[f(models[id][metric]) for id in ('adore-lite-v3','zenkai-a','zenkai-b')])
 lines+=table(['Domain','Metric','V3','A','B'],rows)
 lines+=['','All A/V3%, B/V3%, B/A% and individual regressions retained in `docs/zenkai-screening-results.md`. Both regress MAE/SSIM/EdgeMAE in all5domains; cleanLPIPS improves but heavy/codecLPIPS regress. Composite is not the winner criterion.','', '## 5. REAL VIDEO','','Same3clips,48frames each,144total, CPUFP32. Same360p→720p source manifests and references. No frame reuse. Lower is better.','']
 rows=[]
 for sequence,cs in zip(v['sequences'],cv['sequences']):
  assert sequence['manifestSha256']==cs['manifestSha256']
  for metric in ('mae','referenceCompensatedTemporalMae'):
   rows.append([Path(sequence['manifest']).parent.name,metric]+[f(sequence['values'][id][metric]) for id in ('v3','a','b')]+[f(cs['models']['adore-v3-retrained-control'][metric])])
 lines+=table(['Clip','Metric','V3','A','B','RetrainedControl'],rows)
 lines+=['','A/B/control all lose to historicalV3 in two clips and win in one. B beats A and control in all3clips, insufficient to override historical regressions. Reference-compensated temporal pixel error is not optical-flow compensation or flicker certification. Full all-transition errors/cut exclusions retained.','', 'Still review inspected matched native-size lineart/eyes/hair/action/bright-background crops; gains over bilinear, subtle A/B/V3 differences, no clear Zenkai superiority. Fine details remain absent. Moving perceptual flicker/shimmer/detail-popping review:NOT EXECUTED. Blind human preference study:NOT EXECUTED. Nine matched panels and full frame previews preserved.','', '## 6. GRADIENT ANALYSIS','','Initialization/1k/3k/5k CPUFP32 audit complete. Body gradients initially zero from zero head are expected. At5k, deeply attenuated gradients persist, with near-zero centered gates rather than broad saturation. First feature/first-block shortcuts provide a strong route to the head.','']
 g=read('diagnostics/zenkai-gradient-analysis.json');rows=[]
 for run,records in g['runs'].items():
  probe=records[-1]['probes'][0]
  for block,stats in probe['gradient']['groups'].items():
   update=probe['simulatedUpdate']['groups'][block]
   rows.append([run,block,f(stats['gradientL2']),f(update['weightL2']),f(update['updateL2']),f(update['updateToWeightRatio']) if update['updateToWeightRatio'] is not None else 'undefined'])
 lines+=table(['Run5k clean','Block','Grad norm','Weight norm','Simulated update norm','Simulated update/weight'],rows)
 lines+=['','Update rows are fresh-state AdamW simulations on copies, not actual historical updates. A/B optimizer state was not saved. Accumulated initialization-weight deltas and GRep branch activity in diagnostic JSON.','', 'Actual matched ROCm precision probe complete. At loss scale262144 (matches final training log), second block gradients zero in AMP:A97.7–98.5%, B99.996–100%, versus0% exactly-zero FP32 elements on these samples. FP32 norms already tiny, so precision aggravates architecture attenuation rather than solely explaining it. Checkpoint-local observations; no historical scaler reconstruction. `diagnostics/zenkai-gradient-analysis.json`, `diagnostics/zenkai-gradient-precision.json`, `docs/zenkai-gradient-analysis.md`.','', '## 7. RETRAINED V3 CONTROL','','Executed:YES. Exact24channels/4blocks/.2 internal residual scale,5k, seed929, same new protocol. Training exit0, fixed quality/performance/144video complete. HistoricalV3 preserved.',f"Duration:{control['trainingSeconds']:.3f}s. Best composite:{control['best']}. Final loss:{control['lastTrainingLoss']:.9f}.",'', 'Control also regresses fixed MAE/SSIM all domains and video two clips. The new training recipe/budget contributes; a Zenkai-only explanation is insufficient. B improves control but at much higher latency. Historical→control compares full protocol/budget, not isolated dataset causality. Detailed all-metric table in `docs/zenkai-retrained-control.md`. Optimizer/RNG snapshots saved; resume functionality not implemented/tested.','', '## 8. SCREENING DECISION','','A:FAIL. B:FAIL for advancement from this screening. Neither demonstrates coherent gain over historicalV3 across quality/video/latency. B is stronger than A on video, but costlier. Preferred product baseline:historicalAdoreV3. Controlled ablations complete; identity bypass restores learning but fails advancement after3k. No candidate declared final.','', '### Controlled short experiments','','Each uses same architecture A, initialization seed, dataset/crops, codec, batch, patch, and first1k prefix of5k LR schedule. No GAN/VGG/LPIPS loss. Output residual gain and identity skip are separate-family probes rejected by ordinary exporter.','']
 rows=[]
 for run,label in [('zenkai-ablation-frequency-1k','Reference frequency'),('zenkai-ablation-charbonnier-1k','Charbonnier only'),('zenkai-ablation-output-residual-02-1k','Output learned residual gain.2 vs1'),('zenkai-ablation-spab-identity-skip-1k','SPAB identity bypass gain1 vs0')]:
  path=ROOT/f'models/zenkai/experiments/{run}/report.json'
  if path.exists():
   rr=json.loads(path.read_text());rows.append([label,rr['stepsExecuted'],f(rr['history'][-1]['summary']['composite']),f(rr['trainingSeconds'])])
  else:rows.append([label,'PENDING','NOT MEASURED','NOT MEASURED'])
 lines+=table(['Experiment','Completed steps','Composite (descriptive)','Duration s'],rows)
 for status in ('ablations1','residual-probe','skip-probe'):
  path=ROOT/f'diagnostics/zenkai-training/{status}.status.json'
  if path.exists():ss=read(path);lines.append(f"\n{status}:{ss['state']} / {ss['phase']}. {ss['message']}")
 lines+=['', '## 9. STAGE2','','Started:NO. No consistent A/B win. Requested faithful5k→15k continuation also unavailable from weights-only A/B masters; no fabricated optimizer/RNG state. No30k run. No10k/15k evaluations because no Stage2 authorized by evidence gate. Short ablations inform tomorrow’s architecture/loss decision.','', '## 10. PROBLEMS FOUND','','- Standalone MIOpen used unwritable external default cache; tested workspace cache fix, successful retry. Original failure report/logs preserved.', '- A/B checkpoint format lacks optimizer/scaler/RNG state; diagnosed rather than resetting state.', '- B foreground CPU video interrupted by a steered turn after2clips; verified process ended, evaluated only remaining third clip, tested merge retained source hashes. No training restarted.', '- Deep FP32 gradient attenuation and additional AMP zeroing observed.', '- Historical V3 control shares recipe regressions; dataset/loss/budget effects need separate diagnosis.', '- No automatic driver temperature/throttle telemetry available. OOM/NaN/Inf/process failures stop the stage; no blind restart.', '- Read-only review found no outstanding Critical/Important issues. Initial stale-exit concern retracted after2host subprocess tests.', '- Native suite135tests passed; git diff --check passed with existing CRLF conversion warnings. No unrelated UI refactor.', '', '## 11. WHAT I SHOULD REVIEW TOMORROW','','1. Individual heavy/codec/video regressions against historicalV3, not composite alone.', '2. Retrained-control results:recipe/budget effects versus architecture effects.', '3. Near-zero SPAB chain and measured AMP zeroing; identity-bypass smoke versus original.', '4. Loss and residual-gain ablations, preserving image quality and motion limits before longer training.', '5. Latency/P95, resume-state gap, frozen final-hidden gate and no automatic product release.', '', 'Full configs/checkpoints/hashes/history/source reports preserved in separate run directories. Source commit021d0c724aa19a1d31bb9b68a997c906f4bcead6 plus recorded uncommitted source hashes; no commit/push in this execution.']
 (ROOT/'docs/zenkai-overnight-report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print('Overnight report regenerated from measurements')
if __name__=='__main__':main()
