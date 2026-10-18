"""Save the matched smoke-run evidence; never approves a candidate."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
def main():
 quality=read('diagnostics/zenkai-ablations/loss-1k-quality/quality.json')
 assert quality['state']=='complete'
 lines=['# Zenkai controlled short ablations','','All runs are screening probes, not product models. Each changes one variable versus new same-seed frequency1k reference. OriginalA/B5k never modified. No final-hidden, perceptual/GAN loss, headx1.5 or Vulkan kernel.', '', '## Matched loss pair1k', '', 'Same train/validation/codec manifests, online sampler/crops, seed929, patch128, batch4×2, optimizer and first1k prefix of5k schedule. Small rerun variance versus originalA1k was observed, so a new matched reference was trained. Frequency loss is Charbonnier+HF+DCT; Charbonnier-only removes the frequency terms as one loss-profile variable. LPIPS remains evaluation only.', '', '| Domain | Metric | Better | Frequency | Charbonnier | Charbonnier/reference % |', '|---|---|---|---:|---:|---:|']
 for domain,models in quality['cases'].items():
  a=models['zenkai-frequency-1k'];b=models['zenkai-charbonnier-1k']
  for metric in a:
   direction='higher' if metric in ('psnr','ssim','msSsim') else 'lower'
   lines.append(f'| {domain} | {metric} | {direction} | {a[metric]:.9f} | {b[metric]:.9f} | {(b[metric]/a[metric]-1)*100:+.4f} |')
 lines+=['', 'Charbonnier-only does not show coherent benefit:cleanMAE+0.945%, EdgeMAE+1.329%, LPIPS+4.987%; heavy/codec mostly similar or worse. Isolated tinySSIM/H264edge improvements do not establish a winner. Do not extend Charbonnier automatically. This single-seed1k comparison does not prove it can never improve with longer training.', '', '## Output residual gain and attention path probes', '', 'Output gain.2 changes only the learned output correction multiplier from1.0→.2; SPAB gates unchanged. Identity bypass changes only per-SPAB skip gain0→1, including tail, keeping original centered gate and all underlying seeded parameters. Wrappers consume no extra RNG draws. Family tags prevent incorrect ordinary export. Deployment equivalence tests passed on nonzero heads. No new model UI/default.', '', 'Probe quality covers clean/heavy32each at1k, all10metrics. Full fixed clean/heavy/H264/H265/AV1 evaluation and ROCm fusion/FP16 parity completed for all probes. Real video and performance at1k:NOT MEASURED; matched3k pair is being measured separately.', '', '| Experiment | Steps | Duration s | Composite descriptive | Status |', '|---|---:|---:|---:|---|']
 reports={}
 for run in ['zenkai-ablation-frequency-1k','zenkai-ablation-charbonnier-1k','zenkai-ablation-output-residual-02-1k','zenkai-ablation-spab-identity-skip-1k']:
  folder=ROOT/'models/zenkai/experiments'/run;rp=folder/'report.json'
  if not rp.exists():lines.append(f'| {run} | NOT MEASURED | — | — | PENDING |');continue
  r=json.loads(rp.read_text(encoding='utf-8'));reports[run]=r
  lines.append(f"| {run} | {r['stepsExecuted']} | {r['trainingSeconds']:.3f} | {r['history'][-1]['summary']['composite']:.9f} | recorded, experimental |")
  meta={'run':run,'stepFinal':r['stepsExecuted'],'trainingSeconds':r['trainingSeconds'],'config':read(folder/'config.json'),
    'validationHistory':r['history'],'best':r['best'],'lastTrainingLoss':r['history'][-1]['trainingLoss'],
    'checkpointsSha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('*.pt')},
    'releaseApproved':False,'resumeStatus':'Optimizer/scaler/RNG/data RNG/sampler state saved; restricted matched frequency/skip1k-to3k continuation implemented and CPU state-equivalence tested'}
  state='ablations1' if run.endswith(('frequency-1k','charbonnier-1k')) else 'residual-probe' if 'residual' in run else 'skip-probe'
  s=read(f'diagnostics/zenkai-training/{state}.status.json');meta['runnerStatus']=s
  meta['trainingExitCode']=0 if s['state']=='complete' else 'NOT RECORDED HERE: consult stage log/status'
  (folder/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
 base=reports['zenkai-ablation-frequency-1k']
 for run,rr in reports.items():
  if run==base['run']:continue
  lines+=['',f'### {run}: every clean/heavy metric', '', '| Domain | Metric | Reference | Probe | Probe/reference % |','|---|---|---:|---:|---:|']
  for domain,models in rr['history'][-1]['screening']['cases'].items():
   if domain not in ('clean','heavy'):continue
   for metric,b in models['candidate'].items():
    a=base['history'][-1]['screening']['cases'][domain]['candidate'][metric]
    lines.append(f'| {domain} | {metric} | {a:.9f} | {b:.9f} | {(b/a-1)*100:+.4f} |')
 lines+=['', '## Decisions and limitations', '', 'Four1k smokes completed. Skip restores deep-block learning but quality remains mixed. Only matched frequency/skip pair continues1k→3k TOTAL with optimizer/scaler/RNG/sampler retained and original5k LR schedule. No15k Stage2. Individual metrics outweigh composite. New132-test suite passed; CPU resumed stochastic optimizer update matches uninterrupted reference. This CPU test does not claim bitwise full ROCm training equivalence.', '', 'CPU/GPU gradient diagnostics support deep-chain attenuation and AMP zeroing; counterfactual graph changes increase gradients but do not certify restored image quality. Preserve failed hypotheses alongside promising ones.', '', 'Raw paired quality:`diagnostics/zenkai-ablations/loss-1k-quality/quality.json`. Config/history/checkpoints/hashes in separate `models/zenkai/experiments/zenkai-ablation-*` directories.']
 full=read('diagnostics/zenkai-ablations/probes-1k-quality/quality.json')
 assert full['state']=='complete'
 lines+=['','## Full fixed probe comparison1k','','All160 fixed samples:32/domain; each percentage compares with matched frequency1k. Errors lower, PSNR/SSIM higher.','']
 for domain,models in full['cases'].items():
  ref=models['zenkai-frequency-1k']
  lines+=['',f'### {domain}','','| Metric | V3 | Frequency | Output gain.2 | Skip | Skip/ref % |','|---|---:|---:|---:|---:|---:|']
  for metric,a in ref.items():
   skip=models['zenkai-identity-skip-1k'][metric]
   lines.append(f"| {metric} | {models['adore-lite-v3'][metric]:.9f} | {a:.9f} | {models['zenkai-output-gain-02-1k'][metric]:.9f} | {skip:.9f} | {(skip/a-1)*100:+.4f} |")
 lines+=['','Actual stored initial→1k block1 update-to-weight ratio:frequency0; skip0.22656299. Skip AMP exact-zero-gradient fraction0% versus frequency~99.996%. These are accumulated changes, not per-step optimizer updates. Raw:`diagnostics/zenkai-ablation-gradient-analysis.json`.','']
 status=read('diagnostics/zenkai-training/ablations3k.status.json')
 lines+=['## Matched3k continuation', '',f"Runner state:{status['state']}; phase:{status['phase']}. Artifacts have separate continuation folders, originals immutable. Fixedcodec,50/500 performance and same144 real-video frames use protocols prepared before observing3k results."]
 (ROOT/'docs/zenkai-ablations.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print('Ablation report and available metadata saved')
if __name__=='__main__':main()
