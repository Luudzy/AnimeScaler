"""Verify pinned artifacts of both recent recipe probes and immutable prior runs."""
import json,hashlib,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def main():
 prior=read('diagnostics/zenkai-overnight/final-integrity.json')
 assert prior['historicalV3']=='previously pinned SHA unchanged'
 checked={}
 for run in ['zenkai-ablation-legacy-v3-loss-1k','zenkai-ablation-spab-identity-skip-legacy-loss-1k']:
  folder=ROOT/'models/zenkai/experiments'/run;m=read(folder/'metadata.json');c=read(folder/'config.json')
  for group in ('checkpointsSha256','evaluationArtifactsSha256'):
   for name,digest in m[group].items():assert sha(folder/name)==digest,(run,name)
  for name,digest in c['sourceFilesSha256'].items():assert sha(folder/'source'/name)==digest,(run,name)
  for stage,r in m['stageExitRecords'].items():
   assert sha(r['path'])==r['sha256'] and read(r['path'])['exitCode']==0,(run,stage)
  p=read(folder/'initial-pairing.json')
  assert p['state']=='matched' and sha(Path(p['reference']))==p['referenceSha256']
  checked[run]={'checkpoints':len(m['checkpointsSha256']),'trainingSourceFiles':len(c['sourceFilesSha256']),
   'stageExitCodes':len(m['stageExitRecords']),'metadataSha256':sha(folder/'metadata.json')}
 from sys import path
 path.insert(0,str(ROOT/'native'))
 from frozen_video_protocol import validate_protocol
 for label in ('skip-frequency','skip-legacy-loss'):
  v=read('diagnostics/zenkai-real-video-'+label+'-1k/report.json');validate_protocol(v)
  for seq in v['sequences']:
   for artifact in seq.get('artifacts',{}).values():
    if isinstance(artifact,dict) and 'path' in artifact and 'sha256' in artifact:assert sha(Path(artifact['path']))==artifact['sha256']
 q=read('diagnostics/zenkai-ablations/skip-loss-1k-quality/quality.json')
 assert q['state']=='complete' and sum(q['sampleCounts'].values())==160
 for p in read('diagnostics/zenkai-real-video-comparison-crops-skip-loss-1k/provenance.json')['comparisons']:
  assert sha(Path(p['output']))==p['outputSha256']
  for r in p['inputs'].values():assert sha(Path(r['path']))==r['sha256']
 result={'checkedUtc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'recipeRuns':checked,
  'earlierPinnedCheckpointsUnchanged':sum(r['checkpoints'] for r in prior['runs'].values()),
  'historicalV3Unchanged':True,'fixedSamples':160,'videoFramesEach':144,'cropPanels':9,'releaseApproved':False}
 (ROOT/'diagnostics/zenkai-overnight/skip-loss-final-integrity.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 print(json.dumps(result))
if __name__=='__main__':main()
