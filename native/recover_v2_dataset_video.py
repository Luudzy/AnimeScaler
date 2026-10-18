"""Complete a fresh full CPU video operation after orphaned coordinator interruption."""
import json,sys
from pathlib import Path
from hip_operation import run_operation,save_json
from frozen_video_protocol import validate_protocol
from prepare_video_degradation import file_hash
ROOT=Path(__file__).resolve().parents[1]
RUN='zenkai-ablation-skip-v2-dataset-1k'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
 base=ROOT/'diagnostics/zenkai-v2-dataset-1k';path=base/'pipeline.json';p=read(path)
 if p['state']!='RUNNING' or p['phase']!='video' or (base/'pipeline-before-orphan-recovery1.json').exists():raise ValueError('Not the recorded single orphaned operation')
 if any(p['results'][k]['state']!='PASS' for k in ('quality','gradients','benchmark')):raise ValueError('Preserve complete GPU operations')
 old=read(base/'video/operation.json')
 partial=read(base/'video/attempt-1/video/report.json')
 if old['state']!='RUNNING' or partial['state']!='running' or len(partial['sequences'])>=3:raise ValueError('Unexpected completed original video')
 save_json(base/'pipeline-before-orphan-recovery1.json',p)
 save_json(base/'video/operation-before-orphan-recovery1.json',old)
 old['state']='INCONCLUSIVE';old['reason']='Coordinator and CPU child no longer running; only incomplete running report and no exit code. Cause unknown; not HIP/model failure. Partial frames not used.'
 for a in old['attempts']:a['state']='INCONCLUSIVE';a['exitCode']=None
 save_json(base/'video/operation.json',old)
 p.update(phase='video-orphan-recovery1',interruption=old['reason']);save_json(path,p)
 frozen=read(ROOT/'diagnostics/zenkai-real-video-skip-frequency-1k/report.json');validate_protocol(frozen)
 checkpoint=ROOT/'models/zenkai/experiments'/RUN/'step-1000-master-fp32.pt';pin=file_hash(checkpoint)
 def command(a,out):
  args=[sys.executable,ROOT/'native/evaluate_real_video.py','--probe-checkpoint',checkpoint,'--probe-step','1000','--output',out/'video']
  for s in frozen['sequences']:args+=['--manifest',s['manifest']]
  return args
 def validate(out):
  result=out/'video/report.json';r=read(result);validate_protocol(r)
  if r['state']!='complete' or sum(s['frameCount'] for s in r['sequences'])!=144 or r['artifacts'].get(RUN,{}).get('sha256')!=pin:raise ValueError('Incomplete/wrong candidate')
  if r['precision']!=frozen['precision'] or r['backend']!=frozen['backend'] or r['cutPolicy']!=frozen['cutPolicy']:raise ValueError('Protocol changed')
  for a,b in zip(frozen['sequences'],r['sequences']):
   if a['manifestSha256']!=b['manifestSha256'] or a['suspectedCutTransitions']!=b['suspectedCutTransitions'] or RUN not in b['models']:raise ValueError('Frozen frame identity changed')
  return {'path':str(result),'sha256':file_hash(result)}
 p['results']['video']=run_operation(base/'video-orphan-recovery1',command,validate,timeout=1200)
 p.update(state='COMPLETE',phase='completed',releaseApproved=False);save_json(path,p)
 print(json.dumps({'pipeline':p['state'],'video':p['results']['video']['state']}))
if __name__=='__main__':main()
