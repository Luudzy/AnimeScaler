"""Verify original and experimental artifacts against saved, not newly inferred hashes."""
import datetime
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()

def main():
    summary={'checkedUtc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'runs':{},'releaseApproved':False}
    for run in ['zenkai-lite-v1-a-5k','zenkai-lite-v1-b-5k','adore-v3-retrained-control-5k',
        'zenkai-ablation-frequency-1k','zenkai-ablation-charbonnier-1k','zenkai-ablation-output-residual-02-1k',
        'zenkai-ablation-spab-identity-skip-1k','zenkai-ablation-frequency-3k-continuation',
        'zenkai-ablation-spab-identity-skip-3k-continuation']:
        folder=ROOT/'models/zenkai/experiments'/run;m=read(folder/'metadata.json')
        pins=m.get('checkpointsSha256') or {r['file']:r['sha256'] for r in m['checkpoints']}
        assert pins
        for name,digest in pins.items():assert sha(folder/name)==digest,(run,name)
        for name,digest in m.get('evaluationArtifactsSha256',{}).items():assert sha(folder/name)==digest,(run,name)
        for row in m.get('evaluationArtifacts',{}).values():assert sha(Path(row['path']))==row['sha256'],(run,row['path'])
        summary['runs'][run]={'checkpoints':len(pins),'status':'unchanged by previously pinned SHA','step':m['stepFinal']}
    perf=read('diagnostics/zenkai-ablations/performance-3k/benchmark_results.json')
    assert perf['complete'] and len(perf['cases'])==6
    for row in perf['cases']:
        assert sha(Path(row['artifactPath']))==row['artifactSha256']
        if row['model']=='adore-lite-v3':
            assert row['artifactSha256']=='aa1a50da4e74ba342f598f92bafc73e9678f68c214f637574d354b20452500cf'
            summary['historicalV3']='previously pinned SHA unchanged'
    import sys
    sys.path.insert(0,str(ROOT/'native'))
    from frozen_video_protocol import validate_protocol
    for path in ['diagnostics/zenkai-real-video-frequency-3k/report.json','diagnostics/zenkai-real-video-skip-3k-complete/report.json']:
        validate_protocol(read(path))
    assert read('diagnostics/zenkai-ablations/probes-3k-quality/quality.json')['state']=='complete'
    assert read('diagnostics/zenkai-training/ablations3k-gradients.exit.json')['exitCode']==0
    summary.update(fixedQuality='complete',videoFramesEach=144,gradient3k='measured, exit0')
    (ROOT/'diagnostics/zenkai-overnight/final-integrity.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary))

if __name__=='__main__':main()
