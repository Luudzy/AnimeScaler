"""Final factor audit against preserved training snapshots and report pins."""
import datetime
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(ROOT/'native'))
from final_integrity import main as verify_prior,read,sha
from verify_skip_loss_integrity import main as verify_losses
from frozen_video_protocol import validate_protocol


def main():
    verify_prior()
    verify_losses()
    runs={}
    for factor in ('dataset-sampling','crop96','codec50'):
        run='zenkai-ablation-skip-factor-'+factor+'-1k'
        folder=ROOT/'models/zenkai/experiments'/run
        metadata=read(folder/'metadata.json');config=read(folder/'config.json')
        assert metadata['stepFinal']==1000 and metadata['trainingExitCode']==0 and not metadata['releaseApproved']
        for group in ('checkpointsSha256','evaluationArtifactsSha256'):
            for name,digest in metadata[group].items():assert sha(folder/name)==digest,(run,name)
        for name,digest in config['sourceFilesSha256'].items():assert sha(folder/'source'/name)==digest,(run,name)
        pair=read(folder/'initial-pairing.json')
        assert pair['state']=='matched' and sha(Path(pair['reference']))==pair['referenceSha256']
        for stage,record in metadata['stageExitRecords'].items():
            assert sha(record['path'])==record['sha256'] and read(record['path'])['exitCode']==0,stage
        video=read(folder/'real-video-report.json');validate_protocol(video)
        runs[run]={'checkpoints':len(metadata['checkpointsSha256']),
                   'sourceSnapshots':len(config['sourceFilesSha256']), 'videoFrames':144,
                   'metadataSha256':sha(folder/'metadata.json')}
    interrupted=ROOT/'models/zenkai/experiments/zenkai-ablation-skip-factor-scheduler-constant-1k'
    config=read(interrupted/'config.json')
    for name,digest in config['sourceFilesSha256'].items():assert sha(interrupted/'source'/name)==digest
    assert not (interrupted/'step-1000-master-fp32.pt').exists()
    panels=read('diagnostics/zenkai-real-video-comparison-crops-factors-1k/provenance.json')
    assert len(panels['comparisons'])==9
    for comparison in panels['comparisons']:
        assert sha(Path(comparison['output']))==comparison['outputSha256']
        for row in comparison['inputs'].values():assert sha(Path(row['path']))==row['sha256']
    results=read('diagnostics/zenkai-factor-screens-results.json')
    assert len(results['qualityRows'])==150 and len(results['videoRows'])==27
    assert sha('docs/zenkai-factor-screens.md')==results['reportSha256']
    assert results['unavailableFactors']['scheduler-constant'].startswith('FAILED')
    summary={'checkedUtc':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'runs':runs,
        'earlierPinnedCheckpoints':62,'priorLossCheckpoints':8,'historicalV3Unchanged':True,
        'fixedSamplesEach':160,'videoFramesEach':144,'qualityRows':150,'videoRows':27,'cropPanels':9,
        'failedSchedulerInitialSha256':sha(interrupted/'initial-training-state.pt'),
        'schedulerStatus':'FAILED, no post-initial checkpoint, not restarted', 'releaseApproved':False}
    (ROOT/'diagnostics/zenkai-overnight/factor-final-integrity.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary))


if __name__=='__main__':main()
