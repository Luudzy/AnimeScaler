"""Report checkpoint-local gradients separately from accumulated weight changes."""
import json
from pathlib import Path
from build_codec3k_report import table

ROOT = Path(__file__).resolve().parents[2]


def main():
    operation = json.loads((ROOT/'diagnostics/zenkai-c5k/gradients/operation.json').read_text())
    if operation['state'] != 'PASS':
        raise ValueError('Only completed gradient diagnostics can be reported')
    report = json.loads(Path(operation['result']['path']).read_text())
    rows = []
    for case in report['cases']:
        for name, fp32 in case['fp32']['groups'].items():
            amp = case['amp']['groups'][name]
            rows.append([case['case'], name, fp32['gradientL2'], amp['gradientL2'],
                fp32['zeroElementsPct'], amp['zeroElementsPct'],
                fp32['missingElementsPct'], amp['missingElementsPct']])
    lines = ['# C5k gradient analysis', '',
        'Completed checkpoint-local backward on the pinned clean and heavy sample in FP32 and scaled AMP. No optimizer update, checkpoint modification or new training was performed. This does not measure every minibatch or prove absence of all gate saturation/underflow.', '',
        f"Snapshot loss scale: {report['actualSnapshotLossScale']}. Master SHA256: `{report['checkpointSha256']}`.", '',
        'The four full blocks have nonzero gradients in both sampled domains. The identity-skip candidate has no dead-block evidence in these probes. This is separate from the earlier original A/B gradient problem.', '']
    lines += table(['Domain', 'Group', 'FP32 gradient L2', 'AMP unscaled gradient L2',
        'FP32 zeros %', 'AMP zeros %', 'FP32 missing %', 'AMP missing %'], rows)
    for key, title in [('accumulatedInitTo5k', 'Initialization to5k'), ('accumulated3kTo5k', 'Actual3k to5k')]:
        lines += ['', '## '+title, '',
            'Accumulated checkpoint difference, not a single-step update-to-weight ratio. The denominator is the weight L2 at the start of this interval. AdamW steps are recorded in the full state; no missing per-step instrumentation is fabricated.', '']
        lines += table(['Group', 'Start weight L2', 'Accumulated delta L2', 'Delta/start ratio',
            'Mean absolute change', 'Changed elements %'], [[name, r['initialL2'], r['deltaL2'],
            r['deltaToInitialRatio'], r['meanAbsChange'], r['changedElementsPct']]
            for name, r in report[key].items()])
    lines += ['', 'True per-step update norms over the historical run: NOT MEASURED. Attention saturation and individual GRep branch attribution are not established by these group norms alone. Raw diagnostics preserve all groups and sample hashes.', '']
    (ROOT/'docs/zenkai-c5k-gradient-analysis.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
