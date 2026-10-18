"""Pair measured A/B real-video reports without silently hiding regressions."""
import argparse
import json
import math
import copy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ('mae', 'referenceCompensatedTemporalMae')


def merge_completed_sequences(first,rest):
    if first.get('state') not in ('running','complete') or rest.get('state')!='complete':
        raise ValueError('Only fully recorded sequence results may be merged')
    for field in ('backend','precision','torch','cutPolicy','artifacts'):
        if field not in first or first[field]!=rest.get(field):raise ValueError(f'Merge protocol mismatch: {field}')
    merged=copy.deepcopy(first)
    merged['sequences']=copy.deepcopy(first['sequences']+rest['sequences'])
    hashes=[row['manifestSha256'] for row in merged['sequences']]
    if len(set(hashes))!=len(hashes):raise ValueError('Duplicate sequence cannot stand in for missing frames')
    merged['state']='complete'
    merged['interruptionNote']='Initial CPU process interrupted after completed sequences; remaining sequence evaluated separately'
    return merged


def compare_reports(a, b):
    if a.get('state') != 'complete' or b.get('state') != 'complete':
        raise ValueError('Both reports must be complete')
    for field in ('backend', 'precision', 'torch', 'cutPolicy'):
        if field not in a or a[field] != b.get(field):
            raise ValueError(f'Unmatched runtime/protocol: {field}')
    if a['artifacts']['adore-lite-v3']['sha256'] != b['artifacts']['adore-lite-v3']['sha256']:
        raise ValueError('V3 artifact mismatch')
    for report, name in ((a, 'zenkai-a'), (b, 'zenkai-b')):
        if report['artifacts'][name].get('step') != 5000:
            raise ValueError('Expected matched step-5000 candidates')
    sequences_a, sequences_b = a['sequences'], b['sequences']
    if not sequences_a or len(sequences_a) != len(sequences_b):
        raise ValueError('Unmatched sequences')
    result = {'state': 'complete', 'releaseApproved': False,
              'scope': 'Paired CPU FP32 screening; relative metric changes are not perceptual certification',
              'artifacts': {'a': a['artifacts'], 'b': b['artifacts']}, 'sequences': []}
    for first, second in zip(sequences_a, sequences_b):
        for field in ('manifestSha256', 'frameCount', 'suspectedCutTransitions'):
            if field not in first or first[field] != second.get(field):
                raise ValueError(f'Unmatched sequence protocol: {field}')
        values = {'a': first['models']['zenkai-a'], 'b': second['models']['zenkai-b'],
                  'v3': first['models']['adore-lite-v3']}
        baseline_b = second['models']['adore-lite-v3']
        if values['v3'].get('includedTransitions') != baseline_b.get('includedTransitions'):
            raise ValueError('Unmatched temporal transitions')
        for rows in values.values():
            if rows.get('includedTransitions', 0) < 1 or rows['includedTransitions'] != values['v3']['includedTransitions']:
                raise ValueError('Missing or unmatched temporal transitions')
            for metric in METRICS:
                value = rows.get(metric)
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError('Invalid numeric metric')
        for metric in METRICS:
            if not math.isclose(values['v3'][metric], baseline_b.get(metric, float('nan')), rel_tol=1e-6, abs_tol=1e-8):
                raise ValueError('V3 repeat differs; investigate before comparison')
        row = {'manifest': first['manifest'], 'manifestSha256': first['manifestSha256'],
               'suspectedCutTransitions': first['suspectedCutTransitions'],
               'values': {name: {metric: data[metric] for metric in METRICS} for name, data in values.items()}}
        for label, source in (('bVersusV3', 'v3'), ('bVersusA', 'a')):
            row[label] = {metric: {'absoluteChange': values['b'][metric] - values[source][metric],
                'changePercent': (values['b'][metric] / values[source][metric] - 1) * 100 if values[source][metric] else None,
                'regression': values['b'][metric] > values[source][metric]} for metric in METRICS}
        result['sequences'].append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a', type=Path, required=True)
    parser.add_argument('--b', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        raise ValueError('Output must be a new workspace file')
    report = compare_reports(json.loads(args.a.read_text(encoding='utf-8')),
                             json.loads(args.b.read_text(encoding='utf-8')))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'state': report['state'], 'output': str(output), 'releaseApproved': False}))


if __name__ == '__main__': main()
