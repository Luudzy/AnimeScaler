"""CPU metadata-only audit of the actual first8000 sampler indices; no image IO."""
import collections
import hashlib
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]


def main():
    manifest = ROOT / 'native/training-data/zenkai/manifest.jsonl'
    raw = manifest.read_bytes()
    rows = [json.loads(line) for line in raw.decode().splitlines() if line]
    rows = [row for row in rows if row['split'] == 'train']
    assert len(rows) == 1792
    groups = collections.Counter(row['cluster'] for row in rows)
    report = {'scope': 'Actual sampler prefix, metadata only; no final-hidden pixels or other pixels opened',
              'manifestSha256': hashlib.sha256(raw).hexdigest(), 'trainIdentities': len(rows),
              'heuristicGroupCount': len(groups), 'groupsTrainSize': dict(groups), 'runs': {}}
    for run in ('zenkai-ablation-spab-identity-skip-1k',
                'zenkai-ablation-skip-factor-dataset-sampling-1k'):
        path = ROOT / 'models/zenkai/experiments' / run / 'step-1000-training-state.pt'
        saved = torch.load(path, map_location='cpu', weights_only=True)
        indices = list(map(int, saved['samplerIndices'][:8000]))
        assert len(indices) == 8000 and saved['step'] == 1000
        assert all(0 <= index < len(rows) for index in indices)
        actual = collections.Counter(rows[index]['cluster'] for index in indices)
        expected = {group: 8000 / len(groups) if 'factor-dataset' in run else 8000 * size / len(rows)
                    for group, size in groups.items()}
        report['runs'][run] = {'trainingStateSha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'prefixSamples': 8000, 'uniqueTrainIdentitiesSampled': len(set(indices)),
            'groupsObserved': dict(actual), 'groupsExpectedFromPolicy': expected,
            'minimumGroupSamples': min(actual.get(group, 0) for group in groups),
            'maximumGroupSamples': max(actual.values()),
            'scope': 'Expected counts describe sampler policy, not quality or verified episode balance'}
    output = ROOT / 'diagnostics/zenkai-factor-sampling-analysis.json'
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({run: {k: r[k] for k in ('uniqueTrainIdentitiesSampled', 'minimumGroupSamples', 'maximumGroupSamples')}
                      for run, r in report['runs'].items()}))


if __name__ == '__main__':
    main()
