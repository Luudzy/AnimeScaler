"""Measure parameter movement between initialization and the 500-step smoke."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--initial', type=Path, required=True)
    parser.add_argument('--final', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    initial = torch.load(args.initial, map_location='cpu', weights_only=False)['model']
    final = torch.load(args.final, map_location='cpu', weights_only=False)['model']
    grouped = defaultdict(lambda: {'weightNorm': 0.0, 'updateNorm': 0.0, 'elements': 0})
    for name, before in initial.items():
        after = final[name]
        prefix = name.split('.')[0] if not name.startswith('blocks.') else '.'.join(name.split('.')[:2])
        row = grouped[prefix]
        row['weightNorm'] += float(after.float().norm()) ** 2
        row['updateNorm'] += float((after.float() - before.float()).norm()) ** 2
        row['elements'] += before.numel()
    result = {'state': 'complete', 'initial': str(args.initial.resolve()), 'final': str(args.final.resolve()), 'blocks': {}}
    for name, row in grouped.items():
        row['weightNorm'] = row['weightNorm'] ** 0.5
        row['updateNorm'] = row['updateNorm'] ** 0.5
        row['updateToWeight'] = row['updateNorm'] / row['weightNorm'] if row['weightNorm'] else None
        row['status'] = 'active' if row['updateToWeight'] is not None and row['updateToWeight'] >= 1e-3 else 'small-update'
        result['blocks'][name] = row
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'state': result['state'], 'blocks': len(result['blocks']), 'output': str(args.output)}))


if __name__ == '__main__': main()
