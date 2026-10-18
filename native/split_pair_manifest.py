"""Create a source-disjoint train/holdout split from an unsplit pair manifest."""
import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from prepare_real_video_pairs import _rename_directory_new, partition_clip_rows


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_asset(value, input_root, output_root):
    path = Path(value)
    if not path.is_absolute(): path = input_root / path
    try:
        return Path(os.path.relpath(path.resolve(), output_root.resolve())).as_posix()
    except ValueError as error:
        raise ValueError('Source media and output must share a volume for portable relative paths') from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Directory containing an unsplit manifest.jsonl')
    parser.add_argument('--output', type=Path, required=True, help='New output directory; existing outputs are preserved')
    parser.add_argument('--holdout-every', type=int, default=10,
                        help='Assign every Nth sorted source group to holdout (minimum 2)')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    source_manifest = args.input / 'manifest.jsonl'
    source_bytes = source_manifest.read_bytes()
    try:
        rows = [json.loads(line) for line in source_bytes.decode('utf-8').splitlines() if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f'Invalid JSONL manifest: {source_manifest}') from error
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('file'), str) or not row['file']:
            raise ValueError('Every pair row requires a file path')
        digest = row.get('sha256')
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in '0123456789abcdefABCDEF' for char in digest):
            raise ValueError('Every pair row requires a valid SHA-256')
    train, holdout = partition_clip_rows(rows, args.holdout_every)
    input_root = args.input.resolve()
    output_root = args.output.resolve()
    for row in train + holdout:
        row['file'] = _relative_asset(row['file'], input_root, output_root)
        if row.get('pairedLr') is not None:
            row['pairedLr'] = _relative_asset(row['pairedLr'], input_root, output_root)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{args.output.name}.', suffix='.pending', dir=args.output.parent))
    try:
        train_path = staging / 'manifest.jsonl'
        holdout_path = staging / 'holdout.jsonl'
        train_path.write_text('\n'.join(json.dumps(row) for row in train) + '\n', encoding='utf-8')
        holdout_path.write_text('\n'.join(json.dumps(row) for row in holdout) + '\n', encoding='utf-8')
        metadata = {
            'state': 'complete',
            'trainPairs': len(train),
            'holdoutPairs': len(holdout),
            'source': Path(os.path.relpath(input_root, output_root)).as_posix(),
            'inputManifestSha256': _sha256(source_manifest),
            'sourceSha256': _sha256(train_path),
            'holdoutSha256': _sha256(holdout_path),
            'splitStrategy': 'whole source clips; frames from one clip never cross splits',
            'pathPolicy': 'POSIX paths relative to the published dataset directory; source assets remain external.',
            'holdoutEverySourceClips': args.holdout_every,
            'sourceGroups': len({row['source'] for row in train + holdout}),
            'limitations': [
                'Source clip identity does not prove distinct production or scene; record production-level groups when available.'
            ],
        }
        (staging / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
        _rename_directory_new(staging, args.output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(json.dumps({'state': 'complete', 'trainPairs': len(train), 'holdoutPairs': len(holdout),
                      'sourceGroups': metadata['sourceGroups'], 'output': str(args.output)}))


if __name__ == '__main__':
    main()
