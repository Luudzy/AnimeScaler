"""Local model inventory without loading models or reading private credentials."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'native'))
STORAGE_ROOTS = (
    ('models', 'model_or_experiment'),
    ('diagnostics/native-cache', 'runtime_cache'),
)


def _raise_walk_error(error):
    raise error


def build_storage_inventory(root=ROOT):
    root = Path(root).resolve()
    entries = []
    for relative_root, category in STORAGE_ROOTS:
        base = root / relative_root
        if not base.exists() or base.is_symlink():
            continue
        if not base.is_dir():
            raise NotADirectoryError(base)
        for directory, child_directories, filenames in os.walk(
                base, topdown=True, followlinks=False, onerror=_raise_walk_error):
            current = Path(directory)
            child_directories[:] = sorted(
                name for name in child_directories if not (current / name).is_symlink())
            for name in sorted(filenames):
                path = current / name
                if path.is_symlink():
                    continue
                info = path.stat(follow_symlinks=False)
                if not path.is_file():
                    continue
                entries.append({
                    'path': path.relative_to(root).as_posix(),
                    'category': category,
                    'bytes': info.st_size,
                    'modifiedUtc': datetime.fromtimestamp(info.st_mtime, timezone.utc).isoformat(),
                    'protected': True,
                    'retention': 'user_authorization_required',
                })
    entries.sort(key=lambda item: item['path'])
    return {
        'capturedAt': datetime.now(timezone.utc).isoformat(),
        'scope': [relative_root for relative_root, _ in STORAGE_ROOTS],
        'totals': {'files': len(entries), 'bytes': sum(item['bytes'] for item in entries)},
        'files': entries,
    }


def sha256_file(path, chunk_size=1024 * 1024):
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError('Hash chunk size must be a positive integer')
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--storage-only', action='store_true',
                        help='inventory model/cache sizes without importing the model registry')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous inventory')
    entries = []
    if not args.storage_only:
        from model_registry import build_registry
        for metadata in build_registry(ROOT).list_models():
            item = metadata.to_dict()
            file = Path(item.pop('weights_path'))
            item.update(path=file.relative_to(ROOT).as_posix(), bytes=file.stat().st_size,
                        sha256=sha256_file(file))
            entries.append(item)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        'scope': ('Read-only model/cache storage inventory; no registry imports or inference'
                  if args.storage_only else
                  'Available local models plus read-only storage inventory; no inference or quality endorsement'),
        'models': entries,
        'storage': build_storage_inventory(ROOT),
    }
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'models': len(entries), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
