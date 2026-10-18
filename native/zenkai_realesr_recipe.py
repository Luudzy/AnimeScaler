"""Configuration and artifact guard for the Zenkai-RealESR Lite smoke."""
import hashlib
import json
import re
from pathlib import Path

RECIPE_ID = 'zenkai-realesr-lite-v1'
RUN_ID = 'zenkai-realesr-lite-v1-smoke-500'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def make_recipe(source_config, run=RUN_ID, steps=500):
    if not re.fullmatch(re.escape(RUN_ID) + r'(?:-retry[1-9][0-9]*)?', run) and not re.fullmatch(r'zenkai-(?:realesr-lite-v1|animesr-lite-v1)-[a-z0-9-]+-500', run):
        raise ValueError('Smoke uses a reserved or explicitly named 500-step identity')
    if steps != 500:
        raise ValueError('Smoke must be exactly 500 steps')
    source = dict(source_config)
    return {
        **source,
        'family': 'Zenkai',
        'name': 'Zenkai-RealESR Lite experimental',
        'run': run,
        'experimentKind': RECIPE_ID,
        'architecture': 'SRVGG-inspired PReLU',
        'channels': 32,
        'blocks': 6,
        'nativeScale': 2,
        'steps': steps,
        'releaseApproved': False,
        'promoted': False,
        'realVideoValidation': 'PENDING',
    }


def create_run_dir(root, recipe):
    root = Path(root)
    path = root / 'models' / 'zenkai' / 'experiments' / recipe['run']
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite experiment: {path}')
    path.mkdir(parents=True)
    (path / 'config.json').write_text(json.dumps(recipe, indent=2), encoding='utf-8')
    return path


def verify_baseline(path, expected_sha256):
    actual = sha256(path)
    if actual != expected_sha256:
        raise ValueError(f'Baseline artifact changed: {path}')
    return actual
