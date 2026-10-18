"""Fail closed when a comparison uses different or missing pinned artifacts."""
import hashlib
from pathlib import Path


def validate_checkpoint_file(path, pin):
    if not pin or hashlib.sha256(Path(path).read_bytes()).hexdigest() != pin:
        raise ValueError(f'checkpoint file hash mismatch: {path}')


def validate_artifact_pins(observed, expected):
    for label, pin in expected.items():
        if observed.get(label, {}).get('sha256') != pin:
            raise ValueError(f'Comparison artifact hash changed or missing: {label}')
