"""Crash-safe replacement for small JSON reports and training checkpoints."""
import json
import hashlib
import os
from pathlib import Path
import shutil
import tempfile


def _atomic_write(path, write_payload):
    target = Path(path)
    if not target.parent.is_dir():
        raise FileNotFoundError(f'Artifact directory does not exist: {target.parent}')
    descriptor, temporary = tempfile.mkstemp(prefix=f'.{target.name}.', suffix='.pending', dir=target.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            write_payload(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _atomic_create(path, write_payload, artifact_kind):
    """Write a new file atomically and fail if another writer claimed the path."""
    target = Path(path)
    if not target.parent.is_dir():
        raise FileNotFoundError(f'Artifact directory does not exist: {target.parent}')
    descriptor, temporary = tempfile.mkstemp(prefix=f'.{target.name}.', suffix='.pending', dir=target.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            write_payload(stream)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise FileExistsError(f'Refusing to overwrite {artifact_kind}: {target}') from error
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def atomic_torch_save(value, path):
    """Serialize in the target directory, then atomically replace the file."""
    import torch
    _atomic_write(path, lambda stream: torch.save(value, stream))


def atomic_torch_save_new(value, path):
    """Serialize a new checkpoint without replacing a path claimed by another writer."""
    import torch
    _atomic_create(path, lambda stream: torch.save(value, stream), 'checkpoint')


def _copy_checkpoint(source_path, stream):
    with Path(source_path).open('rb') as input_stream:
        shutil.copyfileobj(input_stream, stream)


def atomic_copy_file(source_path, path):
    """Copy a checkpoint atomically, replacing the destination after a complete copy."""
    _atomic_write(path, lambda stream: _copy_checkpoint(source_path, stream))


def atomic_copy_file_new(source_path, path):
    """Copy a new checkpoint atomically without replacing a path claimed by another writer."""
    _atomic_create(path, lambda stream: _copy_checkpoint(source_path, stream), 'checkpoint')


def atomic_json_write(value, path):
    """Reject non-finite report values before atomically replacing JSON."""
    encoded = json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False).encode('utf-8')
    _atomic_write(path, lambda stream: stream.write(encoded))


def atomic_json_create(value, path):
    """Create a new JSON report atomically without replacing a path claimed by another writer."""
    encoded = json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False).encode('utf-8')
    _atomic_create(path, lambda stream: stream.write(encoded), 'report')


def publish_training_checkpoint(value, checkpoint_path, index_path, index, checkpoint_key,
                               completed_step, last_checkpoint, report_path=None, report=None):
    """Publish checkpoint first, then hash index; retain old index on failure."""
    checkpoint=Path(checkpoint_path)
    if checkpoint.exists():
        raise FileExistsError(f'Refusing to overwrite checkpoint: {checkpoint}')
    checkpoints=index.get('checkpoints',{})
    if not isinstance(checkpoints,dict):
        raise ValueError('Invalid checkpoint index')
    updated={**index,'checkpoints':dict(checkpoints)}
    updated['checkpoints'][checkpoint_key]='pending-digest'
    updated.update(lastCompleteStep=completed_step,lastCheckpoint=last_checkpoint)
    json.dumps(updated,allow_nan=False)
    if report_path is not None:
        json.dumps(report,allow_nan=False)
    atomic_torch_save_new(value,checkpoint)
    try:
        digest=hashlib.sha256()
        with checkpoint.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
        updated['checkpoints'][checkpoint_key]=digest.hexdigest()
        atomic_json_write(updated,index_path)
    except BaseException:
        try: checkpoint.unlink()
        except FileNotFoundError: pass
        raise
    if report_path is not None:
        atomic_json_write(report,report_path)
    return updated
