"""Explicit registry opt-in for parity-checked Zenkai experimental artifacts."""
import hashlib
import json
from pathlib import Path
import re
from model_registry import ModelMetadata


def register_zenkai_export(registry, folder, model_id):
    folder = Path(folder).resolve()
    if not re.fullmatch(r'zenkai-[a-z0-9-]+', model_id):
        raise ValueError('Experimental identity must start with zenkai-')
    report = json.loads((folder / 'export-report.json').read_text(encoding='utf-8'))
    if (report.get('schemaVersion') != 1 or report.get('state') != 'complete' or
            report.get('parityApproved') is not True or report.get('family') != 'Zenkai' or
            report.get('status') != 'experimental' or report.get('nativeScale') != 2):
        raise ValueError('Experimental export has not passed numeric validation')
    config = report['training']
    if config.get('channels') != 32 or config.get('depth') not in (5, 6):
        raise ValueError('Unreviewed experimental architecture')
    expected = report.get('artifacts', {})
    paths = {name: folder / name for name in ('deploy-fp32.pt', 'dynamic-fp32.onnx')}
    for name, path in paths.items():
        if name not in expected or hashlib.sha256(path.read_bytes()).hexdigest() != expected[name]['sha256']:
            raise ValueError(f'Experimental artifact changed: {name}')

    def validate(path, digest):
        if path.parent.resolve() != folder or path.name not in paths or digest != expected[path.name]['sha256']:
            raise ValueError(f'Experimental artifact identity mismatch: {path.name}')

    def factory():
        import torch
        from zenkai_model import ZenkaiLite
        saved = torch.load(paths['deploy-fp32.pt'], map_location='cpu', weights_only=True)
        if not saved.get('config', {}).get('deploy') or saved.get('sourceSha256') != report['checkpointSha256']:
            raise ValueError('Deploy checkpoint does not match export report')
        state = saved['model']
        if any(value.dtype != torch.float32 or not torch.isfinite(value).all() for value in state.values()):
            raise ValueError('Invalid deploy FP32 state')
        model = ZenkaiLite(channels=config['channels'], depth=config['depth']).deploy_copy()
        model.load_state_dict(state, strict=True)
        return model

    meta = ModelMetadata(model_id, 'Zenkai Lite V1 experimental', 'Zenkai',
                         'ESPAN-inspired GRep SPAB x2', 2, 'experimental',
                         provider='AnimeScaler experimental training',
                         code_license='MIT-derived ESPAN components; see THIRD_PARTY_NOTICES.md',
                         weights_path=str(paths['deploy-fp32.pt']), capture_graph=False)
    registry.register(meta, factory, paths['dynamic-fp32.onnx'], artifact_validator=validate)
    return meta


def add_zenkai_exports(registry, assignments):
    for assignment in assignments or ():
        model_id, separator, folder = assignment.partition('=')
        if not separator or not folder:
            raise ValueError('Use --zenkai-export zenkai-ID=EXPORT_DIRECTORY')
        register_zenkai_export(registry, folder, model_id)


def register_control_checkpoint(registry,path):
    from zenkai_experiments import load_control_checkpoint
    path=Path(path).resolve();expected=hashlib.sha256(path.read_bytes()).hexdigest()
    load_control_checkpoint(path)
    def validate(artifact,digest):
        if digest!=expected:raise ValueError('Control artifact changed')
    def factory():return load_control_checkpoint(path)[0]
    meta=ModelMetadata('adore-v3-retrained-control','Adore V3 Retrained Control','AdoreControl',
        'Exact historical AdoreLiteV3 24ch4blocks',2,'experimental',weights_path=str(path),capture_graph=False)
    registry.register(meta,factory,artifact_validator=validate)
    return meta


def register_probe_checkpoint(registry,path,expected_step=3000):
    from zenkai_probe_evaluation import load_probe
    path=Path(path).resolve();expected=hashlib.sha256(path.read_bytes()).hexdigest()
    _,saved=load_probe(path,expected_step)
    config=saved['config'];model_id=config['run']
    def validate(artifact,digest):
        if Path(artifact).resolve()!=path or digest!=expected:raise ValueError('Probe artifact changed')
    def factory():return load_probe(path,expected_step)[0].deploy_copy()
    meta=ModelMetadata(model_id,'Zenkai smoke probe',config['family'],'Experimental SPAB identity/output gain probe',
        2,'experimental',weights_path=str(path),capture_graph=False)
    registry.register(meta,factory,artifact_validator=validate)
    return meta
