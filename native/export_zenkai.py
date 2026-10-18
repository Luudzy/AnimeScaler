"""Immutable experimental export with measured fusion, precision and ONNX gates.

Exporting does not register a model, enable an option or approve a release.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import torch
from zenkai_model import ZenkaiLite, GRepConv

ROOT = Path(__file__).resolve().parents[1]
TOLERANCES = {
    'fp32': {'max_error': 2e-5, 'mean_error': 2e-6},
    'fp16': {'max_error': .004, 'mean_error': .0005},
}


def configure_cache(root=ROOT):
    for key,folder in [('MIOPEN_USER_DB_PATH','db'),('MIOPEN_CUSTOM_CACHE_DIR','kernels')]:
        directory=Path(root)/'diagnostics/native-cache'/folder
        directory.mkdir(parents=True,exist_ok=True)
        os.environ.setdefault(key,str(directory))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parity_metrics(reference, actual):
    reference = torch.as_tensor(reference).detach().float().cpu()
    actual = torch.as_tensor(actual).detach().float().cpu()
    if reference.shape != actual.shape:
        raise ValueError('Parity output shape mismatch')
    if not torch.isfinite(reference).all() or not torch.isfinite(actual).all():
        raise ValueError('Non-finite parity output')
    error = (reference - actual).abs()
    return {'max_abs_error': float(error.max()), 'mean_abs_error': float(error.mean())}


def require_parity(scores, max_error, mean_error):
    if not (0 <= scores['max_abs_error'] <= max_error and
            0 <= scores['mean_abs_error'] <= mean_error):
        raise ValueError(f'Parity gate failed: {scores}; limits max={max_error}, mean={mean_error}')
    return True


def load_checkpoint(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    config = saved.get('config', {})
    if (config.get('family') != 'Zenkai' or config.get('nativeScale') != 2 or
            config.get('channels') != 32 or config.get('depth') not in (5, 6)):
        raise ValueError('Not a reviewed Zenkai Lite V1 master configuration')
    if type(saved.get('step')) is not int or saved['step'] < 1:
        raise ValueError('Checkpoint has no positive training step')
    state = saved.get('model')
    if not state or any(not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or
                        not torch.isfinite(value).all() for value in state.values()):
        raise ValueError('Master weights must be finite FP32 tensors')
    model = ZenkaiLite(channels=config['channels'], depth=config['depth']).eval()
    model.load_state_dict(state, strict=True)
    return model, {'config': config, 'step': saved['step']}


def parity_inputs():
    generator = torch.Generator().manual_seed(193)
    for height, width in ((1, 1), (17, 19), (32, 40), (40, 48)):
        yield f'random-{height}x{width}', torch.rand(1, 3, height, width, generator=generator)
    yield 'constant', torch.full((1, 3, 24, 32), .5)
    grid = (torch.arange(24)[:, None] + torch.arange(32)[None, :]) % 2
    yield 'checkerboard', grid.float()[None, None].repeat(1, 3, 1, 1)


def export_candidate(checkpoint, output, device='cpu'):
    checkpoint = Path(checkpoint).resolve()
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT / 'models/zenkai'):
        raise ValueError('Experimental export must remain inside models/zenkai')
    if output.exists():
        raise FileExistsError('Refusing to overwrite an experimental export')
    if device not in ('cpu', 'cuda'):
        raise ValueError('Unsupported validation device')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested GPU is unavailable')
    original_hash = sha256(checkpoint)
    source, saved = load_checkpoint(checkpoint)
    master_state = {key: value.clone() for key, value in source.state_dict().items()}
    deploy = source.deploy_copy().eval()
    if any(isinstance(module, GRepConv) for module in deploy.modules()):
        raise ValueError('Incomplete reparameterization')
    half = copy.deepcopy(deploy).half().eval()
    report = {
        'schemaVersion': 1, 'family': 'Zenkai', 'version': 'Lite V1',
        'id': 'zenkai-lite-v1-experimental', 'status': 'experimental',
        'checkpoint': str(checkpoint), 'checkpointSha256': original_hash,
        'step': saved['step'], 'training': saved['config'], 'architecture': source.config,
        'nativeScale': 2, 'masterPrecision': 'fp32', 'deployPrecision': 'fp16',
        'trainingParameters': sum(value.numel() for value in source.parameters()),
        'deployParameters': sum(value.numel() for value in deploy.parameters()),
        'validationDevice': device, 'tolerances': TOLERANCES, 'cases': [],
        'releaseApproved': False, 'promoted': False,
        'realVideoValidation': 'REAL VIDEO VALIDATION NOT EXECUTED',
        'state': 'validating', 'artifacts': {},
    }
    output.mkdir(parents=True)
    try:
        inputs = list(parity_inputs())
        source.to(device); deploy.to(device); half.to(device)
        with torch.inference_mode():
            for name, value in inputs:
                x = value.to(device)
                expected = source(x)
                fp32 = deploy(x)
                fp16 = half(x.half())
                fusion_score = parity_metrics(expected, fp32)
                half_score = parity_metrics(expected, fp16)
                require_parity(fusion_score, **TOLERANCES['fp32'])
                require_parity(half_score, **TOLERANCES['fp16'])
                report['cases'].append({
                    'name': name, 'shape': list(value.shape),
                    'inputSha256': hashlib.sha256(value.numpy().tobytes()).hexdigest(),
                    'fusionFp32': fusion_score, 'deployFp16': half_score,
                })
        source.cpu(); deploy.cpu(); half.cpu()
        if not all(torch.equal(value, source.state_dict()[key]) for key, value in master_state.items()):
            raise ValueError('Export changed FP32 master values')
        torch.save({'model': deploy.state_dict(), 'config': deploy.config,
                    'sourceSha256': original_hash}, output / 'deploy-fp32.pt')
        np.savez(output / 'deploy-fp16.npz', **{
            key: value.detach().numpy() for key, value in half.state_dict().items()})
        import onnx
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        for precision, model in (('fp32', deploy), ('fp16', half)):
            path = output / f'dynamic-{precision}.onnx'
            dtype = torch.float16 if precision == 'fp16' else torch.float32
            torch.onnx.export(model, torch.zeros(1, 3, 24, 32, dtype=dtype), str(path),
                              input_names=['input'], output_names=['output'], opset_version=17,
                              dynamic_axes={'input': {2: 'height', 3: 'width'},
                                            'output': {2: 'out_height', 3: 'out_width'}}, dynamo=False)
            onnx.checker.check_model(str(path))
            session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
            with torch.inference_mode():
                for row, (_, value) in zip(report['cases'], inputs):
                    actual = session.run(None, {'input': value.to(dtype).numpy()})[0]
                    scores = parity_metrics(source(value), actual)
                    require_parity(scores, **TOLERANCES[precision])
                    row[f'onnx{precision.upper()}'] = scores
            del session
        if sha256(checkpoint) != original_hash:
            raise ValueError('Checkpoint changed during export')
        for path in output.iterdir():
            if path.is_file():
                report['artifacts'][path.name] = {'sha256': sha256(path), 'bytes': path.stat().st_size}
        report.update(state='complete', parityApproved=True, onnxProvider='CPUExecutionProvider',
                      onnxGpuValidation='NOT EXECUTED', performance='NOT MEASURED',
                      scope='Synthetic numeric export parity; no quality or temporal promotion')
    except Exception as error:
        report.update(state='failed', parityApproved=False, error=str(error))
        raise
    finally:
        (output / 'export-report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def validate_runtime(folder, output, device='cuda'):
    """Measure FP32 fusion/FP16 parity on the deployment GPU without ONNX imports."""
    configure_cache()
    from model_registry import ModelRegistry
    from zenkai_artifacts import register_zenkai_export
    folder=Path(folder).resolve();output=Path(output).resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        raise ValueError('Runtime report must be new and inside workspace')
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('GPU unavailable')
    exported=json.loads((folder/'export-report.json').read_text(encoding='utf-8'))
    if sha256(exported['checkpoint'])!=exported['checkpointSha256']:
        raise ValueError('Master checkpoint changed')
    registry=ModelRegistry();register_zenkai_export(registry,folder,'zenkai-runtime-test')
    source,_=load_checkpoint(exported['checkpoint']);source.to(device)
    runtime=registry.load('zenkai-runtime-test','torch',device=device,precision='fp32')
    half=copy.deepcopy(runtime.backend.model).half()
    report={'state':'validating','device':device,'hip':torch.version.hip,'torch':str(torch.__version__),
            'gpu':torch.cuda.get_device_name() if device=='cuda' else None,
            'checkpointSha256':exported['checkpointSha256'],
            'exportReportSha256':sha256(folder/'export-report.json'),
            'artifact':runtime.artifact,'tolerances':TOLERANCES,'cases':[],
            'releaseApproved':False,'onnxGpuValidation':'NOT EXECUTED'}
    try:
        with torch.inference_mode():
            for name,value in parity_inputs():
                x=value.to(device);reference=source(x)
                fp32=parity_metrics(reference,runtime.infer(x))
                fp16=parity_metrics(reference,half(x.half()))
                require_parity(fp32,**TOLERANCES['fp32']);require_parity(fp16,**TOLERANCES['fp16'])
                report['cases'].append({'name':name,'shape':list(value.shape),'fusionFp32':fp32,'deployFp16':fp16})
        report.update(state='complete',parityApproved=True)
    except Exception as error:
        report.update(state='failed',parityApproved=False,error=str(error));raise
    finally:
        registry.unload('zenkai-runtime-test')
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint', type=Path)
    source.add_argument('--validate-runtime', type=Path,help='Export directory to verify on the selected device; output is a new JSON file')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = (validate_runtime(args.validate_runtime,args.output,args.device) if args.validate_runtime
              else export_candidate(args.checkpoint,args.output,args.device))
    print(json.dumps({'state': report['state'], 'output': str(args.output), 'promoted': False}))


if __name__ == '__main__':
    main()
