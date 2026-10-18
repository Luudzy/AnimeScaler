"""CPU-only backend environment check. Reads package metadata; imports no GPU libraries."""
import argparse
from importlib import metadata
import os
from pathlib import Path
import re
import sys


REQUIRED = {
    'rocm': ('torch', 'spandrel', 'numpy', 'aiohttp'),
    'tensorrt': ('torch', 'tensorrt', 'spandrel', 'numpy', 'aiohttp'),
    'opencl': ('numpy', 'aiohttp', 'idna'),
    'fsr3': ('aiohttp', 'numpy'),
}


def backend_issues(backend, python_version, packages, sdk_present=False, opencl_worker=False):
    if backend not in REQUIRED:
        return [f'Unsupported backend: {backend}']
    errors = []
    for name in REQUIRED[backend]:
        if not packages.get(name):
            errors.append(f'Missing package: {name}')
    if backend == 'rocm':
        torch_version = packages.get('torch') or ''
        if not re.search(r'rocm|hip', torch_version, re.IGNORECASE):
            errors.append('PyTorch ROCm/HIP build required')
    elif backend == 'tensorrt':
        torch_version = packages.get('torch') or ''
        if not torch_version or re.search(r'rocm|hip', torch_version, re.IGNORECASE):
            errors.append('PyTorch CUDA build required')
        tensorrt_version = packages.get('tensorrt') or ''
        if not tensorrt_version.startswith('10.'):
            errors.append('TensorRT 10.x required')
    elif backend == 'fsr3':
        if tuple(python_version[:2]) != (3, 10):
            errors.append('Python 3.10 required for FSR3')
        if not sdk_present:
            errors.append('FidelityFX SDK include tree required')
    elif backend == 'opencl' and not opencl_worker:
        errors.append('Anime4KCPP OpenCL worker and runtime files required')
    return errors


def anime4kcpp_worker_available(root):
    root=Path(root)
    worker=root/'third_party/Anime4KCPP-master/worker/build/bin/Release/anime4kcpp_worker.exe'
    runtime=root/'third_party/Anime4KCPP-master/build-opencl-md/bin/Release'
    return worker.is_file() and runtime.is_dir()


def sdk_include_tree(path):
    if not path:
        return False
    root = Path(path)
    return (root / 'Kits' / 'FidelityFX' / 'api' / 'include').is_dir() and (root / 'Kits' / 'FidelityFX' / 'upscalers' / 'include').is_dir()


def resolve_fsr_sdk_path(explicit=None, environ=None):
    environment = os.environ if environ is None else environ
    value = explicit or environment.get('ANIMESCALER_FSR_SDK')
    if not value:
        raise ValueError('Set --sdk or configure ANIMESCALER_FSR_SDK with the official FidelityFX SDK path')
    return Path(value).expanduser()


def installed_packages(backend):
    versions = {}
    for name in REQUIRED[backend]:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=tuple(REQUIRED), required=True)
    parser.add_argument('--sdk', default=os.environ.get('ANIMESCALER_FSR_SDK'))
    args = parser.parse_args(argv)
    packages = installed_packages(args.backend)
    root=Path(__file__).resolve().parents[1]
    errors = backend_issues(args.backend, sys.version_info, packages,
                            sdk_include_tree(args.sdk) if args.backend == 'fsr3' else False,
                            anime4kcpp_worker_available(root) if args.backend == 'opencl' else False)
    if errors:
        for error in errors:
            print(f'ERROR: {error}', file=sys.stderr)
        return 1
    print(f'Preflight OK for {args.backend}; package metadata only, no GPU or DLL loaded.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
