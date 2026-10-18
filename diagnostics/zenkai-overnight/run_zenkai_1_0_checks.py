"""Serial clean-process runtime/benchmark checks for the consolidated model."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'native'))
from hip_operation import run_operation, save_json
from zenkai_1_0 import MODEL_ID, PACKAGE_RELATIVE, digest, validate_package

GPU = 'F:/comfyui-rocm-071926/python_env/python.exe'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'diagnostics/zenkai-1.0')
    args = parser.parse_args()
    base = args.output.resolve()
    if not base.is_relative_to(ROOT / 'diagnostics'):
        raise ValueError('New diagnostic operation inside workspace required')
    if (base / 'pipeline.json').exists():
        raise FileExistsError('Preserve previous Zenkai 1.0 checks')
    validate_package(ROOT / PACKAGE_RELATIVE)
    baseline = read(ROOT / 'diagnostics/zenkai-domain10-1k/final-verification.json')
    for path, pin in baseline['selectedArtifactsSha256'].items():
        if digest(ROOT / path) != pin:
            raise ValueError('Previous completed round changed: ' + path)
    base.mkdir(exist_ok=True)
    pipeline = {'state': 'RUNNING', 'operations': {}, 'releaseApproved': False, 'newTrainingExecuted': False}
    save_json(base / 'pipeline.json', pipeline)
    def runtime_result(out):
        report = read(out / 'runtime.json')
        if report['state'] != 'PASS' or report['model'] != MODEL_ID or len(report['cases']) != 7 or report['renderer']['state'] != 'PASS':
            raise ValueError('Incomplete runtime parity')
        return {'path': str(out / 'runtime.json'), 'sha256': digest(out / 'runtime.json')}
    runtime = run_operation(base / 'runtime', lambda attempt, out: [GPU, ROOT / 'diagnostics/zenkai-overnight/validate_zenkai_1_0.py',
                            '--device', 'cuda', '--output', out / 'runtime.json'], runtime_result, timeout=600)
    pipeline['operations']['runtime'] = runtime
    save_json(base / 'pipeline.json', pipeline)
    def benchmark_result(out):
        path = out / 'benchmark/benchmark_results.json'
        report = read(path)
        if not report['complete'] or report['provider'] != 'ROCm' or len(report['cases']) != 4:
            raise ValueError('Incomplete registered model benchmark')
        for row in report['cases']:
            if row['warmup'] != 50 or row['iterations'] != 500:
                raise ValueError('Changed benchmark budget')
            for component in (row['components']['totalMeasured'], row['components']['modelInference']):
                if any(key not in component for key in ('meanMs','medianMs','p50Ms','p95Ms','p99Ms','minMs','maxMs')):
                    raise ValueError('Partial timing fields')
        return {'path': str(path), 'sha256': digest(path)}
    if runtime['state'] == 'PASS':
        benchmark = run_operation(base / 'benchmark', lambda attempt, out: [GPU, ROOT / 'native/benchmark_sr_models.py',
            '--model', MODEL_ID, '--model', 'adore-lite-v3', '--warmup', '50', '--iterations', '500',
            '--output', out / 'benchmark'], benchmark_result, timeout=1200)
        pipeline['operations']['benchmark'] = benchmark
    else:
        pipeline['operations']['benchmark'] = {'state': 'NOT_MEASURED', 'reason': 'Runtime integration not verified', 'attempts': []}
    pipeline['state'] = 'COMPLETE' if all(row['state'] == 'PASS' for row in pipeline['operations'].values()) else 'INCONCLUSIVE'
    save_json(base / 'pipeline.json', pipeline)
    print(json.dumps({'state': pipeline['state'], 'operations': {key: value['state'] for key, value in pipeline['operations'].items()}}))


if __name__ == '__main__':
    main()
