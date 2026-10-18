"""Package the measured C5k consolidation as experimental Zenkai 1.0; no retraining."""
import argparse
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from zenkai_1_0 import (ARTIFACT_PINS, MODEL_ID, PACKAGE_RELATIVE, RECIPE, SOURCE_MASTER_SHA,
                        SOURCE_RUN, digest, validate_package)

ROOT = Path(__file__).resolve().parents[1]
EXPORT_RELATIVE = Path('models/zenkai/exports/zenkai-lite-v1-c-codec50-5k-export-attempt-1')


def build_package(root, source_root=None):
    root = Path(root).resolve()
    source_root = Path(source_root).resolve() if source_root is not None else root
    output = root / PACKAGE_RELATIVE
    if output.exists():
        raise FileExistsError('Preserve existing Zenkai 1.0 package')
    source_master = source_root / 'models/zenkai/experiments' / SOURCE_RUN / 'step-5000-master-fp32.pt'
    if digest(source_master) != SOURCE_MASTER_SHA:
        raise ValueError('Measured C5k master checkpoint changed')
    sources = {name: source_root / EXPORT_RELATIVE / name for name in ARTIFACT_PINS}
    sources['research-comparison.json'] = source_root / 'diagnostics/zenkai-c5k/comparison.json'
    sources['research-decision.json'] = source_root / 'docs/zenkai-c5k-decision.json'
    for name, path in sources.items():
        if digest(path) != ARTIFACT_PINS[name]:
            raise ValueError('Frozen Zenkai 1.0 source evidence changed: ' + name)
    comparison = json.loads(sources['research-comparison.json'].read_text(encoding='utf-8'))
    decision = json.loads(sources['research-decision.json'].read_text(encoding='utf-8'))
    if comparison['state'] != 'COMPLETE' or decision['advancementAgainstHistoricalV3'] != 'FAIL':
        raise ValueError('Measured consolidation evidence is incomplete or misrepresented')
    output.mkdir(parents=True)
    for name, path in sources.items():
        shutil.copyfile(path, output / name)
    manifest = {'schemaVersion': 1, 'id': MODEL_ID, 'name': 'Zenkai 1.0', 'version': '1.0.0',
                'state': 'complete', 'status': 'experimental', 'family': 'ZenkaiIdentitySkipProbe',
                'nativeScale': 2, 'sourceRun': SOURCE_RUN, 'sourceCheckpointSha256': SOURCE_MASTER_SHA,
                'trainingSteps': 5000, 'newTrainingExecuted': False, 'recipe': RECIPE,
                'files': ARTIFACT_PINS, 'qualityAdvancement': 'FAIL', 'releaseApproved': False,
                'promoted': False, 'provenance': 'Exact C5k weights and inherited research measurements; not a new train',
                'backends': {'rocm': 'PyTorch FP16 exact fused identity-skip graph',
                             'tensorrt': 'FP16 build through existing NVIDIA backend; NOT MEASURED on RX5500XT',
                             'webgpu': 'NOT INTEGRATED; native selection never falls back silently'},
                'exclusions': ['crop96', 'group sampler', 'constant scheduler', 'inconclusive V2/domain10 mix',
                               'GAN/perceptual training loss', 'x1.5 head', 'custom kernels', 'automatic Stage2'],
                'knownLimitations': ['No coherent quality win over Historical V3',
                                    'Measured C5k 720-to-1080 model+resize exceeds33.3ms',
                                    'One seed/episode, moving player and temporal H265/AV1 not certified']}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    validate_package(output)
    if digest(source_master) != SOURCE_MASTER_SHA:
        raise ValueError('Packaging altered source master')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    path = build_package(ROOT)
    print(json.dumps({'state': 'PASS', 'package': str(path), 'newTrainingExecuted': False, 'releaseApproved': False}))


if __name__ == '__main__':
    main()
