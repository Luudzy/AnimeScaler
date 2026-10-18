"""Exact, hash-pinned Zenkai 1.0 experimental deployment. No quality promotion."""
import hashlib
import json
from pathlib import Path

MODEL_ID = 'zenkai-1-0'
PACKAGE_RELATIVE = Path('models/zenkai/zenkai-1.0')
SOURCE_RUN = 'zenkai-lite-v1-c-codec50-5k'
SOURCE_MASTER_SHA = 'ffa3d68c324331aeed02267dbef5907b53dc3e8d52ce9618a3d1710f4d297571'
EXPORT_REPORT_SHA = '07fb4f34b973916f1c5c363a32335be31a0f4e60163ba31aed2fdf499459f6ad'
COMPARISON_SHA = 'e74ce0dd469f7f95d84ecc3fddc37d626857450afc18281e27f2643765a886ae'
DECISION_SHA = '9297f5e8278ed285c20ddea5783bbdc5a084ca550237e5f2b345980589374a4d'
ARTIFACT_PINS = {
    'deploy-fp32.pt': '4180beb38aa2eee59a6f4af313e7b235f26cfa5577f9cd74c6d0c94483170b2e',
    'deploy-fp16.npz': 'd1113296b2e05a798310ae05dbacaa0297a18e057a55b0497e3e45d802730ceb',
    'dynamic-fp32.onnx': '30d40daf334d9dc5ebabec2d8ac72857b3cef875a63d99271c14a401cbd77b16',
    'dynamic-fp16.onnx': 'c645e313b9ec45e30a852978a3c44c4d3a3cdef19d50b6ea7a12a01511bc507a',
    'export-report.json': EXPORT_REPORT_SHA,
    'research-comparison.json': COMPARISON_SHA,
    'research-decision.json': DECISION_SHA,
}
RECIPE = {'channels': 32, 'depth': 5, 'nativeScale': 2, 'spabIdentitySkipGain': 1.,
          'loss': 'zenkai_frequency', 'codecFraction': .5, 'patchLR': 128, 'seed': 929,
          'batch': 4, 'accumulation': 2, 'effectiveBatch': 8, 'scheduleTotalSteps': 5000}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_package(folder):
    folder = Path(folder).resolve()
    try:
        manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
        expected = {'schemaVersion': 1, 'id': MODEL_ID, 'name': 'Zenkai 1.0', 'version': '1.0.0',
                    'state': 'complete', 'status': 'experimental', 'family': 'ZenkaiIdentitySkipProbe',
                    'nativeScale': 2, 'sourceRun': SOURCE_RUN, 'sourceCheckpointSha256': SOURCE_MASTER_SHA,
                    'trainingSteps': 5000, 'newTrainingExecuted': False, 'recipe': RECIPE,
                    'qualityAdvancement': 'FAIL', 'releaseApproved': False, 'promoted': False}
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError('Unreviewed Zenkai 1.0 identity/recipe/status')
        if manifest.get('files') != ARTIFACT_PINS:
            raise ValueError('Zenkai 1.0 artifact manifest does not match frozen source pins')
        for name, pin in ARTIFACT_PINS.items():
            path = folder / name
            if path.resolve().parent != folder or digest(path) != pin:
                raise ValueError('Zenkai 1.0 artifact changed: ' + name)
        report = json.loads((folder / 'export-report.json').read_text(encoding='utf-8'))
        if (report.get('state') != 'complete' or report.get('parityApproved') is not True
                or report.get('family') != expected['family'] or report.get('checkpointSha256') != SOURCE_MASTER_SHA):
            raise ValueError('Zenkai 1.0 exact-family export has not passed parity')
        if any(report['training'].get(key) != value for key, value in RECIPE.items()):
            raise ValueError('Zenkai 1.0 export training recipe changed')
        return manifest
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError('Incomplete or unreadable Zenkai 1.0 package') from error


def register_zenkai_1_0(registry, root):
    folder = Path(root).resolve() / PACKAGE_RELATIVE
    if not folder.exists():
        return None
    validate_package(folder)
    from model_registry import ModelMetadata

    def validate(path, pin):
        validate_package(folder)
        if Path(path).resolve().parent != folder or pin != ARTIFACT_PINS.get(Path(path).name):
            raise ValueError('Zenkai 1.0 executed artifact identity changed')

    def factory():
        validate_package(folder)
        from export_zenkai_skip import load_deploy
        report = json.loads((folder / 'export-report.json').read_text(encoding='utf-8'))
        # Keep the identity wrapper. Vanilla Zenkai has different state keys
        # and forward behavior despite sharing channel/depth dimensions.
        return load_deploy(folder, report)

    metadata = ModelMetadata(MODEL_ID, 'Zenkai 1.0', 'Zenkai', 'GRep SPAB identity-skip x2',
                             2, 'experimental', provider='AnimeScaler C5k consolidation',
                             code_license='MIT-derived ESPAN components; see THIRD_PARTY_NOTICES.md',
                             weights_path=str(folder / 'deploy-fp32.pt'), capture_graph=False)
    registry.register(metadata, factory, folder / 'dynamic-fp32.onnx', artifact_validator=validate)
    return metadata
