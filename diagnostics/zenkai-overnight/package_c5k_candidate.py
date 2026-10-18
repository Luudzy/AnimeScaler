"""Immutable local research package; numerical success does not imply release quality."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[2]
RUN = 'zenkai-lite-v1-c-codec50-5k'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    report_path = ROOT/'diagnostics/zenkai-c5k/comparison.json'
    report = read(report_path)
    decision_path = ROOT/'docs/zenkai-c5k-decision.json'
    decision = read(decision_path)
    if report['state'] != 'COMPLETE' or decision['comparisonSha256'] != sha(report_path):
        raise ValueError('Completed measurements and matching decision required')
    for op in ('export', 'runtime'):
        if report['operations'][op]['state'] != 'PASS':
            raise ValueError('Numerical export and runtime parity required')
    export = report['export']
    runtime = report['runtime']
    export_folder = Path(report['operations']['export']['result']['folder'])
    if not export['parityApproved'] or not runtime['parityApproved']:
        raise ValueError('Parity failed')
    if runtime['exportReportSha256'] != sha(export_folder/'export-report.json'):
        raise ValueError('Runtime checked a different export')
    checkpoint = Path(export['checkpoint'])
    if sha(checkpoint) != export['checkpointSha256'] or sha(checkpoint) != report['checkpointsSha256']['step-5000-master-fp32.pt']:
        raise ValueError('Wrong source master')
    destination = ROOT/'models/zenkai/review-packages/c5k'
    if destination.exists():
        raise FileExistsError('Preserve existing package')
    package = destination/RUN
    package.mkdir(parents=True)
    for name, record in export['artifacts'].items():
        if Path(name).name != name or sha(export_folder/name) != record['sha256']:
            raise ValueError('Mutated export artifact')
        shutil.copyfile(export_folder/name, package/name)
    shutil.copyfile(export_folder/'export-report.json', package/'export-report.json')
    shutil.copyfile(report['operations']['runtime']['result']['path'], package/'rocm-parity.json')
    shutil.copyfile(checkpoint, package/'training-master-fp32.pt')
    shutil.copyfile(report_path, package/'measurements.json')
    shutil.copyfile(decision_path, package/'decision.json')
    (package/'config.json').write_text(json.dumps(report['config'], indent=2), encoding='utf-8')
    shutil.copyfile(ROOT/'native/vendor/espan/LICENSE', package/'ESPAN-MIT-NOTICE.txt')
    source = package/'archived-training-source'
    source.mkdir()
    run_folder = ROOT/'models/zenkai/experiments'/RUN
    for path in (run_folder/'source').iterdir():
        if path.is_file():
            shutil.copyfile(path, source/path.name)
    quality_gate = decision['advancementAgainstHistoricalV3']
    card = [f'# {RUN}: experimental review artifact', '',
        f'Technical export: PASS. Quality advancement against HistoricalV3: {quality_gate}. Not release-approved.', '',
        'Native x2, RGB NCHW, batch1, values0-1. ONNX input `input`, output `output`, dynamic spatial dimensions, opset17. Bilinear x2 plus learned residual is already in the graph. Clamp output to0-1; do not add another bilinear base. The720-to1080 path uses x2-to1440 followed by the existing bilinear post-resize; no x1.5 head.', '',
        'Exact identity-skip family:32 channels/depth5, Frequency loss, Codec50, seed929, LR128/HR256, batch4x2, original5000-step warmup/cosine. Faithful actual3000-to5000 continuation. The deploy FP32 state requires the explicit IdentitySkip factory with its block/tail bypasses. Archived training sources are provenance snapshots, not a standalone installed Python package. ONNX graphs are standalone.', '',
        'FP32 fusion, FP16, ONNX CPU parity and reloaded PyTorch ROCm parity passed the fixed synthetic cases. ONNX ROCm/TensorRT/WebGPU execution and end-to-end player FPS: NOT MEASURED. Backend support cannot be inferred from successful export.', '',
        'Measurements cover160 fixed samples in clean/heavy/H264/H265/AV1, the same144 frames from three excerpts of one compressed WebRip and50 warmups/500 distinct benchmark inputs. Individual regressions and all timing components are preserved in measurements.json. No blind or moving preference study was executed. Final-hidden remains unopened.', '',
        'This is a local private-review candidate, not an extension default. HistoricalV3 and all original runs remain preserved. No dataset/reference images are included. ESPAN inspiration notice is included; no new public redistribution license for weights or training material is assigned.', '',
        'Package member SHA256/size records are in package-manifest.json; ZIP CRC and all recorded member hashes were verified after writing.']
    (package/'README.md').write_text('\n'.join(card)+'\n', encoding='utf-8')
    members = {str(p.relative_to(package)).replace('\\', '/'):
        {'sha256': sha(p), 'bytes': p.stat().st_size}
        for p in sorted(package.rglob('*')) if p.is_file()}
    manifest = {'run': RUN, 'step': 5000, 'family': export['family'], 'status': 'experimental',
        'technicalExport': 'PASS', 'qualityAdvancementAgainstHistoricalV3': quality_gate,
        'releaseApproved': False, 'promoted': False, 'members': members}
    (package/'package-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    archive = destination/(RUN+'.zip')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(package.rglob('*')):
            if p.is_file():
                z.write(p, str(p.relative_to(package)))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError('ZIP CRC failed')
        if set(z.namelist()) != set(members) | {'package-manifest.json'}:
            raise ValueError('Unexpected ZIP membership')
        if json.loads(z.read('package-manifest.json')) != manifest:
            raise ValueError('ZIP manifest changed')
        for name, record in members.items():
            if hashlib.sha256(z.read(name)).hexdigest() != record['sha256']:
                raise ValueError('ZIP member SHA failed')
    result = {'state': 'PACKAGED EXPERIMENTAL', 'archive': str(archive), 'sha256': sha(archive),
        'bytes': archive.stat().st_size, 'members': len(members)+1,
        'qualityGate': quality_gate, 'releaseApproved': False, 'promoted': False}
    (destination/'packages.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
