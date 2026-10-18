"""Export reviewed RDNA1 source changes and existing CPU evidence; run no tests/GPU.

Requires byte-preserved originals and the reports created by the CPU-only work.
The output directory must be new. ZIPs contain only the explicitly selected
recipes, verification tools and MIT license; never weights or installed DLLs.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write_zip(path, files):
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, source in sorted(files.items()):
            entry = zipfile.ZipInfo(name, date_time=(2026, 10, 11, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o644 << 16
            archive.writestr(entry, source.read_bytes())


def export(source, diagnostics, output):
    source, diagnostics, output = map(lambda p: Path(p).resolve(), (source, diagnostics, output))
    if output.exists():
        raise FileExistsError('Choose a new output directory: ' + str(output))
    manifest = json.loads((diagnostics / 'backup-manifest.json').read_text(encoding='utf-8'))
    if Path(manifest['sourceRoot']).resolve() != source:
        raise ValueError('Backup manifest does not describe the selected source')
    original, updated, rows, diff = {}, {}, [], []
    for row in manifest['files']:
        name = row['path'].replace('\\', '/')
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid source path in manifest')
        before, after = diagnostics / 'original' / relative, source / relative
        if digest(before) != row['sha256']:
            raise ValueError('Original backup hash changed: ' + name)
        original[name] = before
        if before.read_bytes() != after.read_bytes():
            updated[name] = after
    new_names = [v + '/build-options' for v in ('ROCm-5.4', 'ROCm-6.3', 'ROCm-Edge')]
    new_names += ['Verification/' + n for n in ('inventory_runtime.py', 'prepare_animescaler_cases.py', 'README-CPU.md')]
    new_names += ['Verification/tests/' + n for n in ('test_build_recipes.py', 'test_docker_pipeline.py', 'test_inventory.py', 'test_miopen_verifier.py', 'test_patch_arguments.py', 'test_torch_onnx_stages.py', 'test_torch_verifier.py', 'test_verify_entrypoint.py', 'test_workload_preparation.py')]
    new_names += ['PLANO-RDNA1-ANIMESCALER.md']
    for name in new_names:
        if name in original:
            raise ValueError('Expected new source has an original backup: ' + name)
        updated[name] = source / name
    for name, after in sorted(updated.items()):
        before = original.get(name)
        old_lines = before.read_text(encoding='utf-8').splitlines(keepends=True) if before else []
        new_lines = after.read_text(encoding='utf-8').splitlines(keepends=True)
        diff.append('diff --git a/' + name + ' b/' + name + '\n')
        for line in difflib.unified_diff(old_lines, new_lines, fromfile='a/' + name if before else '/dev/null', tofile='b/' + name):
            diff.append(line if line.endswith('\n') else line + '\n\\ No newline at end of file\n')
        rows.append({'path': name, 'change': 'modified' if before else 'added',
                     'originalSha256': digest(before) if before else None,
                     'updatedSha256': digest(after), 'updatedBytes': after.stat().st_size})
    original['LICENSE'] = source / 'LICENSE'
    updated['LICENSE'] = source / 'LICENSE'
    reports = {'runtime-inventory.json': 'runtime-inventory-final.json', 'cpu-parity.json': 'cpu-parity-final.json',
               'animescaler-cases.json': 'animescaler-cases.json', 'cpu-suite.txt': 'final-cpu-suite-32-verified.txt',
               'syntax.json': 'final-syntax-128.json', 'backup-manifest.json': 'backup-manifest.json'}
    suite = (diagnostics / reports['cpu-suite.txt']).read_text(encoding='utf-8')
    if 'Ran 32 tests' not in suite or not suite.rstrip().endswith('OK'):
        raise ValueError('Expected complete passing CPU suite is missing')
    if json.loads((diagnostics / reports['syntax.json']).read_text())['errors']:
        raise ValueError('Syntax checks failed')
    output.mkdir(parents=True)
    for name, report in reports.items():
        shutil.copyfile(diagnostics / report, output / name)
    (output / 'changes.patch').write_text(''.join(diff), encoding='utf-8', newline='\n')
    write_zip(output / 'original-source.zip', original)
    write_zip(output / 'updated-source.zip', updated)
    artifacts = [{'path': p.name, 'sha256': digest(p), 'bytes': p.stat().st_size} for p in sorted(output.iterdir()) if p.is_file()]
    index = {'scope': 'CPU-only stabilization; no GPU or full ROCm compilation acceptance',
             'sourceRoot': str(source), 'originalBackupsVerified': len(manifest['files']),
             'modifiedFiles': sum(row['change'] == 'modified' for row in rows),
             'addedFiles': sum(row['change'] == 'added' for row in rows),
             'patchScope': 'Normalized text review diff; ZIPs preserve exact source bytes',
             'sourceFiles': rows, 'artifacts': artifacts, 'gpuInitialized': False}
    (output / 'index.json').write_text(json.dumps(index, indent=2), encoding='utf-8', newline='\n')
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--diagnostics', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = export(args.source, args.diagnostics, args.output)
    print(json.dumps({key: report[key] for key in ('originalBackupsVerified', 'modifiedFiles', 'addedFiles', 'gpuInitialized')}))


if __name__ == '__main__':
    main()
