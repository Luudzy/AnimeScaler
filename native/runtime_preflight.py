"""Inspect Windows ROCm files before Torch import. Never load DLLs or probe a GPU."""
import argparse
import ast
from datetime import datetime, timezone
from email.parser import Parser
import fnmatch
import hashlib
from importlib.machinery import PathFinder
import json
import os
from pathlib import Path
import re
import struct
import sys
import uuid


def read_source(path):
    if path.stat().st_size > 1024 * 1024:
        raise ValueError('Source exceeds static inspection limit')
    return ast.parse(path.read_text(encoding='utf-8'))


def assignments(tree):
    values = {}
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        for target in targets:
            if isinstance(target, ast.Name) and node.value is not None:
                try:
                    values[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    values.pop(target.id, None)
    return values


def package_path(name, paths):
    spec = PathFinder.find_spec(name, paths)
    if spec is None or spec.origin is None:
        return None
    if not spec.submodule_search_locations:
        raise ValueError(f'{name} resolves to a module, not a package')
    return Path(next(iter(spec.submodule_search_locations))).resolve()


def pe_machine(path):
    with path.open('rb') as stream:
        if stream.read(2) != b'MZ':
            raise ValueError('Missing PE DOS signature')
        stream.seek(0x3c)
        offset_bytes = stream.read(4)
        if len(offset_bytes) != 4:
            raise ValueError('Truncated PE header')
        offset = struct.unpack('<I', offset_bytes)[0]
        stream.seek(offset)
        header = stream.read(6)
        if len(header) != 6 or header[:4] != b'PE\0\0':
            raise ValueError('Missing PE signature')
        return struct.unpack('<H', header[4:])[0]


def target_paths(python):
    if python.resolve() == Path(sys.executable).resolve():
        return list(sys.path), 'current-process-path', False
    pth_files = sorted(python.parent.glob('python*._pth'))
    if pth_files:
        if len(pth_files) != 1:
            raise ValueError('Multiple portable ._pth files; select process search paths explicitly')
        entries, import_site = [], False
        for line in pth_files[0].read_text(encoding='utf-8-sig').splitlines():
            line = line.strip()
            if line == 'import site':
                import_site = True
            if line and not line.startswith('#') and line != 'import site':
                if line.startswith('import '):
                    raise ValueError('Unsupported executable entry in portable ._pth')
                entries.append(str((python.parent / line.replace('\\', os.sep)).resolve()))
        if import_site:
            standard_site = str(python.parent / 'Lib/site-packages')
            if standard_site not in entries:
                entries.append(standard_site)
        return entries, 'portable-pth-static', import_site
    return [str(python.parent / 'Lib/site-packages')], 'conventional-layout-static', True


def sdk_entries(tree):
    packages, libraries = {}, {}
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            call = node.value
            if call.func.id not in ('PackageEntry', 'LibraryEntry'):
                continue
            args = [ast.literal_eval(arg) for arg in call.args]
            kw = {arg.arg: ast.literal_eval(arg.value) for arg in call.keywords}
            if call.func.id == 'PackageEntry':
                packages[args[0]] = args[1]
            else:
                libraries[args[0]] = {'package': args[1], 'pattern': args[3], 'optional': kw.get('optional', args[5] if len(args) > 5 else False)}
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr == 'dist_package_template' and isinstance(target.value, ast.Subscript) and isinstance(target.value.value, ast.Name) and target.value.value.id == 'ALL_PACKAGES':
                    packages[ast.literal_eval(target.value.slice)] = ast.literal_eval(node.value)
    return packages, libraries


def inspect_runtime(python=None, *, search_paths=None, environ=None):
    python = Path(python or sys.executable).resolve()
    environ = os.environ if environ is None else environ
    report = {'date': datetime.now(timezone.utc).isoformat(), 'python': str(python),
              'scope': 'Static source/metadata, selected filesystem candidates and PE signatures/machines only; not a loader trace, ABI, driver or GPU validation',
              'gpuProbePerformed': False, 'runtimeImported': False, 'errors': [], 'warnings': [],
              'torch': {}, 'libraries': [], 'packages': [], 'expandedPayloadVersions': {}, 'expandedPayloadAliases': {}}
    def issue(kind, code, message):
        report[kind].append({'code': code, 'message': message})
    incomplete = False
    selected_roots = {}
    suffix = None
    try:
        machine = pe_machine(python)
    except (OSError, ValueError) as error:
        issue('errors', 'python-pe-invalid', f'Executável Python sem assinatura PE legível: {error}')
        report['status'] = 'failed'
        return report
    try:
        report['pythonMachine'] = machine
        if search_paths is None:
            search_paths, report['searchPathScope'], site_unexecuted = target_paths(python)
            if site_unexecuted:
                issue('warnings', 'site-hooks-unverified', 'Caminhos convencionais de site considerados; hooks .pth/sitecustomize não foram executados.')
        else:
            search_paths = list(search_paths)
            report['searchPathScope'] = 'provided-process-path'
        torch_path = package_path('torch', search_paths)
        if torch_path is None:
            issue('errors', 'torch-missing', 'Torch não encontrado nos caminhos deste Python.')
        else:
            report['torchPath'] = str(torch_path)
            version = assignments(read_source(torch_path / 'version.py'))
            report['torch'] = {key: version[key] for key in ('__version__', 'hip', 'cuda', 'rocm', 'git_version') if key in version}
            if 'hip' in version and not version['hip']:
                issue('errors', 'torch-not-rocm', 'Este Python contém Torch sem HIP/ROCm; confira ANIMESCALER_PYTHON.')
            elif 'hip' not in version:
                incomplete = True
                issue('warnings', 'torch-version-unverified', 'Versão HIP não pôde ser lida estaticamente.')
            bootstrap = torch_path / '_rocm_init.py'
            calls = []
            if bootstrap.is_file():
                calls = [node for node in ast.walk(read_source(bootstrap)) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'initialize_process']
            if len(calls) != 1:
                incomplete = True
                issue('warnings', 'bootstrap-unverified', 'Bootstrap SDK não reconhecido; runtime legado não foi validado.')
            else:
                config, unknown_arguments = {}, []
                for kw in calls[0].keywords:
                    if kw.arg in ('preload_shortnames', 'check_version', 'env_override'):
                        try:
                            config[kw.arg] = ast.literal_eval(kw.value)
                        except (ValueError, TypeError):
                            unknown_arguments.append(kw.arg)
                if unknown_arguments:
                    incomplete = True
                    issue('warnings', 'bootstrap-arguments-unverified', 'Argumentos dinâmicos do bootstrap não foram executados; argumentos conhecidos continuam sendo verificados.')
                report['torchBootstrap'] = {'path': str(bootstrap), 'sha256': hashlib.sha256(bootstrap.read_bytes()).hexdigest(), **config, 'unknownArguments': unknown_arguments}
                sdk_path = package_path('rocm_sdk', search_paths)
                if sdk_path is None:
                    issue('errors', 'sdk-missing', 'Torch requer rocm_sdk, mas esse pacote não foi encontrado.')
                else:
                    report['sdkPath'] = str(sdk_path)
                    tree = read_source(sdk_path / '_dist_info.py')
                    sdk_values = assignments(tree)
                    report['sdkVersion'] = sdk_values.get('__version__')
                    expected = config.get('check_version')
                    if isinstance(expected, str) and not fnmatch.fnmatchcase(report['sdkVersion'] or '', expected):
                        issue('warnings', 'sdk-version-differs', f'Torch espera SDK {expected}; SDK declara {report["sdkVersion"]}. Compatibilidade ABI não determinada.')
                    schema_recognized = True
                    try:
                        packages, libraries = sdk_entries(tree)
                        if not packages or not libraries:
                            raise ValueError('Missing literal SDK descriptors')
                    except (ValueError, TypeError, KeyError, IndexError):
                        schema_recognized = False
                        incomplete = True
                        issue('warnings', 'sdk-schema-unverified', 'Descritores SDK não puderam ser lidos por AST; bibliotecas não foram adivinhadas nem carregadas.')
                    suffix = sdk_values.get('PY_PACKAGE_SUFFIX_NONCE')
                    if not isinstance(suffix, str):
                        incomplete = True
                        issue('warnings', 'sdk-suffix-unverified', 'Sufixo dos pacotes SDK não reconhecido; caminhos de bibliotecas não foram adivinhados.')
                    preloads = list(config.get('preload_shortnames', []))
                    if config.get('env_override', True) and 'env_override' not in unknown_arguments:
                        preloads += [part.strip() for part in re.split('[,;]', environ.get('ROCM_SDK_PRELOAD_LIBRARIES', '')) if part.strip()]
                    for shortname in dict.fromkeys(preloads) if schema_recognized else []:
                        entry = libraries.get(shortname)
                        if entry is None:
                            issue('errors', 'unknown-preload', 'Bootstrap/ROCM_SDK_PRELOAD_LIBRARIES solicita biblioteca desconhecida pelo SDK.')
                            continue
                        row = {'shortname': shortname}
                        report['libraries'].append(row)
                        if not entry['pattern']:
                            row['status'] = 'not-built-on-windows'
                            continue
                        if not isinstance(suffix, str):
                            row['status'] = 'package-suffix-unverified'
                            continue
                        dist_name = packages[entry['package']]
                        if '{target_family}' in dist_name:
                            family = environ.get('ROCM_SDK_TARGET_FAMILY')
                            if not family:
                                incomplete = True
                                row['status'] = 'target-family-unverified'
                                issue('warnings', 'target-family-unverified', 'SDK requer seleção de família em runtime; nenhuma GPU foi consultada.')
                                continue
                            dist_name = dist_name.format(target_family=family)
                        module = '_' + dist_name.replace('-', '_') + suffix
                        root = package_path(module, search_paths)
                        if root is not None:
                            selected_roots[entry['package']] = root
                        matches = sorted(path for path in (root / 'bin').iterdir() if path.is_file() and fnmatch.fnmatchcase(path.name.lower(), entry['pattern'].lower())) if root and (root / 'bin').is_dir() else []
                        if not matches:
                            row['status'] = 'optional-absent' if entry['optional'] else 'missing'
                            if not entry['optional']:
                                issue('errors', 'required-dll-missing', f'Biblioteca obrigatória {shortname} ausente no pacote {module}.')
                            continue
                        selected = matches[0]
                        row.update(path=str(selected), resolvedPath=str(selected.resolve()), bytes=selected.stat().st_size, status='candidate-present')
                        try:
                            row['machine'] = pe_machine(selected)
                            if row['machine'] != machine:
                                row['status'] = 'machine-mismatch'
                                issue('errors', 'pe-machine-mismatch', f'Arquitetura PE de {shortname} difere da arquitetura deste Python.')
                        except (OSError, ValueError) as error:
                            row['status'] = 'invalid-pe'
                            issue('errors', 'invalid-pe', f'Arquivo de {shortname} não apresenta header PE legível: {error}')
        sites = list(dict.fromkeys(Path(p).resolve() for p in search_paths if Path(p).is_dir()))
        for site in sites:
            for metadata in sorted(site.glob('rocm*.dist-info/METADATA')):
                try:
                    if metadata.stat().st_size > 1024 * 1024:
                        raise ValueError('Metadata exceeds inspection limit')
                    data = Parser().parsestr(metadata.read_text(encoding='utf-8'))
                    name, version = data.get('Name'), data.get('Version')
                    row = {'name': name, 'version': version, 'metadata': str(metadata), 'selected': False}
                    report['packages'].append(row)
                    if name and name.startswith('rocm-sdk-') and isinstance(suffix, str):
                        module = '_' + name.replace('-', '_') + suffix
                        selected_root = selected_roots.get(name[len('rocm-sdk-'):]) or package_path(module, search_paths)
                        candidate_root = site / module
                        row['selected'] = selected_root == candidate_root
                        payload = candidate_root / '.info/version'
                        if row['selected'] and payload.is_file():
                            value = payload.read_text(encoding='utf-8').strip()
                            report['expandedPayloadVersions'].setdefault(name, value)
                            core_root = selected_roots.get('core')
                            core_version = core_root / '.info/version' if core_root else None
                            report['expandedPayloadAliases'].setdefault(name, {'path': str(payload), 'sameFileAsCore': bool(core_version and core_version.is_file() and os.path.samefile(payload, core_version)), 'hardlinkCount': payload.stat().st_nlink})
                            major = re.match(r'^\d+\.\d+\.\d+', version or '')
                            if major and major.group() != value:
                                issue('warnings', 'payload-version-differs', f'{name}: metadata {version}, payload {value}; divergência não comprova incompatibilidade.')
                except (OSError, ValueError):
                    incomplete = True
                    issue('warnings', 'metadata-unverified', 'Metadados opcionais não puderam ser lidos; isso não comprova incompatibilidade do runtime.')
        core_path = selected_roots.get('core')
        manifest = core_path / 'share/therock/therock_manifest.json' if core_path else None
        if manifest and manifest.is_file():
            try:
                if manifest.stat().st_size > 1024 * 1024:
                    raise ValueError('Manifest exceeds inspection limit')
                report['sourceManifest'] = {'scope': 'Local build record of selected core package, not independently authenticated DLL provenance or attribution of standalone devel headers',
                                            'path': str(manifest), 'sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
                                            'data': json.loads(manifest.read_text(encoding='utf-8'))}
            except (OSError, ValueError):
                issue('warnings', 'manifest-unverified', 'Manifesto de origem não pôde ser lido; não foi usado para certificar DLLs.')
    except (OSError, SyntaxError, ValueError, TypeError, KeyError, IndexError) as error:
        incomplete = True
        issue('warnings', 'inspection-unverified', f'Não foi possível concluir o diagnóstico estático ({type(error).__name__}); runtime não certificado.')
    report['status'] = 'failed' if report['errors'] else 'incomplete' if incomplete else 'files-checked'
    return report


def write_report(report, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


def require_rocm_runtime(*, output_dir=None):
    report = inspect_runtime(search_paths=sys.path)
    path = None
    if output_dir is not None:
        name = datetime.now(timezone.utc).strftime('preflight-%Y%m%dT%H%M%SZ-') + uuid.uuid4().hex[:8] + '.json'
        path = Path(output_dir) / name
        try:
            write_report(report, path)
        except OSError:
            path = None
            report['warnings'].append({'code': 'report-write-failed', 'message': 'Não foi possível gravar o relatório ROCm; o resultado do diagnóstico permanece em memória.'})
    if report['errors']:
        message = 'Diagnóstico ROCm rejeitou a inicialização antes do import Torch: ' + '; '.join(e['message'] for e in report['errors'][:3])
        if path:
            message += f' Relatório: {path}'
        raise RuntimeError(message)
    if report['warnings']:
        print('Diagnóstico ROCm: ' + '; '.join(w['message'] for w in report['warnings'][:2]), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', type=Path, default=Path(sys.executable), help='Interpreter to inspect, never executed')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.output and args.output.exists():
        parser.error('Choose a new output path; existing reports are preserved')
    report = inspect_runtime(args.python)
    if args.output:
        write_report(report, args.output)
    print(json.dumps(report, ensure_ascii=False))
    return 1 if report['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
