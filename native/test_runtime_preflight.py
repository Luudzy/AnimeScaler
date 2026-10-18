"""Exercise static ROCm checks with poisoned packages and tiny fake PE files."""
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).with_name('runtime_preflight.py')


def fake_pe(machine=0x8664):
    data = bytearray(160)
    data[:2] = b'MZ'
    struct.pack_into('<I', data, 0x3c, 0x80)
    data[0x80:0x84] = b'PE\0\0'
    struct.pack_into('<H', data, 0x84, machine)
    return data


class RuntimePreflightTests(unittest.TestCase):
    def load_checker(self):
        self.assertTrue(SCRIPT.is_file(), 'CPU-only startup preflight is not implemented')
        spec = importlib.util.spec_from_file_location('runtime_preflight_fixture', SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def fixture(self, root):
        executable = root / 'python.exe'; executable.write_bytes(fake_pe())
        site = root / 'Lib/site-packages'
        for name in ['torch', 'rocm_sdk', '_rocm_sdk_core', '_rocm_sdk_libraries', '_rocm_sdk_devel']:
            folder = site/name; folder.mkdir(parents=True)
            (folder/'__init__.py').write_text("raise RuntimeError('Package code must not execute')\n")
        (site/'torch/version.py').write_text("__version__='2.12.0+rocm7.15'\nhip='7.15.26290'\ncuda=None\ngit_version='"+'f'*40+"'\nraise RuntimeError('No version code execution')\n")
        (site/'torch/_rocm_init.py').write_text("def initialize():\n import rocm_sdk\n rocm_sdk.initialize_process(preload_shortnames=['amdhip64','miopen','hipsparselt'],check_version='7.15.0a20260728')\n")
        (site/'rocm_sdk/_dist_info.py').write_text("""raise RuntimeError('No SDK code execution')
PackageEntry('core','rocm-sdk-core',pure_py_package_name='rocm_sdk_core',template_directory='core',required=True)
PackageEntry('libraries','rocm-sdk-libraries-{target_family}',pure_py_package_name='rocm_sdk_libraries',template_directory='libs')
LibraryEntry('amdhip64','core','libamdhip64.so*','amdhip64*.dll')
LibraryEntry('miopen','libraries','libMIOpen.so*','MIOpen*.dll')
LibraryEntry('hipsparselt','libraries','libhipsparselt.so*','')
LibraryEntry('optional','libraries','liboptional.so*','optional*.dll',optional=True)
__version__='7.15.0a20260728'
PY_PACKAGE_SUFFIX_NONCE=''
ALL_PACKAGES['libraries'].dist_package_template='rocm-sdk-libraries'
""")
        for name, filename in [('_rocm_sdk_core','amdhip64_7.dll'),('_rocm_sdk_libraries','MIOpen.dll')]:
            folder=site/name/'bin'; folder.mkdir(); (folder/filename).write_bytes(fake_pe())
        for name, version in [('rocm-sdk-core','7.15.0a20260728'),('rocm-sdk-devel','10.1.0a20260802')]:
            folder=site/(name.replace('-','_')+'-'+version+'.dist-info'); folder.mkdir()
            (folder/'METADATA').write_text(f'Name: {name}\nVersion: {version}\n')
            payload=site/('_'+name.replace('-','_'))/'.info'; payload.mkdir(parents=True,exist_ok=True)
            (payload/'version').write_text('7.15.0\n')
        return executable,site

    def test_valid_files_and_devel_number_difference_do_not_import_or_block(self):
        checker=self.load_checker()
        with tempfile.TemporaryDirectory() as directory:
            executable,site=self.fixture(Path(directory))
            before=set(sys.modules)
            report=checker.inspect_runtime(executable,search_paths=[str(site)],environ={})
            self.assertFalse(report['errors'],report)
            self.assertTrue(any(w['code']=='payload-version-differs' for w in report['warnings']))
            self.assertFalse(report['gpuProbePerformed'])
            self.assertEqual(report['torch']['git_version'],'f'*40)
            self.assertFalse({'rocm_sdk','_rocm_sdk_core','_rocm_sdk_libraries'} & (set(sys.modules)-before))

    def test_required_dll_missing_is_an_error(self):
        checker=self.load_checker()
        with tempfile.TemporaryDirectory() as directory:
            executable,site=self.fixture(Path(directory)); (site/'_rocm_sdk_core/bin/amdhip64_7.dll').unlink()
            report=checker.inspect_runtime(executable,search_paths=[str(site)],environ={})
            self.assertTrue(any(e['code']=='required-dll-missing' for e in report['errors']))

    def test_windows_unsupported_and_optional_preloads_are_not_missing_errors(self):
        checker=self.load_checker()
        with tempfile.TemporaryDirectory() as directory:
            executable,site=self.fixture(Path(directory))
            report=checker.inspect_runtime(executable,search_paths=[str(site)],environ={'ROCM_SDK_PRELOAD_LIBRARIES':'optional;hipsparselt'})
            self.assertFalse(report['errors'],report)
            statuses={row['shortname']:row['status'] for row in report['libraries']}
            self.assertEqual(statuses['hipsparselt'],'not-built-on-windows')
            self.assertEqual(statuses['optional'],'optional-absent')

    def test_x86_dll_in_x64_runtime_is_rejected(self):
        checker=self.load_checker()
        with tempfile.TemporaryDirectory() as directory:
            executable,site=self.fixture(Path(directory)); (site/'_rocm_sdk_libraries/bin/MIOpen.dll').write_bytes(fake_pe(0x14c))
            report=checker.inspect_runtime(executable,search_paths=[str(site)],environ={})
            self.assertTrue(any(e['code']=='pe-machine-mismatch' for e in report['errors']))

    def test_corrupt_required_dll_is_rejected_without_loading_it(self):
        checker=self.load_checker()
        with tempfile.TemporaryDirectory() as directory:
            executable,site=self.fixture(Path(directory)); (site/'_rocm_sdk_libraries/bin/MIOpen.dll').write_bytes(b'not a DLL')
            report=checker.inspect_runtime(executable,search_paths=[str(site)],environ={})
            self.assertTrue(any(e['code']=='invalid-pe' for e in report['errors']))

    def test_cuda_only_torch_is_rejected(self):
        checker=self.load_checker()
        with tempfile.TemporaryDirectory() as directory:
            executable,site=self.fixture(Path(directory)); (site/'torch/version.py').write_text("__version__='2.9.0+cu130'\nhip=None\ncuda='13.0'\n")
            report=checker.inspect_runtime(executable,search_paths=[str(site)],environ={})
            self.assertTrue(any(e['code']=='torch-not-rocm' for e in report['errors']))

    def test_sdk_expected_version_difference_is_reported_without_claiming_bad_abi(self):
        checker=self.load_checker()
        with tempfile.TemporaryDirectory() as directory:
            executable,site=self.fixture(Path(directory))
            path=site/'torch/_rocm_init.py';path.write_text(path.read_text().replace('7.15.0a20260728','7.14.0'))
            report=checker.inspect_runtime(executable,search_paths=[str(site)],environ={})
            self.assertFalse(report['errors'],report)
            self.assertTrue(any(w['code']=='sdk-version-differs' for w in report['warnings']))

    def test_missing_sdk_required_by_torch_is_rejected(self):
        checker=self.load_checker()
        with tempfile.TemporaryDirectory() as directory:
            executable,site=self.fixture(Path(directory));(site/'rocm_sdk/__init__.py').unlink()
            report=checker.inspect_runtime(executable,search_paths=[str(site)],environ={})
            self.assertTrue(any(e['code']=='sdk-missing' for e in report['errors']))

    def test_unknown_preload_override_is_rejected_without_dumping_environment(self):
        checker=self.load_checker()
        with tempfile.TemporaryDirectory() as directory:
            executable,site=self.fixture(Path(directory))
            secret='secret-marker-should-not-be-logged'
            report=checker.inspect_runtime(executable,search_paths=[str(site)],environ={'ROCM_SDK_PRELOAD_LIBRARIES':secret,'SECRET_TOKEN':secret})
            self.assertTrue(any(e['code']=='unknown-preload' for e in report['errors']))
            self.assertNotIn(secret,json.dumps(report))

    def test_portable_pth_is_read_without_executing_python_or_site(self):
        checker=self.load_checker()
        with tempfile.TemporaryDirectory() as directory:
            executable,site=self.fixture(Path(directory))
            (executable.parent/'python312._pth').write_text('python312.zip\nLib/site-packages\nimport site\n')
            report=checker.inspect_runtime(executable,environ={})
            self.assertFalse(report['errors'],report)
            self.assertEqual(report['torchPath'],str(site/'torch'))
            self.assertEqual(report['searchPathScope'],'portable-pth-static')

    def test_first_package_in_process_search_path_is_checked(self):
        checker=self.load_checker()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);executable,site=self.fixture(root)
            shadow=root/'shadow';(shadow/'torch').mkdir(parents=True)
            (shadow/'torch/__init__.py').write_text("raise RuntimeError('Must not execute')")
            (shadow/'torch/version.py').write_text("hip=None\n__version__='wrong-python'\n")
            report=checker.inspect_runtime(executable,search_paths=[str(shadow),str(site)],environ={})
            self.assertEqual(report['torchPath'],str(shadow/'torch'))
            self.assertTrue(any(e['code']=='torch-not-rocm' for e in report['errors']))

    def test_portable_import_site_adds_standard_site_without_executing_hooks(self):
        checker=self.load_checker()
        with tempfile.TemporaryDirectory() as directory:
            executable,site=self.fixture(Path(directory))
            (executable.parent/'python312._pth').write_text('python312.zip\n.\n..\nimport site\n')
            (site/'poison.pth').write_text("import definitely_must_not_execute\n")
            report=checker.inspect_runtime(executable,environ={})
            self.assertFalse(report['errors'],report)
            self.assertEqual(report['torchPath'],str(site/'torch'))
            self.assertTrue(any(w['code']=='site-hooks-unverified' for w in report['warnings']))

    def test_nonliteral_sdk_schema_is_unverified_instead_of_guessed_incompatible(self):
        checker=self.load_checker()
        with tempfile.TemporaryDirectory() as directory:
            executable,site=self.fixture(Path(directory))
            path=site/'rocm_sdk/_dist_info.py'
            path.write_text(path.read_text().replace("'amdhip64*.dll'","make_pattern()"))
            report=checker.inspect_runtime(executable,search_paths=[str(site)],environ={})
            self.assertFalse(report['errors'],report)
            self.assertEqual(report['status'],'incomplete')
            self.assertTrue(any(w['code']=='sdk-schema-unverified' for w in report['warnings']))

    def test_manifest_and_payload_hardlinks_are_recorded_as_local_evidence(self):
        checker=self.load_checker()
        with tempfile.TemporaryDirectory() as directory:
            executable,site=self.fixture(Path(directory))
            core=site/'_rocm_sdk_core'
            manifest=core/'share/therock/therock_manifest.json';manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({'the_rock_commit':'a'*40,'rocm_package_version':'7.15.0a20260728','rocm_version':'7.15.0','submodules':[{'submodule_name':'rocm-libraries','pin_sha':'b'*40}]}))
            devel=site/'_rocm_sdk_devel/.info/version';devel.unlink();os.link(core/'.info/version',devel)
            report=checker.inspect_runtime(executable,search_paths=[str(site)],environ={})
            self.assertFalse(report['errors'],report)
            self.assertIn('sourceManifest',report)
            self.assertEqual(report['sourceManifest']['data']['the_rock_commit'],'a'*40)
            self.assertTrue(report['expandedPayloadAliases']['rocm-sdk-devel']['sameFileAsCore'])
            self.assertIn('local',report['sourceManifest']['scope'].lower())

    def test_legacy_runtime_without_sdk_bootstrap_remains_unverified(self):
        checker=self.load_checker()
        with tempfile.TemporaryDirectory() as directory:
            executable,site=self.fixture(Path(directory));(site/'torch/_rocm_init.py').unlink()
            report=checker.inspect_runtime(executable,search_paths=[str(site)],environ={})
            self.assertFalse(report['errors'],report)
            self.assertEqual(report['status'],'incomplete')
            self.assertTrue(any(w['code']=='bootstrap-unverified' for w in report['warnings']))

    def test_dynamic_bootstrap_arguments_are_incomplete_not_incompatible(self):
        checker=self.load_checker()
        for expression in ["check_version=compile_version()", "preload_shortnames=select_libraries()", "env_override=choose_override()"]:
            with self.subTest(expression=expression), tempfile.TemporaryDirectory() as directory:
                executable,site=self.fixture(Path(directory))
                (site/'torch/_rocm_init.py').write_text(f'def initialize():\n rocm_sdk.initialize_process({expression})\n')
                report=checker.inspect_runtime(executable,search_paths=[str(site)],environ={})
                self.assertFalse(report['errors'],report)
                self.assertEqual(report['status'],'incomplete')

    def test_dynamic_sdk_suffix_does_not_search_an_invented_package(self):
        checker=self.load_checker()
        with tempfile.TemporaryDirectory() as directory:
            executable,site=self.fixture(Path(directory))
            path=site/'rocm_sdk/_dist_info.py'
            path.write_text(path.read_text().replace("PY_PACKAGE_SUFFIX_NONCE=''",'PY_PACKAGE_SUFFIX_NONCE=choose_suffix()'))
            for name in ['_rocm_sdk_core','_rocm_sdk_libraries']:
                (site/name).rename(site/(name+'_actual'))
            report=checker.inspect_runtime(executable,search_paths=[str(site)],environ={})
            self.assertFalse(report['errors'],report)
            self.assertEqual(report['status'],'incomplete')
            self.assertTrue(any(w['code']=='sdk-suffix-unverified' for w in report['warnings']))

    def test_dynamic_version_keeps_checks_of_known_required_preloads(self):
        checker=self.load_checker()
        for failure,expected in [('missing','required-dll-missing'),('corrupt','invalid-pe'),('sdk-missing','sdk-missing')]:
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                executable,site=self.fixture(Path(directory))
                path=site/'torch/_rocm_init.py'
                path.write_text(path.read_text().replace("check_version='7.15.0a20260728'",'check_version=compile_version()'))
                dll=site/'_rocm_sdk_core/bin/amdhip64_7.dll'
                if failure=='missing':dll.unlink()
                elif failure=='corrupt':dll.write_bytes(b'not PE')
                else:(site/'rocm_sdk/__init__.py').unlink()
                report=checker.inspect_runtime(executable,search_paths=[str(site)],environ={})
                self.assertTrue(any(error['code']==expected for error in report['errors']),report)

    def test_unreadable_optional_metadata_does_not_reject_runtime(self):
        checker=self.load_checker()
        with tempfile.TemporaryDirectory() as directory:
            executable,site=self.fixture(Path(directory))
            original=Path.read_text
            def read(path,*args,**kwargs):
                if path.name=='METADATA':raise PermissionError('fixture access denied')
                return original(path,*args,**kwargs)
            with patch.object(Path,'read_text',read):
                report=checker.inspect_runtime(executable,search_paths=[str(site)],environ={})
            self.assertFalse(report['errors'],report)
            self.assertTrue(any(w['code']=='metadata-unverified' for w in report['warnings']))

    def test_selected_provenance_follows_process_path_not_alphabetical_order(self):
        checker=self.load_checker()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ['a-shadowed','z-selected']:
                (root/name).mkdir()
                executable,site=self.fixture(root/name)
                manifest=site/'_rocm_sdk_core/share/therock/therock_manifest.json'
                manifest.parent.mkdir(parents=True)
                manifest.write_text(json.dumps({'the_rock_commit':name}))
                (site/'_rocm_sdk_core/.info/version').write_text(name)
            selected=root/'z-selected/Lib/site-packages'; shadowed=root/'a-shadowed/Lib/site-packages'
            report=checker.inspect_runtime(executable,search_paths=[str(selected),str(shadowed)],environ={})
            self.assertFalse(report['errors'],report)
            self.assertEqual(report['sourceManifest']['data']['the_rock_commit'],'z-selected')
            self.assertEqual(report['expandedPayloadVersions']['rocm-sdk-core'],'z-selected')
            self.assertTrue(any(not entry.get('selected',True) for entry in report['packages']))

    def test_unreadable_sources_are_unverified_without_false_dll_claims(self):
        checker=self.load_checker()
        for source in ['torch/version.py','torch/_rocm_init.py','rocm_sdk/_dist_info.py']:
            with self.subTest(source=source), tempfile.TemporaryDirectory() as directory:
                executable,site=self.fixture(Path(directory))
                (site/source).write_text('not valid python !!!')
                report=checker.inspect_runtime(executable,search_paths=[str(site)],environ={})
                self.assertFalse(report['errors'],report)
                self.assertEqual(report['status'],'incomplete')

    def test_unwritable_report_warns_and_preserves_runtime_rejection(self):
        checker=self.load_checker()
        valid={'errors':[],'warnings':[],'status':'files-checked'}
        with patch.object(checker,'inspect_runtime',return_value=valid), patch.object(checker,'write_report',side_effect=PermissionError('fixture denied')), patch('builtins.print'):
            result=checker.require_rocm_runtime(output_dir='unused')
            self.assertIs(result,valid)
            self.assertTrue(any(w['code']=='report-write-failed' for w in result['warnings']))
        invalid={'errors':[{'message':'required DLL missing'}],'warnings':[],'status':'failed'}
        with patch.object(checker,'inspect_runtime',return_value=invalid), patch.object(checker,'write_report',side_effect=PermissionError('fixture denied')), patch('builtins.print'):
            with self.assertRaisesRegex(RuntimeError,'required DLL missing'):
                checker.require_rocm_runtime(output_dir='unused')

    def test_cli_is_standalone_and_preserves_existing_reports(self):
        self.load_checker()
        with tempfile.TemporaryDirectory() as directory:
            executable,_=self.fixture(Path(directory));output=Path(directory)/'report.json'
            command=[sys.executable,'-I','-S',str(SCRIPT),'--python',str(executable),'--output',str(output)]
            result=subprocess.run(command,capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr)
            original=output.read_bytes();report=json.loads(original)
            self.assertFalse(report['gpuProbePerformed'])
            repeat=subprocess.run(command,capture_output=True,text=True,timeout=10)
            self.assertNotEqual(repeat.returncode,0)
            self.assertEqual(output.read_bytes(),original)


class RendererPreflightTests(unittest.TestCase):
    def test_invalid_windows_rocm_fails_before_torch_import(self):
        import builtins
        import server
        original=builtins.__import__;calls=[]
        def importer(name,*args,**kwargs):
            if name=='torch':
                calls.append(name)
                raise AssertionError('Torch must not be imported before preflight rejection')
            return original(name,*args,**kwargs)
        with patch.object(server.sys,'platform','win32'), patch.dict(os.environ,{'ANIMESCALER_BACKEND':'rocm'}), patch('server.require_rocm_runtime',create=True,side_effect=RuntimeError('preflight rejected')), patch('builtins.__import__',side_effect=importer):
            with self.assertRaisesRegex(RuntimeError,'preflight rejected'):
                server.Renderer()
        self.assertEqual(calls,[])

    def test_tensorrt_and_invalid_backend_do_not_run_rocm_preflight(self):
        import builtins
        import server
        original=builtins.__import__
        def importer(name,*args,**kwargs):
            if name=='torch':raise RuntimeError('stop before any GPU')
            return original(name,*args,**kwargs)
        with patch.object(server.sys,'platform','win32'), patch('server.require_rocm_runtime',create=True) as check, patch('builtins.__import__',side_effect=importer):
            with patch.dict(os.environ,{'ANIMESCALER_BACKEND':'tensorrt'}), self.assertRaisesRegex(RuntimeError,'stop before any GPU'):
                server.Renderer()
            with patch.dict(os.environ,{'ANIMESCALER_BACKEND':'wrong'}), self.assertRaisesRegex(RuntimeError,'desconhecido'):
                server.Renderer()
            check.assert_not_called()


if __name__=='__main__':
    unittest.main()
