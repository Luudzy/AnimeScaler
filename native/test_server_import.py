import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ServerImportTests(unittest.TestCase):
    def test_import_does_not_create_runtime_cache_directories(self):
        script='''
from pathlib import Path
original=Path.mkdir
created=[]
def record(path,*args,**kwargs):
    if path.parts[-2:] in (('native-cache','db'),('native-cache','kernels')):
        created.append(path)
        return None
    return original(path,*args,**kwargs)
Path.mkdir=record
import server
assert not created, f'import created runtime caches: {created}'
'''
        result=subprocess.run([sys.executable,'-c',script],cwd=Path(__file__).parent,
            env=os.environ.copy(),text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_explicit_cache_setup_creates_directories_and_sets_defaults(self):
        from server import configure_miopen_cache
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);environment={}
            configure_miopen_cache(root,environment)
            for name in ('db','kernels'):
                path=root/'diagnostics/native-cache'/name
                self.assertTrue(path.is_dir())
                key='MIOPEN_USER_DB_PATH' if name=='db' else 'MIOPEN_CUSTOM_CACHE_DIR'
                self.assertEqual(environment[key],str(path))


if __name__=='__main__': unittest.main()
