import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Fsr3IsolatedImportTests(unittest.TestCase):
    def test_service_imports_local_worker_under_isolated_python_without_starting_service(self):
        probe = r'''
import importlib.util
import sys
import types

aiohttp = types.ModuleType("aiohttp")
aiohttp.web = types.SimpleNamespace()
sys.modules["aiohttp"] = aiohttp
server = types.ModuleType("server")
server.make_app = lambda *args, **kwargs: None
sys.modules["server"] = server

spec = importlib.util.spec_from_file_location("fsr3_server_isolated_probe", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print("isolated-import-ok")
'''
        result = subprocess.run(
            [sys.executable, '-I', '-c', probe, str(ROOT / 'native/fsr3_server.py')],
            cwd=ROOT, capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('isolated-import-ok', result.stdout)


if __name__ == '__main__':
    unittest.main()
