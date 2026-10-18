import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from connection_file import new_connection_token, publish_connection_file


class ConnectionFileTests(unittest.TestCase):
    def test_new_token_is_fresh_and_fixed_format(self):
        first=new_connection_token();second=new_connection_token()
        self.assertRegex(first,r'^[0-9a-f]{64}$')
        self.assertNotEqual(first,second)

    def test_publish_replaces_previous_token(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'connection.json';path.write_text(json.dumps({'url':'old','token':'old-token'}))
            token=new_connection_token();publish_connection_file(path,'http://127.0.0.1:8766',token)
            result=json.loads(path.read_text())
            self.assertEqual(result,{'url':'http://127.0.0.1:8766','token':token})
            self.assertNotEqual(result['token'],'old-token')

    def test_replace_failure_preserves_previous_connection_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'connection.json';path.write_text('{"url":"old","token":"old-token"}')
            with patch('artifact_io.os.replace',side_effect=PermissionError('file locked')):
                with self.assertRaisesRegex(PermissionError,'file locked'):
                    publish_connection_file(path,'http://127.0.0.1:8766',new_connection_token())
            self.assertEqual(json.loads(path.read_text()),{'url':'old','token':'old-token'})


if __name__=='__main__': unittest.main()
