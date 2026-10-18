"""Private local connection token creation and atomic publication."""
import re
import secrets

from artifact_io import atomic_json_write


def new_connection_token():
    return secrets.token_hex(32)


def publish_connection_file(path,url,token):
    if not isinstance(token,str) or re.fullmatch(r'[0-9a-f]{64}',token) is None:
        raise ValueError('Invalid local connection token')
    if not isinstance(url,str) or not url.startswith('http://127.0.0.1:'):
        raise ValueError('Invalid local connection URL')
    atomic_json_write({'url':url,'token':token},path)
