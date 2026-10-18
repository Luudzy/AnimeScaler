"""Loopback-only development assets; never serve private native credentials."""
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


class Assets(SimpleHTTPRequestHandler):
    def do_GET(self):
        path = unquote(urlsplit(self.path).path).lstrip('/')
        target = (ROOT / path).resolve()
        allowed = path.startswith(('src/', 'popup/', 'diagnostics/video-sequences/rezero-real-01/')) or path in (
            'diagnostics/fsr3-native-viewer.html',
            'diagnostics/fsr1-improvement-check.html',
            'diagnostics/fsr1-baseline-engine-20261010.js',
            'diagnostics/temporal.html',
            'diagnostics/temporal-check.js',
        )
        if not allowed or not target.is_relative_to(ROOT) or not target.is_file():
            self.send_error(404)
            return
        super().do_GET()

    def log_message(self, *_):
        pass


if __name__ == '__main__':
    HTTPServer(('127.0.0.1', 8765), lambda *args, **kwargs: Assets(*args, directory=str(ROOT), **kwargs)).serve_forever()
