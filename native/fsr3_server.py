"""Separate authenticated localhost FSR3 service; no ROCm/Torch initialization."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

# Python isolated mode omits the script directory from module lookup.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aiohttp import web
from connection_file import new_connection_token, publish_connection_file
from fsr3_worker import Fsr3Renderer
from server import make_app

ROOT = Path(__file__).resolve().parents[1]


async def main():
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='fsr3')
    renderer = None
    try:
        renderer = await asyncio.get_running_loop().run_in_executor(executor, Fsr3Renderer, ROOT)
        connection = ROOT / 'native/fsr3-connection.json'
        token = new_connection_token()
        app = make_app(renderer, executor, token, port=8767)
        runner = web.AppRunner(app, access_log=None, handler_cancellation=False)
        await runner.setup()
        try:
            await web.TCPSite(runner, '127.0.0.1', 8767).start()
            publish_connection_file(connection, 'http://127.0.0.1:8767', token)
            print(f"AnimeScaler FSR {renderer.info['provider']} DirectX 12 pronto: {renderer.info['gpu']}. Experimental, sem frame generation.", flush=True)
            await asyncio.Event().wait()
        finally:
            await runner.cleanup()
    finally:
        if renderer is not None:
            renderer.close()
        executor.shutdown(wait=True)


if __name__ == '__main__':
    asyncio.run(main())
