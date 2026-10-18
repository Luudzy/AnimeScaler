import asyncio
from contextlib import redirect_stdout
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from threading import Event
from types import SimpleNamespace
import unittest

from aiohttp import web
from protocol import HEADER, MAGIC
from server import make_app,make_app_runner


def request(data):
    async def read():return data
    return SimpleNamespace(read=read)


def disconnected_request():
    async def read():raise ConnectionResetError('client disconnected')
    return SimpleNamespace(read=read)


class ServerBusyTests(unittest.IsolatedAsyncioTestCase):
    async def test_runner_does_not_cancel_handler_after_client_disconnect(self):
        runner=make_app_runner(web.Application())
        await runner.setup()
        try:self.assertFalse(runner.server.handler_cancellation)
        finally:await runner.cleanup()

    def get_frame_handler(self,app):
        return next(route.handler for route in app.router.routes() if route.method=='POST' and route.resource.canonical=='/frame')

    async def test_invalid_frame_releases_busy_state(self):
        with ThreadPoolExecutor(max_workers=1) as executor:
            renderer=SimpleNamespace(render=lambda data:(b'ok',1.0))
            app=make_app(renderer,executor,'token');handler=self.get_frame_handler(app)
            invalid=await handler(request(b'invalid'))
            valid_data=HEADER.pack(MAGIC,1,40,40,80,80)+bytes(40*40*4)
            valid=await handler(request(valid_data))
        self.assertEqual(invalid.status,400)
        self.assertEqual(valid.status,200)

    async def test_body_disconnect_is_not_logged_as_backend_failure_and_releases_busy(self):
        calls=[]
        renderer=SimpleNamespace(render=lambda data:(calls.append(data) or (b'ok',1.0)))
        with ThreadPoolExecutor(max_workers=1) as executor:
            app=make_app(renderer,executor,'token');handler=self.get_frame_handler(app)
            output=StringIO()
            with redirect_stdout(output):
                disconnected=await handler(disconnected_request())
                data=HEADER.pack(MAGIC,1,40,40,80,80)+bytes(40*40*4)
                valid=await handler(request(data))
        self.assertEqual(disconnected.status,499)
        self.assertNotIn('processing failed',output.getvalue())
        self.assertEqual(valid.status,200)
        self.assertEqual(calls,[data])

    async def test_second_frame_gets_busy_while_first_render_runs(self):
        started=Event();release=Event()
        def render(data):
            started.set()
            if not release.wait(2):raise TimeoutError('test render was not released')
            return b'ok',1.0
        data=HEADER.pack(MAGIC,1,40,40,80,80)+bytes(40*40*4)
        with ThreadPoolExecutor(max_workers=1) as executor:
            app=make_app(SimpleNamespace(render=render),executor,'token');handler=self.get_frame_handler(app)
            first=asyncio.create_task(handler(request(data)))
            self.assertTrue(await asyncio.wait_for(asyncio.to_thread(started.wait),1))
            busy=await handler(request(data))
            release.set()
            completed=await first
        self.assertEqual(busy.status,429)
        self.assertEqual(completed.status,200)


if __name__=='__main__':unittest.main()
