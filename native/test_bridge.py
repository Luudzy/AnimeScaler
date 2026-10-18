"""Contract + HTTP checks without loading GPU libraries."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import unittest
from aiohttp import web, ClientSession
from protocol import HEADER, MAGIC, decode_frame
from server import make_app


def packet(width=40, height=40):
    return HEADER.pack(MAGIC, 7, width, height, width*2, height*2) + bytes(width*height*4)


class ProtocolTests(unittest.TestCase):
    def test_validation(self):
        self.assertEqual(decode_frame(packet()), (7, 40, 40, 80, 80))
        for value in [b'', packet()[:-1], packet(41), packet(36), HEADER.pack(MAGIC, 1, 40, 40, 81, 80)+bytes(6400)]:
            with self.assertRaises(ValueError):
                decode_frame(value)


class HttpTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import threading
        self.started = threading.Event()
        self.release = threading.Event()
        owner = self
        class FakeRenderer:
            info = {'protocol': 1, 'models': ['adore', 'adore-lite']}
            frame_revision=0
            def render(self, data, model_name='adore',transport='rgba',allow_repeat=False):
                owner.last_model = model_name
                owner.started.set()
                owner.release.wait(timeout=2)
                if allow_repeat:return b'repeat',1
                self.frame_revision+=1
                return b'output', 1
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.runner = web.AppRunner(make_app(FakeRenderer(), self.executor, 'test-token'))
        await self.runner.setup()
        site = web.TCPSite(self.runner, '127.0.0.1', 0)
        await site.start()
        self.url = f'http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}'
        self.client = ClientSession(headers={'Host': '127.0.0.1:8766', 'Authorization': 'Bearer test-token'})

    async def asyncTearDown(self):
        self.release.set()
        await self.client.close()
        await self.runner.cleanup()
        self.executor.shutdown()

    async def test_auth_origin_and_dimensions(self):
        for headers, code in [({'Authorization': ''}, 401), ({'Origin':'https://untrusted.example'}, 403),
                              ({'Origin':'http://127.0.0.1:61858'}, 403),
                              ({'Origin':'http://127.0.0.1:8765'}, 200),
                              ({'Host':'attacker.example'}, 403), ({}, 200)]:
            async with self.client.get(self.url+'/health', headers=headers) as response:
                self.assertEqual(response.status, code)
        async with self.client.post(self.url+'/frame', data=b'bad') as response:
            self.assertEqual(response.status, 400)

    async def test_busy_requests_are_rejected_not_queued(self):
        first = asyncio.create_task(self.client.post(self.url+'/frame', data=packet()))
        for _ in range(100):
            if self.started.is_set(): break
            await asyncio.sleep(.01)
        self.assertTrue(self.started.is_set())
        async with self.client.post(self.url+'/frame', data=packet()) as response:
            self.assertEqual(response.status, 429)
        self.release.set()
        response = await first
        self.assertEqual(response.status, 200)
        await response.read()

    async def test_stream_authentication_and_binary_reply(self):
        from aiohttp import WSMsgType
        async with self.client.ws_connect(self.url+'/stream') as socket:
            await socket.send_str('wrong-token')
            response = await socket.receive()
            self.assertEqual(response.type, WSMsgType.CLOSE)
        self.release.set()
        async with self.client.ws_connect(self.url+'/stream') as socket:
            await socket.send_str('test-token')
            self.assertEqual((await socket.receive_json())['protocol'], 1)
            await socket.send_bytes(packet())
            self.assertEqual((await socket.receive_json())['processingMs'], 1)
            self.assertEqual((await socket.receive()).data, b'output')

    async def test_diagnostic_stream_requires_fixed_browser_origin(self):
        from aiohttp import WSServerHandshakeError
        with self.assertRaises(WSServerHandshakeError) as denied:
            await self.client.ws_connect(self.url+'/stream', origin='http://127.0.0.1:61858')
        self.assertEqual(denied.exception.status,403)
        async with self.client.ws_connect(self.url+'/stream', origin='http://127.0.0.1:8765') as socket:
            await socket.send_str('test-token')
            self.assertEqual((await socket.receive_json())['protocol'],1)
        self.assertFalse(self.started.is_set(),'origin validation needs no renderer workload')

    async def test_stream_model_selection(self):
        from aiohttp import WSMsgType
        self.release.set()
        async with self.client.ws_connect(self.url+'/stream?model=unknown') as socket:
            await socket.send_str('test-token')
            self.assertEqual((await socket.receive()).type, WSMsgType.CLOSE)
        async with self.client.ws_connect(self.url+'/stream?model=adore-lite') as socket:
            await socket.send_str('test-token')
            self.assertEqual((await socket.receive_json())['model'], 'adore-lite')
            await socket.send_bytes(packet())
            await socket.receive_json()
            self.assertEqual((await socket.receive()).data, b'output')
            self.assertEqual(self.last_model, 'adore-lite')

    async def test_maximum_1080p_input_is_accepted_by_both_transports(self):
        self.release.set()
        data = packet(1920, 1080)
        async with self.client.post(self.url+'/frame', data=data) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(await response.read(), b'output')
        async with self.client.ws_connect(self.url+'/stream') as socket:
            await socket.send_str('test-token')
            await socket.receive_json()
            await socket.send_bytes(data)
            self.assertEqual((await socket.receive_json())['processingMs'], 1)
            self.assertEqual((await socket.receive()).data, b'output')

    async def test_repeat_requires_a_result_delivered_on_the_same_connection(self):
        self.release.set()
        async def request(socket):
            await socket.send_bytes(packet());await socket.receive_json()
            return (await socket.receive()).data
        async with self.client.ws_connect(self.url+'/stream?repeat=1') as first:
            await first.send_str('test-token');await first.receive_json()
            self.assertEqual(await request(first),b'output')
            self.assertEqual(await request(first),b'repeat')
            async with self.client.ws_connect(self.url+'/stream?repeat=1') as second:
                await second.send_str('test-token');await second.receive_json()
                self.assertEqual(await request(second),b'output')
                self.assertEqual(await request(first),b'output')
                self.assertEqual(await request(first),b'repeat')


if __name__ == '__main__':
    unittest.main()
