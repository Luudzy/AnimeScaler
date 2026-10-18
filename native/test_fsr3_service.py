import unittest
from unittest.mock import Mock
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from aiohttp import web, ClientSession, WSMsgType
from server import make_app
from fsr3_worker import Fsr3Renderer, Fsr3Worker, REQUEST
from protocol import HEADER, MAGIC


class Fsr3ContractTests(unittest.TestCase):
    def test_native_pipe_controls_preserve_frame_delta_and_payload(self):
        worker=Fsr3Worker.__new__(Fsr3Worker)
        worker.process=Mock()
        worker.process.stdin.write.side_effect=lambda value: min(101,len(value))
        response=HEADER.pack(MAGIC,4,80,80,80,80)
        chunks=[response,b'x'*(80*80*4)]
        worker.read_exact=lambda stream,length: chunks.pop(0)
        body=HEADER.pack(MAGIC,4,40,40,80,80)+b'\0'*(40*40*4)
        reply=worker._exchange(body,True,1001/24)
        self.assertEqual(len(reply),HEADER.size+80*80*4)
        first=worker.process.stdin.write.call_args_list[0].args[0]
        self.assertEqual(REQUEST.unpack_from(first),(MAGIC,4,40,40,80,80,1,41708))
        self.assertGreater(worker.process.stdin.write.call_count,1)

    def test_response_identity_mismatch_is_rejected(self):
        worker=Fsr3Worker.__new__(Fsr3Worker);worker.process=Mock()
        worker.process.stdin.write.side_effect=lambda value:len(value)
        worker.read_exact=lambda stream,length: HEADER.pack(MAGIC,99,80,80,80,80)
        data=HEADER.pack(MAGIC,4,40,40,80,80)+b'\0'*(40*40*4)
        with self.assertRaises(ConnectionError): worker._exchange(data,False,41.7)

    def test_renderer_does_not_reuse_frames_or_clear_reset_before_success(self):
        renderer=Fsr3Renderer.__new__(Fsr3Renderer)
        renderer.worker=Mock();renderer.shape=None;renderer.history_reset_pending=True;renderer.frame_delta_ms=1001/24;renderer.frame_revision=0
        data=HEADER.pack(MAGIC,4,40,40,80,80)+b'\0'*(40*40*4)
        renderer.worker.infer.side_effect=RuntimeError('GPU failure')
        with self.assertRaises(RuntimeError):renderer.render(data)
        self.assertTrue(renderer.history_reset_pending)
        renderer.worker.infer.side_effect=None;renderer.worker.infer.return_value=b'output'
        renderer.render(data);renderer.render(data)
        self.assertEqual(renderer.worker.infer.call_count,3)
        self.assertEqual(renderer.frame_revision,2)
        self.assertFalse(renderer.history_reset_pending)
        with self.assertRaises(ValueError):renderer.render(data,'adore')
        with self.assertRaises(ValueError):renderer.render(data,transport='jpeg')


class Fsr3StreamTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.renderer=Mock()
        self.renderer.backend='d3d12'
        self.renderer.info={'protocol':1,'models':['fsr3-native']}
        self.executor=ThreadPoolExecutor(max_workers=1)
        self.runner=web.AppRunner(make_app(self.renderer,self.executor,'test-token',port=8767))
        await self.runner.setup()
        site=web.TCPSite(self.runner,'127.0.0.1',0);await site.start()
        self.url=f'http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}'
        self.client=ClientSession(headers={'Host':'127.0.0.1:8767','Authorization':'Bearer test-token'})

    async def asyncTearDown(self):
        await self.client.close();await self.runner.cleanup();self.executor.shutdown()

    async def connect(self,query=''):
        socket=await self.client.ws_connect(self.url+'/stream?model=fsr3-native'+query)
        await socket.send_str('test-token')
        return socket

    async def test_history_has_exclusive_owner_and_releases_on_disconnect(self):
        first=await self.connect();await first.receive_json()
        second=await self.connect()
        self.assertEqual((await second.receive()).type,WSMsgType.CLOSE)
        async with self.client.post(self.url+'/frame',data=b'bad') as response:
            self.assertEqual(response.status,429)
        await first.close();await asyncio.sleep(.02)
        third=await self.connect();self.assertEqual((await third.receive_json())['model'],'fsr3-native')
        self.assertEqual(self.renderer.reset_history.call_count,2)
        await third.close()

    async def test_fsr3_rejects_lossy_or_unsupported_transport_before_handshake(self):
        for transport in ('jpeg','rgb'):
            socket=await self.connect('&transport='+transport)
            self.assertEqual((await socket.receive()).type,WSMsgType.CLOSE)

    async def test_controls_and_authentication(self):
        async with self.client.get(self.url+'/health',headers={'Authorization':''}) as response:
            self.assertEqual(response.status,401)
        async with self.client.get(self.url+'/health',headers={'Origin':'https://untrusted.example'}) as response:
            self.assertEqual(response.status,403)
        first=await self.connect();await first.receive_json()
        await first.send_str(json.dumps({'reset':True,'frameDeltaMs':41.7}))
        await first.send_str(json.dumps({'reset':False,'frameDeltaMs':float('nan')}))
        self.assertEqual((await first.receive()).type,WSMsgType.CLOSE)
        self.renderer.frame_context.assert_called_once_with(True,41.7)


if __name__=='__main__':unittest.main()
