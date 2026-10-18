import unittest
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import io
import json
from types import SimpleNamespace

from protocol import HEADER,MAGIC
from server import make_app,native_error_code,safe_native_error_log


class NativeErrorTests(unittest.TestCase):
    def test_error_codes_and_logs_do_not_copy_exception_messages(self):
        private=RuntimeError('token=private C:\\Users\\secret')
        self.assertEqual(native_error_code(TimeoutError('private detail')),'BACKEND_TIMEOUT')
        self.assertEqual(native_error_code(ValueError('path C:\\Users\\secret')),'BACKEND_INVALID_RESPONSE')
        self.assertEqual(native_error_code(private),'BACKEND_UNAVAILABLE')
        record=safe_native_error_log('opencl',private)
        self.assertIn('opencl',record)
        self.assertIn('RuntimeError',record)
        self.assertNotIn('private',record)
        self.assertNotIn('secret',record)

    def test_frame_handler_returns_stable_codes_and_sanitized_logs(self):
        async def run():
            with ThreadPoolExecutor(max_workers=1) as executor:
                invalid_app=make_app(SimpleNamespace(render=lambda data:None),executor,'token')
                invalid_handler=next(route.handler for route in invalid_app.router.routes()
                                     if route.method=='POST' and route.resource.canonical=='/frame')
                async def read_bad():return b'invalid'
                invalid=await invalid_handler(SimpleNamespace(read=read_bad))
                valid_data=HEADER.pack(MAGIC,1,40,40,80,80)+bytes(40*40*4)
                async def read_good():return valid_data
                def fail_render(data):raise RuntimeError('token=private C:\\Users\\secret')
                failing_app=make_app(SimpleNamespace(render=fail_render,backend='opencl'),executor,'token')
                failing_handler=next(route.handler for route in failing_app.router.routes()
                                     if route.method=='POST' and route.resource.canonical=='/frame')
                output=io.StringIO()
                with redirect_stdout(output):
                    failed=await failing_handler(SimpleNamespace(read=read_good))
                return invalid,failed,output.getvalue()
        invalid,failed,log=asyncio.run(run())
        self.assertEqual((invalid.status,json.loads(invalid.text)),(400,{'errorCode':'BACKEND_INVALID_RESPONSE'}))
        self.assertEqual(failed.status,500)
        self.assertEqual(json.loads(failed.text),{'errorCode':'BACKEND_UNAVAILABLE'})
        self.assertIn('opencl processing failed (RuntimeError)',log)
        self.assertNotIn('private',log);self.assertNotIn('secret',log)


if __name__=='__main__':unittest.main()
