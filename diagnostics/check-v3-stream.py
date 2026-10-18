"""Exercise the exported V3 through the live authenticated extension protocol."""
import asyncio
import json
import struct
from pathlib import Path
import aiohttp
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
HEADER = struct.Struct('<6I')

async def main():
    connection = json.loads((ROOT/'native/connection.json').read_text())
    reference = ort.InferenceSession(str(ROOT/'models/adore-lite-v3-experimental.onnx'), providers=['CPUExecutionProvider'])
    cases = []
    async with aiohttp.ClientSession() as client:
        async with client.ws_connect(connection['url']+'/stream?model=adore-lite-v3&transport=rgba', max_msg_size=40*1024*1024) as socket:
            await socket.send_str(connection['token'])
            info = await socket.receive_json(timeout=30)
            assert info['model'] == 'adore-lite-v3' and info['precision'] == 'fp16', info
            for index,(w,h,ow,oh) in enumerate([(80,48,160,96),(640,360,1280,720),(1280,720,1920,1080),(1920,1080,3840,2160),(80,48,160,96)]):
                pixels = np.random.default_rng(index).integers(0,256,(h,w,4),dtype=np.uint8)
                await socket.send_bytes(HEADER.pack(0x31435341,index,w,h,ow,oh)+pixels.tobytes())
                timing = await socket.receive_json(timeout=120)
                assert 'processingMs' in timing, timing
                response = await socket.receive(timeout=120)
                assert isinstance(response.data,bytes), response
                assert HEADER.unpack_from(response.data)==(0x31435341,index,ow,oh,ow,oh)
                actual = np.frombuffer(response.data,np.uint8,offset=24).reshape(oh,ow,4)
                assert np.all(actual[...,3]==255)
                case = {'input':[w,h],'output':[ow,oh],'processingMs':timing['processingMs']}
                if w==80:
                    x=pixels[...,:3].transpose(2,0,1)[None].astype(np.float32)/255
                    expected=reference.run(None,{'input':x})[0][0].transpose(1,2,0).clip(0,1)
                    error=np.abs(actual[...,:3]/255-expected)
                    assert error.mean()<.005 and error.max()<.05,(error.mean(),error.max())
                    case['maeAgainstOnnx']=float(error.mean())
                cases.append(case)
    report={'passed':True,'model':info['model'],'precision':info['precision'],'cases':cases,'scope':'Live WebSocket, exported weights, dynamic sizes and FP32 ONNX parity on small frames; not browser FPS.'}
    (ROOT/'diagnostics/v3-stream-validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

asyncio.run(main())
