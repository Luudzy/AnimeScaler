"""Check changing frames and resizing through the same authenticated GPU stream."""
import asyncio
import json
from pathlib import Path
import struct
import aiohttp
import numpy as np
import onnxruntime as ort

root = Path(__file__).resolve().parents[1]
connection = json.loads((root/'native/connection.json').read_text())
fixture = root/'diagnostics/native-fixture'
data = np.load(fixture/'input.npy', allow_pickle=False)
reference = np.load(fixture/'reference.npy', allow_pickle=False)
session = ort.InferenceSession(str(root/'models/2x_Adore_renarchi_fp32.onnx'), providers=['CPUExecutionProvider'])
blank = np.zeros((1,3,40,40), np.float32)
blank_ref = session.run(None, {session.get_inputs()[0].name:blank})[0]
blank_same_size = np.zeros_like(data)
blank_same_ref = session.run(None, {session.get_inputs()[0].name:blank_same_size})[0]
header = struct.Struct('<6I')


async def main():
    errors = []
    async with aiohttp.ClientSession() as client:
        async with client.ws_connect(connection['url']+'/stream', max_msg_size=16*1024*1024) as socket:
            await socket.send_str(connection['token'])
            info = await socket.receive_json()
            assert info['backend'] == 'rocm'
            for index, (value, expected) in enumerate([(data,reference),(blank_same_size,blank_same_ref),(blank,blank_ref),(data,reference)]):
                height,width = value.shape[2:]
                pixels = np.full((height,width,4),255,np.uint8)
                pixels[...,:3] = (value[0].transpose(1,2,0)*255).round().astype(np.uint8)
                await socket.send_bytes(header.pack(0x31435341,index,width,height,width*2,height*2)+pixels.tobytes())
                timings = await socket.receive_json(timeout=60)
                assert 'processingMs' in timings, timings
                response = await socket.receive(timeout=60)
                assert header.unpack_from(response.data) == (0x31435341,index,width*2,height*2,width*2,height*2)
                actual = np.frombuffer(response.data,np.uint8,offset=24).reshape(height*2,width*2,4)
                assert (actual[...,3]==255).all()
                delta = np.abs(actual[...,:3]/255-expected[0].transpose(1,2,0).clip(0,1))
                assert delta.mean()<.005 and delta.max()<.05, (delta.mean(),delta.max())
                errors.append({'input':[width,height], 'meanAbsoluteError':float(delta.mean()),'maxAbsoluteError':float(delta.max())})
    report = {'passed':True,'cases':errors,'scope':'Original image, black frame at new size, original image again; compared with ONNX FP32 CPU'}
    (root/'diagnostics/native-stream-quality.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


asyncio.run(main())
