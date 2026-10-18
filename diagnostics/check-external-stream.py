"""Published external models through the live extension transport, not just loaders."""
import asyncio,json,struct
from pathlib import Path
import aiohttp,numpy as np,onnxruntime as ort
import torch

ROOT=Path(__file__).resolve().parents[1];HEADER=struct.Struct('<6I')
async def main():
    connection=json.loads((ROOT/'native/connection.json').read_text());report={};torch.set_num_threads(2)
    async with aiohttp.ClientSession() as client:
        for name,scale in [('hfa2k-cugan',2),('realesr-anime',4),('span-mssim',4)]:
            reference=ort.InferenceSession(str(ROOT/'models'/f'{name}.onnx'),providers=['CPUExecutionProvider'])
            async with client.ws_connect(connection['url']+f'/stream?model={name}&transport=rgb',max_msg_size=40*1024*1024) as socket:
                await socket.send_str(connection['token']);info=await socket.receive_json(timeout=30)
                assert info['model']==name and info['precision']=='fp16';errors=[]
                for index,(w,h,factor) in enumerate([(48,40,2),(64,48,scale),(48,40,2)]):
                    pixels=np.random.default_rng(index).integers(0,256,(h,w,4),dtype=np.uint8);ow=w*factor;oh=h*factor
                    await socket.send_bytes(HEADER.pack(0x31435341,index,w,h,ow,oh)+pixels.tobytes())
                    timing=await socket.receive_json(timeout=120);assert 'processingMs' in timing,timing
                    response=await socket.receive(timeout=120);assert isinstance(response.data,bytes),response
                    assert HEADER.unpack_from(response.data)==(0x33435341,index,ow,oh,ow,oh)
                    actual=np.frombuffer(response.data,np.uint8,offset=24).reshape(oh,ow,3)/255
                    x=pixels[...,:3].transpose(2,0,1)[None].astype(np.float32)/255
                    expected=torch.from_numpy(reference.run(None,{'input':x})[0])
                    if factor!=scale:expected=torch.nn.functional.interpolate(expected,size=(oh,ow),mode='bilinear',align_corners=False)
                    delta=np.abs(actual-expected[0].permute(1,2,0).clamp(0,1).numpy())
                    assert delta.mean()<.005 and delta.max()<.05,(name,delta.mean(),delta.max())
                    errors.append({'input':[w,h],'output':[ow,oh],'mae':float(delta.mean()),'maxError':float(delta.max())})
                report[name]={'passed':True,'nativeScale':scale,'cases':errors}
    (ROOT/'diagnostics/external-stream-validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
asyncio.run(main())
