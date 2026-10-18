"""Actual authenticated WebSocket bridge, distinct frames and model/size switches.

Expected output is computed with the CPU FP32 port, not a second GPU process.
Never prints credentials or image bytes.
"""
import asyncio,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'native'))
import aiohttp,numpy as np,torch
from protocol import HEADER,MAGIC
from model_registry import build_registry

async def main():
    torch.set_num_threads(2)
    connection=json.loads((ROOT/'native/connection.json').read_text())
    token=connection['token'];registry=build_registry(ROOT);rows=[]
    names=['anime4k-s','anime4k-m','realesr-anime4k-s-360']
    async with aiohttp.ClientSession() as session:
        async with session.get('http://127.0.0.1:8766/health',headers={'Authorization':'Bearer '+token}) as response:
            if response.status!=200:raise AssertionError('Health authentication failed')
            info=await response.json()
            if info['backend']!='rocm' or any(n not in info['models'] for n in names):raise AssertionError('Model availability/backend mismatch')
        for name in names:
            model=registry.load(name,device='cpu',precision='fp32')
            async with session.ws_connect(f'http://127.0.0.1:8766/stream?model={name}&transport=rgb') as socket:
                await socket.send_str(token)
                ready=await socket.receive_json()
                if ready.get('error'):raise AssertionError('Bridge rejected selected model')
                for index,(h,w,oh,ow) in enumerate([(40,64,80,128),(48,80,72,120)]):
                    pixels=np.random.default_rng(929+index).integers(0,256,(h,w,4),dtype=np.uint8)
                    packet=HEADER.pack(MAGIC,index,w,h,ow,oh)+pixels.tobytes()
                    await socket.send_bytes(packet)
                    stats=await socket.receive_json()
                    if stats.get('reused') or stats.get('error'):raise AssertionError('Frame reuse/error in actual stream test')
                    message=await socket.receive()
                    if message.type!=aiohttp.WSMsgType.BINARY:raise AssertionError('No binary bridge output')
                    header=HEADER.unpack_from(message.data)
                    if header[1:]!=(index,ow,oh,ow,oh):raise AssertionError('Output identity/size mismatch')
                    actual=np.frombuffer(message.data,dtype=np.uint8,offset=HEADER.size).reshape(oh,ow,3)
                    value=torch.from_numpy(pixels[:,:,:3].copy()).permute(2,0,1)[None].float()/255
                    with torch.inference_mode():expected=model.infer_to(value,(oh,ow))[0].permute(1,2,0).clamp(0,1).mul(255).round().byte().numpy()
                    delta=np.abs(actual.astype(np.int16)-expected.astype(np.int16))
                    row={'model':name,'input':[w,h],'output':[ow,oh],'frameId':index,
                        'maxByteDifference':int(delta.max()),'meanByteDifference':float(delta.mean()),
                        'inputSha256':hashlib.sha256(pixels.tobytes()).hexdigest(),
                        'status':'PASS' if delta.max()<=2 and delta.mean()<=.2 else 'FAIL'}
                    rows.append(row)
            registry.unload(name)
    result={'status':'PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL','cases':rows,
        'scope':'Actual authenticated loopback WebSocket, CPU FP32 equation/reference parity; no browser/player FPS measurement',
        'tolerance':{'maxByteDifference':2,'meanByteDifference':.2}}
    out=Path(__file__).with_name('stream-validation.json')
    if out.exists():raise FileExistsError('Preserve previous stream report')
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'status':result['status'],'cases':len(rows)}))
    if result['status']!='PASS':raise SystemExit(1)
if __name__=='__main__':asyncio.run(main())
