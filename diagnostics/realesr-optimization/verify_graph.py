"""Actual bridge graph parity/freshness; run only after other GPU measurements."""
import argparse,hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'native'))
os.environ['ANIMESCALER_REALESR_OPTIMIZE']='1'
os.environ['ANIMESCALER_HIP_GRAPH']='1'
import torch,numpy as np
from server import Renderer
from protocol import HEADER,MAGIC

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
r=Renderer();rows=[];frame_id=0
with torch.inference_mode():
    for oh,ow in [(720,1280),(1440,2560),(720,1280)]:
        previous_graph=None;output_hashes=[]
        for index in range(2):
            pixels=np.random.default_rng(3100+frame_id).integers(0,256,(360,640,4),dtype=np.uint8)
            packet=HEADER.pack(MAGIC,frame_id,640,360,ow,oh)+pixels.tobytes()
            body,_=r.render(packet,'realesr-anime','rgb',False)
            if r.graph is None or r.last_reused:raise AssertionError('Actual fresh graph inference required')
            if previous_graph is not None and r.graph is not previous_graph:raise AssertionError('Second input must replay same graph')
            previous_graph=r.graph
            value=torch.from_numpy(pixels[...,:3].copy()).permute(2,0,1).unsqueeze(0).cuda().half()/255
            expected=r.model.infer_to(value,(oh,ow))
            expected=expected[0].permute(1,2,0).clamp(0,1).mul(255).round().byte().cpu().numpy()
            actual=np.frombuffer(body,dtype=np.uint8,offset=HEADER.size).reshape(oh,ow,3)
            delta=np.abs(actual.astype(np.int16)-expected.astype(np.int16))
            if delta.max()!=0:raise AssertionError('Graph/eager bridge pixels differ')
            output_hashes.append(hashlib.sha256(body[HEADER.size:]).hexdigest())
            rows.append({'input':[640,360],'output':[ow,oh],'frameId':frame_id,
                         'inputSha256':hashlib.sha256(pixels.tobytes()).hexdigest(),'outputSha256':output_hashes[-1],
                         'maxQuantizedDifference':int(delta.max()),'meanQuantizedDifference':float(delta.mean()),
                         'warmup':r.last_warmup,'reused':r.last_reused})
            frame_id+=1
        if len(set(output_hashes))!=2:raise AssertionError('Distinct inputs did not produce distinct outputs')
a.output.write_text(json.dumps({'status':'PASS','gpu':torch.cuda.get_device_name(),'hip':torch.version.hip,
                               'scope':'Actual Renderer.render graph vs same-model eager RGB output; two distinct frames per shape, shape switch and return',
                               'frames':rows},indent=2),encoding='utf-8')
print('PASS: 6 graph/eager RGB comparisons, exact pixels and fresh distinct outputs',flush=True)
