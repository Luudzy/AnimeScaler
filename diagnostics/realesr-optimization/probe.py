"""Throwaway controlled probe, not production or player FPS."""
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'native'))
import server  # use the same MIOpen cache directories as the bridge
import torch
from user_models import load_user_model
from benchmark_timing import timed_iterations, fill_distinct_input

p = argparse.ArgumentParser()
p.add_argument('--layout', choices=['nchw','nhwc'], default='nchw')
p.add_argument('--search', action='store_true')
p.add_argument('--graph', action='store_true')
p.add_argument('--optimized', action='store_true')
p.add_argument('--ratio', type=float, default=4)
p.add_argument('--tile', type=int, default=768)
p.add_argument('--max-pixels', type=int, default=1280*720)
p.add_argument('--width', type=int, default=640)
p.add_argument('--height', type=int, default=360)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
torch.set_num_threads(2)
torch.backends.cudnn.benchmark = a.search
layout = torch.channels_last if a.layout == 'nhwc' else torch.contiguous_format
m, meta = load_user_model(ROOT, 'realesr-anime', False)
if a.optimized:
    from realesr_optimized import RealESRAnimeVideo
    m = RealESRAnimeVideo(m,tile=a.tile,max_pixels=a.max_pixels)
else:
    from tiled_model import TiledModel
    m = TiledModel(m,4)
m = m.cuda().half().to(memory_format=layout)
x = torch.empty((1,3,a.height,a.width),device='cuda',dtype=torch.float16,memory_format=layout)
size=(int(a.height*a.ratio),int(a.width*a.ratio))
def operation():
    if a.optimized: return m.forward_to(x,size)
    result=m(x)
    return result if result.shape[-2:]==size else torch.nn.functional.interpolate(result,size=size,mode='bilinear',align_corners=False)
with torch.inference_mode():
    fill_distinct_input(x,0)
    baseline = operation()
    parity=None
    if a.optimized:
        expected=torch.nn.functional.interpolate(m.model(x),size=size,mode='bilinear',align_corners=False)
        error=(baseline.float()-expected.float()).abs()
        parity={'maxAbsError':error.max().item(),'meanAbsError':error.mean().item()}
        del expected,error
    torch.cuda.synchronize()
    graph = None
    if a.graph:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3): operation()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream): captured = operation()
    torch.cuda.reset_peak_memory_stats()
    result = timed_iterations(graph.replay if graph else operation, torch.cuda.synchronize, 5, 15,
                              lambda i:fill_distinct_input(x,i))
    result.update({'layout':a.layout,'search':a.search,'graph':a.graph,'shape':list(x.shape),'optimized':a.optimized,
                   'ratio':a.ratio,'tile':a.tile,'parity':parity,
                   'torch':str(torch.__version__),'hip':torch.version.hip,
                   'gpu':torch.cuda.get_device_name(),'peakMiB':torch.cuda.max_memory_allocated()/1048576,
                   'status':'PASS','scope':'Model completion, distinct random frames, exploratory 5+15; not player FPS'})
    a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)
