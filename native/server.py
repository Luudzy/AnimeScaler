"""Local Adore FP16 bridge. Run with the existing ROCm Python, not system Python."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import io
import os
from pathlib import Path
import re
import secrets
import time
import sys
import math

# This portable Python has an explicit ._pth; add only our own module directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aiohttp import web, WSMsgType
import numpy as np
from protocol import HEADER, MAGIC, MAX_PIXELS, decode_frame
from anime4kcpp_worker import Anime4KCPPWorker, MODELS as ANIME4KCPP_MODELS
from runtime_preflight import require_rocm_runtime
from connection_file import new_connection_token,publish_connection_file

ROOT = Path(__file__).resolve().parents[1]

def configure_miopen_cache(root, environment=None):
    environment=os.environ if environment is None else environment
    for key,folder in [('MIOPEN_USER_DB_PATH','db'),('MIOPEN_CUSTOM_CACHE_DIR','kernels')]:
        directory=Path(root)/'diagnostics/native-cache'/folder
        directory.mkdir(parents=True,exist_ok=True)
        environment.setdefault(key,str(directory))


def release_inactive_models(models, keep, release):
    for model_id,model in list(models.items()):
        if model_id==keep: continue
        release(model_id,model)
        del models[model_id]


def native_error_code(error):
    if isinstance(error,TimeoutError):return 'BACKEND_TIMEOUT'
    if isinstance(error,(ValueError,TypeError)):return 'BACKEND_INVALID_RESPONSE'
    return 'BACKEND_UNAVAILABLE'


def safe_native_error_log(backend,error):
    name=backend if backend in ('rocm','tensorrt','opencl','d3d12') else 'native'
    return f'{name} processing failed ({type(error).__name__})'


def registry_service_catalog(registry,backend,root):
    metadata=registry.list_models();models=[];scales={};revisions={};availability={}
    for item in metadata:
        state=registry.availability(item.id,backend,root)
        availability[item.id]={'available':state['available'],'reason':state['reason']}
        scales[item.id]=item.native_scale
        if state['available']:
            models.append(item.id)
            digest=hashlib.sha256()
            with Path(state['artifactPath']).open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
            revisions[item.id]=digest.hexdigest()
    return {'models':models,'scales':scales,'metadata':{item.id:item.to_dict() for item in metadata},
            'revisions':revisions,'modelAvailability':availability}


def anime4kcpp_worker_availability(root):
    root=Path(root)
    exe=root/'third_party/Anime4KCPP-master/worker/build/bin/Release/anime4kcpp_worker.exe'
    opencl=root/'third_party/Anime4KCPP-master/build-opencl-md/bin/Release'
    if not exe.is_file():return {'available':False,'reason':'Anime4KCPP worker binary is missing'}
    if not opencl.is_dir():return {'available':False,'reason':'Anime4KCPP OpenCL runtime files are missing'}
    return {'available':True,'reason':None}


def anime4kcpp_service_catalog(root):
    models=[];scales={};metadata={};revisions={};availability={}
    for model_id,native_model in ANIME4KCPP_MODELS.items():
        state=anime4kcpp_worker_availability(root)
        availability[model_id]=state
        scales[model_id]=2
        metadata[model_id]={
            'id':model_id,'name':f'Anime4KCPP {native_model}','family':'Anime4KCPP',
            'architecture':native_model,'native_scale':2,'status':'experimental-opencl',
            'legacy':False,'anime_specific':True,'provider':'Anime4KCPP OpenCL',
            'code_license':'MIT','source_url':'https://github.com/TianZerL/Anime4KCPP',
            'weights_path':'','capture_graph':False,'capture_graph_max_pixels':0,
        }
        if state['available']:
            models.append(model_id)
            revisions[model_id]='anime4kcpp-3.2.0-opencl-gfx1012'
    return {'models':models,'scales':scales,'metadata':metadata,'revisions':revisions,
            'modelAvailability':availability}


class Renderer:
    def __init__(self):
        self.backend = os.environ.get('ANIMESCALER_BACKEND', 'rocm')
        if self.backend not in ('rocm', 'tensorrt', 'opencl'):
            raise RuntimeError('Backend nativo desconhecido')
        self.models={};self.current_model=None;self.shape=None;self.graph=None
        if self.backend=='opencl':
            self.torch=None;self.registry=None;self.memory_format=None
            self.autotune_realesr=False;self.use_graph=False
            self.model=None;self.external_model=False
            self.info={'protocol':1,'backend':'opencl','model':None,'precision':'native',
                       'gpu':'OpenCL','architecture':'unknown','hip':None,
                       **anime4kcpp_service_catalog(ROOT)}
            return
        configure_miopen_cache(ROOT)
        if self.backend == 'rocm' and sys.platform == 'win32':
            require_rocm_runtime(output_dir=ROOT / 'diagnostics/runtime-preflight')
        import torch
        if not torch.cuda.is_available() or (self.backend == 'rocm' and not torch.version.hip) or (self.backend == 'tensorrt' and torch.version.hip):
            raise RuntimeError('ROCm GPU indisponível neste Python; não será usado CPU.')
        self.torch = torch
        # Small CPU work only; avoid competing with video decoding.
        torch.set_num_threads(2)
        # MIOpen autotuning is opt-in because it can increase startup time and
        # memory on RDNA1. It is enabled only for the profiled RealESR path.
        self.autotune_realesr = os.environ.get('ANIMESCALER_MIOPEN_BENCHMARK', '0') == '1'
        torch.backends.cudnn.benchmark = False
        from model_registry import build_registry
        self.registry=build_registry(ROOT)
        self.memory_format = torch.channels_last if os.environ.get('ANIMESCALER_CHANNELS_LAST', '0') == '1' else torch.contiguous_format
        self.use_graph=self.backend=='rocm' and os.environ.get('ANIMESCALER_HIP_GRAPH','1')=='1'
        props=torch.cuda.get_device_properties(0)
        registry_backend='torch' if self.backend=='rocm' else 'tensorrt'
        self.info={'protocol':1,'backend':self.backend,'model':'adore','precision':'fp16',
                   'gpu':props.name,'architecture':getattr(props,'gcnArchName',str(torch.cuda.get_device_capability())),
                   'hip':torch.version.hip,**registry_service_catalog(self.registry,registry_backend,ROOT)}
        for key,value in anime4kcpp_service_catalog(ROOT).items():
            if key in ('models','scales','metadata','revisions'):
                if key=='models':self.info[key].extend(value)
                else:self.info[key].update(value)
            else:self.info[key].update(value)
        try:
            self.select_model('adore')
        except BaseException:
            self.close()
            raise

    def select_model(self,name):
        if name not in self.info['models']:raise ValueError('Modelo local indisponível')
        def release(model_id,model):
            if model_id in ANIME4KCPP_MODELS: model.close()
            else: self.registry.unload(model_id)
        if name==self.current_model:
            release_inactive_models(self.models,name,release)
            return
        self.graph=None;self.shape=None;self.model=None;self.current_model=None
        release_inactive_models(self.models,name,release)
        if name in ANIME4KCPP_MODELS:
            if name not in self.models:
                self.models[name] = Anime4KCPPWorker(ROOT, name)
            self.graph = None; self.shape = None; self.model = self.models[name]
            self.current_model = name; self.external_model = True
            return
        self.torch.backends.cudnn.benchmark = self.autotune_realesr and name == 'realesr-anime'
        if name not in self.models:
            self.models[name]=self.registry.load(name,backend='torch' if self.backend=='rocm' else 'tensorrt',
                                                device='cuda',precision='fp16',memory_format=self.memory_format,root=ROOT)
        self.graph=None;self.shape=None;self.model=self.models[name];self.current_model=name;self.external_model=False

    def missing_profiles(self,name,shapes):
        if self.backend!='tensorrt':return list(shapes)
        self.select_model(name)
        backend_model=getattr(getattr(self.model,'backend',None),'model',None)
        missing=getattr(backend_model,'missing_cached_profiles',None)
        if not missing:return list(shapes)
        nchw=[(1,3,height,width) for width,height in shapes]
        return [(shape[3],shape[2]) for shape in missing(nchw)]

    def close(self):
        self.graph=None;self.shape=None;self.model=None;self.current_model=None;self.external_model=False
        failures=[]
        for model_id,model in list(self.models.items()):
            try:
                if model_id in ANIME4KCPP_MODELS: model.close()
                else: self.registry.unload(model_id)
            except Exception as error:
                failures.append(error)
            finally:
                self.models.pop(model_id,None)
        if failures: raise RuntimeError('Failed to close one or more native models') from failures[0]

    def compute(self):
        torch = self.torch
        height, width, oh, ow = self.shape
        value = self.gpu_input[..., :3].permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float16).contiguous(memory_format=self.memory_format)
        value.div_(255)
        result = (self.model.infer_to(value,(oh,ow)) if hasattr(self.model,'infer_to') else self.model(value))
        if (oh, ow) != result.shape[-2:]:
            result = torch.nn.functional.interpolate(result, size=(oh, ow), mode='bilinear', align_corners=False)
        self.gpu_output[..., :3].copy_(result[0].permute(1, 2, 0).clamp_(0, 1).mul_(255).round_())

    def can_capture(self,height,width):
        metadata=getattr(self.model,'metadata',None)
        return self.use_graph and (metadata is None or metadata.capture_graph
            or getattr(metadata,'capture_graph_max_pixels',0)>=height*width)

    def capture_graph(self):
        torch = self.torch
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.compute()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            self.compute()
        self.graph = graph

    def encode_output(self,frame_id,ow,oh,transport):
        if transport=='jpeg':
            from PIL import Image
            encoded=io.BytesIO()
            Image.fromarray(self.host_output.numpy(),'RGBA').convert('RGB').save(encoded,format='JPEG',quality=98,subsampling=0)
            return HEADER.pack(0x32435341,frame_id,ow,oh,ow,oh)+encoded.getvalue()
        magic=0x33435341 if transport=='rgb' else MAGIC
        return HEADER.pack(magic,frame_id,ow,oh,ow,oh)+self.host_output.numpy().tobytes()

    def render(self, data, model_name='adore', transport='rgba', allow_repeat=False):
        torch = self.torch
        self.select_model(model_name)
        if getattr(self,'last_transport',None)!=transport:
            self.graph=None;self.shape=None;self.last_transport=transport
        frame_id, width, height, ow, oh = decode_frame(data)
        started = time.perf_counter()
        shape = (height, width, oh, ow)
        self.last_warmup = shape != self.shape
        incoming=np.frombuffer(data,dtype=np.uint8,offset=HEADER.size).reshape(height,width,4)
        # Compare every byte, including small subtitles. No approximate thumbnail
        # match may suppress inference. Model/transport switches reset shape.
        previous_input = self.host_input.numpy() if hasattr(getattr(self, 'host_input', None), 'numpy') else getattr(self, 'host_input', None)
        self.last_reused=(not self.last_warmup and getattr(self,'reusable_frame',False)
                          and previous_input is not None and np.array_equal(incoming,previous_input))
        if self.last_reused:
            body=(HEADER.pack(0x34435341,frame_id,ow,oh,ow,oh) if allow_repeat
                  else self.encode_output(frame_id,ow,oh,transport))
            return body,(time.perf_counter()-started)*1000
        if getattr(self, 'external_model', False):
            self.last_warmup = shape != getattr(self, 'external_shape', None)
            result, duration = self.model.infer(incoming, frame_id, (ow, oh))
            self.host_input = incoming.copy()
            self.external_shape = shape
            self.shape = shape
            if transport == 'jpeg':
                from PIL import Image
                encoded = io.BytesIO()
                Image.fromarray(result[..., :3], 'RGB').save(encoded, format='JPEG', quality=98, subsampling=0)
                body = HEADER.pack(0x32435341, frame_id, ow, oh, ow, oh) + encoded.getvalue()
            elif transport == 'rgb':
                body = HEADER.pack(0x33435341, frame_id, ow, oh, ow, oh) + result[..., :3].tobytes()
            else:
                body = HEADER.pack(MAGIC, frame_id, ow, oh, ow, oh) + result.tobytes()
            self.frame_revision = getattr(self, 'frame_revision', 0) + 1
            self.reusable_frame = True
            return body, duration
        # An interrupted computation must never leave reusable stale output.
        self.reusable_frame=False
        with torch.inference_mode():
            if shape != self.shape:
                if self.backend == 'tensorrt':
                    self.model.prepare((1, 3, height, width))
                # One size at a time: no unbounded cache when adaptive changes size.
                self.graph = None
                self.host_input = torch.empty((height, width, 4), dtype=torch.uint8, pin_memory=True)
                self.gpu_input = torch.empty_like(self.host_input, device='cuda')
                self.host_output = torch.empty((oh, ow, 3 if transport=='rgb' else 4), dtype=torch.uint8, pin_memory=True)
                self.gpu_output = torch.empty_like(self.host_output, device='cuda')
                if transport!='rgb': self.gpu_output[..., 3].fill_(255)
                self.shape = shape
            np.copyto(self.host_input.numpy(),incoming)
            self.gpu_input.copy_(self.host_input, non_blocking=True)
            # These larger networks retain too many intermediate tensors in a
            # captured graph at HD sizes. Eager execution can recycle buffers.
            capture_allowed=self.can_capture(height,width)
            if capture_allowed and self.graph is None:
                try:
                    self.capture_graph()
                except RuntimeError as error:
                    self.use_graph = False
                    print(f'HIP graph indisponível; usando execução normal: {type(error).__name__}', flush=True)
            if self.graph is not None:
                self.graph.replay()
            else:
                self.compute()
            self.host_output.copy_(self.gpu_output, non_blocking=True)
            torch.cuda.synchronize()
            body=self.encode_output(frame_id,ow,oh,transport)
        self.frame_revision=getattr(self,'frame_revision',0)+1
        self.reusable_frame=True
        return body, (time.perf_counter() - started) * 1000


def allowed_origin(origin):
    return origin is None or bool(re.fullmatch(r'chrome-extension://[a-p]{32}', origin)) or origin == 'http://127.0.0.1:8765'


def parse_profile_check(payload,selected_model,available_models):
    if not isinstance(payload,str) or len(payload)>=256:raise ValueError('Invalid TensorRT profile check')
    control=json.loads(payload)
    if (not isinstance(control,dict) or set(control)!={'type','model','shapes'}
            or control['type']!='profile-check' or control['model']!=selected_model
            or selected_model not in available_models or not isinstance(control['shapes'],list)
            or not 1<=len(control['shapes'])<=8):
        raise ValueError('Invalid TensorRT profile check')
    shapes=[]
    for item in control['shapes']:
        if not isinstance(item,list) or len(item)!=2 or type(item[0]) is not int or type(item[1]) is not int:
            raise ValueError('Invalid TensorRT profile shape')
        width,height=item
        if min(width,height)<40 or max(width,height)>1920 or width%4 or height%4 or width*height>MAX_PIXELS:
            raise ValueError('Invalid TensorRT profile shape')
        shape=(width,height)
        if shape in shapes:raise ValueError('Duplicate TensorRT profile shape')
        shapes.append(shape)
    return shapes


async def tensorrt_profile_result(renderer,executor,payload,selected_model):
    if getattr(renderer,'backend',None)!='tensorrt':raise ValueError('TensorRT profile check unavailable')
    shapes=parse_profile_check(payload,selected_model,renderer.info.get('models',[]))
    missing=await asyncio.get_running_loop().run_in_executor(
        executor,renderer.missing_profiles,selected_model,shapes)
    return {'profileCheck':True,'supported':True,'available':not missing,'model':selected_model,
            'shapes':[list(shape) for shape in shapes],'missingShapes':[list(shape) for shape in missing]}


def make_app(renderer, executor, token, port=8766):
    busy = False
    active_stream = None
    fsr3 = getattr(renderer, 'backend', None) == 'd3d12'

    @web.middleware
    async def guard(request, handler):
        origin = request.headers.get('Origin')
        if not allowed_origin(origin) or request.host != f'127.0.0.1:{port}':
            return web.Response(status=403)
        headers = {'Cache-Control': 'no-store'}
        if origin:
            headers.update({'Access-Control-Allow-Origin': origin, 'Vary': 'Origin',
                            'Access-Control-Allow-Headers': 'Authorization, Content-Type',
                            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                            'Access-Control-Allow-Private-Network': 'true',
                            'Access-Control-Expose-Headers': 'X-Processing-Ms'})
        if request.method == 'OPTIONS':
            return web.Response(status=204, headers=headers)
        if request.path != '/stream' and not secrets.compare_digest(request.headers.get('Authorization', ''), 'Bearer ' + token):
            return web.Response(status=401, headers=headers)
        response = await handler(request)
        response.headers.update(headers)
        return response

    async def health(request):
        return web.json_response(renderer.info)

    async def frame(request):
        nonlocal busy
        if busy or (fsr3 and active_stream is not None):
            return web.Response(status=429, text='GPU ocupada; descarte este quadro.')
        busy = True
        try:
            try:
                data = await request.read()
            except ConnectionResetError:
                return web.Response(status=499)
            decode_frame(data)
            body, duration = await asyncio.get_running_loop().run_in_executor(executor, renderer.render, data)
            return web.Response(body=body, content_type='application/octet-stream', headers={'X-Processing-Ms': f'{duration:.3f}'})
        except ValueError as error:
            return web.json_response({'errorCode':native_error_code(error)},status=400)
        except web.HTTPException as error:
            return error
        except Exception as error:
            print(safe_native_error_log(getattr(renderer,'backend',None),error),flush=True)
            return web.json_response({'errorCode':native_error_code(error)},status=500)
        finally:
            busy = False

    async def stream(request):
        nonlocal busy, active_stream
        # aiohttp rejects lengths >= this transport limit. Keep the largest
        # valid 1080p input admissible; decode_frame still enforces exact size.
        socket = web.WebSocketResponse(max_msg_size=HEADER.size + MAX_PIXELS * 4 + 1, compress=False, heartbeat=20)
        await socket.prepare(request)
        try:
            auth = await socket.receive(timeout=5)
            if auth.type != WSMsgType.TEXT or not secrets.compare_digest(auth.data, token):
                await socket.close(code=1008, message=b'Authentication required')
                return socket
            model_name=request.query.get('model','adore')
            transport=request.query.get('transport','rgba')
            if transport not in ('rgba','jpeg','rgb') or (fsr3 and transport != 'rgba'):
                await socket.close(code=1008,message=b'Transport unavailable')
                return socket
            if model_name not in renderer.info.get('models',['adore']):
                await socket.close(code=1008,message=b'Model unavailable')
                return socket
            if fsr3:
                # A temporal context belongs to exactly one stream. Interleaving
                # viewers would contaminate history even with serialized GPU work.
                if active_stream is not None or busy:
                    await socket.close(code=1013,message=b'FSR3 viewer already active')
                    return socket
                active_stream = socket
            await socket.send_json({**renderer.info,'model':model_name,'transport':transport})
            delivered_revision=None
            repeat_enabled=request.query.get('repeat')=='1' and transport!='jpeg'
            if getattr(renderer,'backend',None) == 'd3d12' and hasattr(renderer, 'reset_history'):
                renderer.reset_history()
            async for message in socket:
                if message.type == WSMsgType.TEXT and getattr(renderer,'backend',None) == 'tensorrt':
                    if busy:
                        await socket.send_json({'errorCode':'NATIVE_SESSION_BUSY'})
                        continue
                    busy=True
                    try:
                        result=await tensorrt_profile_result(renderer,executor,message.data,model_name)
                        await socket.send_json(result)
                    except ConnectionError:
                        break
                    except ValueError:
                        await socket.send_json({'errorCode':'BACKEND_INVALID_RESPONSE'})
                        await socket.close(code=1008,message=b'Invalid TensorRT profile check')
                        break
                    except Exception as error:
                        print(safe_native_error_log(getattr(renderer,'backend',None),error),flush=True)
                        await socket.send_json({'errorCode':'BACKEND_UNAVAILABLE'})
                        await socket.close(code=1011,message=b'Profile check failed')
                        break
                    finally:
                        busy=False
                    continue
                if message.type == WSMsgType.TEXT and getattr(renderer,'backend',None) == 'd3d12':
                    try:
                        control=json.loads(message.data) if len(message.data)<256 else None
                        delta=control['frameDeltaMs']
                        if set(control)!={'reset','frameDeltaMs'} or type(control['reset']) is not bool or type(delta) not in (int,float) or not math.isfinite(delta) or not 0<delta<=250:
                            raise ValueError('Invalid frame context')
                        renderer.frame_context(control['reset'],delta)
                    except (ValueError,TypeError,KeyError,AttributeError):
                        await socket.close(code=1008, message=b'Invalid FSR3 control message')
                        break
                    continue
                if message.type != WSMsgType.BINARY:
                    break
                if busy:
                    await socket.send_json({'errorCode':'NATIVE_SESSION_BUSY','error':'GPU ocupada.'})
                    continue
                busy = True
                try:
                    decode_frame(message.data)
                    arguments = (message.data,) if model_name == 'adore' else (message.data,model_name)
                    if transport!='rgba': arguments=(message.data,model_name,transport)
                    if repeat_enabled:
                        allow_repeat=delivered_revision is not None and delivered_revision==getattr(renderer,'frame_revision',None)
                        arguments=(message.data,model_name,transport,allow_repeat)
                    body, duration = await asyncio.get_running_loop().run_in_executor(executor, renderer.render, *arguments)
                    # Timing is a small separate message; pixels stay binary end-to-end.
                    await socket.send_json({'processingMs': duration, 'warmup': getattr(renderer, 'last_warmup', False),
                                            'reused':getattr(renderer,'last_reused',False)})
                    await socket.send_bytes(body)
                    delivered_revision=getattr(renderer,'frame_revision',None)
                except ValueError as error:
                    busy = False
                    await socket.send_json({'errorCode':native_error_code(error)})
                    await socket.close(code=1008, message=b'Invalid frame')
                    break
                except ConnectionError:
                    # Closing the viewer while inference finishes is normal.
                    break
                except Exception as error:
                    busy = False
                    print(safe_native_error_log(getattr(renderer,'backend',None),error),flush=True)
                    try:await socket.send_json({'errorCode':native_error_code(error)})
                    except ConnectionError:pass
                    await socket.close(code=1011, message=native_error_code(error).encode('ascii'))
                    break
                finally:
                    busy = False
        except (TimeoutError, ConnectionError):
            pass
        finally:
            if active_stream is socket:
                active_stream = None
            await socket.close()
        return socket

    app = web.Application(middlewares=[guard], client_max_size=HEADER.size + MAX_PIXELS * 4 + 1)
    app.router.add_get('/health', health)
    app.router.add_post('/frame', frame)
    app.router.add_get('/stream', stream)
    app.router.add_route('OPTIONS', '/{path:.*}', health)
    return app


def make_app_runner(app):
    return web.AppRunner(app,access_log=None,handler_cancellation=False)


async def main():
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='rocm')
    renderer=None;runner=None;runner_ready=False
    loop=asyncio.get_running_loop()
    try:
        renderer = await loop.run_in_executor(executor, Renderer)
        connection_path = ROOT / 'native/connection.json'
        token = new_connection_token()
        app = make_app(renderer, executor, token)
        runner = make_app_runner(app)
        await runner.setup();runner_ready=True
        await web.TCPSite(runner, '127.0.0.1', 8766).start()
        publish_connection_file(connection_path,'http://127.0.0.1:8766',token)
        print(f"AnimeScaler {renderer.backend} pronto: {renderer.info['gpu']} · FP16. Mantenha esta janela aberta.", flush=True)
        await asyncio.Event().wait()
    finally:
        try:
            if runner_ready: await runner.cleanup()
        finally:
            try:
                if renderer is not None: await loop.run_in_executor(executor,renderer.close)
            finally:
                executor.shutdown(wait=True)


if __name__ == '__main__':
    asyncio.run(main())
