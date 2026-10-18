"""Optional TensorRT 10.x runner; GPU buffers remain on the PyTorch CUDA stream."""
import hashlib
import json
import os


class TensorRTModel:
    def __init__(self, root, model, torch, path=None, scale=None):
        import tensorrt as trt
        if not trt.__version__.startswith('10.'):
            raise RuntimeError('Este backend requer TensorRT 10.x')
        self.trt, self.torch = trt, torch
        from user_models import USER_MODELS
        self.scale=scale or USER_MODELS.get(model,{}).get('scale',2)
        names = {'adore': '2x_Adore_renarchi_webgpu_fp16.onnx',
                 'animejanai': '2x_AnimeJaNai_HD_V3_SuperUltraCompact_webgpu_fp16.onnx',
                 'adore-lite': 'adore-lite-experimental.onnx', 'adore-lite-v3':'adore-lite-v3-experimental.onnx'}
        names.update({name:name+'.onnx' for name in ('hfa2k-cugan','realesr-anime','span-mssim')})
        self.path = path if path is not None else root / 'models' / names[model]
        if not self.path.is_file():
            raise RuntimeError(f'Exporte o modelo antes de iniciar TensorRT: {self.path.name}')
        self.cache = root / 'diagnostics/native-cache/tensorrt'
        self.cache.mkdir(parents=True, exist_ok=True)
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.shape = None

    def profile_cache_path(self, shape):
        return self.profile_cache_paths([shape])[0]

    def profile_cache_paths(self, shapes):
        base=[hashlib.sha256(self.path.read_bytes()).hexdigest(),self.trt.__version__,self.torch.version.cuda,
              self.torch.cuda.get_device_name(),self.torch.cuda.get_device_capability()]
        paths=[]
        for shape in shapes:
            key=hashlib.sha256(json.dumps([*base,shape]).encode()).hexdigest()
            paths.append(self.cache/(key+'.engine'))
        return paths

    def has_cached_profile(self, shape):
        return self.profile_cache_path(shape).is_file()

    def missing_cached_profiles(self,shapes):
        return [shape for shape,path in zip(shapes,self.profile_cache_paths(shapes)) if not path.is_file()]

    def prepare(self, shape):
        if self.shape == shape:
            return
        trt, torch = self.trt, self.torch
        path = self.profile_cache_path(shape)
        engine = None
        if path.exists():
            engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if engine is None:
            if os.environ.get('ANIMESCALER_TRT_BUILD') != '1':
                raise RuntimeError('Engine TensorRT ausente. Execute native/prepare_tensorrt.py antes de abrir o viewer.')
            builder = trt.Builder(self.logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, self.logger)
            if not parser.parse(self.path.read_bytes()):
                raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
            if network.num_inputs != 1 or network.num_outputs != 1:
                raise RuntimeError('Expected one image input/output')
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 512 * 1024 * 1024)
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            profile = builder.create_optimization_profile()
            profile.set_shape(network.get_input(0).name, shape, shape, shape)
            config.add_optimization_profile(profile)
            print(f'Compilando TensorRT {self.path.name} {shape}; aguarde antes de abrir o viewer.', flush=True)
            serialized = builder.build_serialized_network(network, config)
            if serialized is None:
                raise RuntimeError('TensorRT engine build failed')
            engine = self.runtime.deserialize_cuda_engine(serialized)
            if engine is None:
                raise RuntimeError('TensorRT engine load failed')
            temporary = path.with_suffix('.tmp')
            temporary.write_bytes(bytes(serialized))
            os.replace(temporary, path)
        if engine is None:
            raise RuntimeError('TensorRT engine load failed')
        self.engine = engine
        self.context = engine.create_execution_context()
        inputs = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
                  if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]
        outputs = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
                   if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError('Unexpected TensorRT I/O')
        self.input_name, self.output_name = inputs[0], outputs[0]
        if not self.context.set_input_shape(self.input_name, shape):
            raise RuntimeError('TensorRT rejected input shape')
        dtypes = {trt.float32: torch.float32, trt.float16: torch.float16}
        self.input = torch.empty(shape, device='cuda', dtype=dtypes[engine.get_tensor_dtype(self.input_name)])
        output_shape = tuple(self.context.get_tensor_shape(self.output_name))
        if output_shape != (shape[0], 3, shape[2]*self.scale, shape[3]*self.scale):
            raise RuntimeError(f'Unexpected model output: {output_shape}')
        self.output = torch.empty(output_shape, device='cuda', dtype=dtypes[engine.get_tensor_dtype(self.output_name)])
        for name, tensor in [(self.input_name, self.input), (self.output_name, self.output)]:
            if not self.context.set_tensor_address(name, tensor.data_ptr()):
                raise RuntimeError('TensorRT tensor binding failed')
        self.shape = shape

    def __call__(self, value):
        self.prepare(tuple(value.shape))
        self.input.copy_(value)
        if not self.context.execute_async_v3(self.torch.cuda.current_stream().cuda_stream):
            raise RuntimeError('TensorRT execution failed')
        return self.output
