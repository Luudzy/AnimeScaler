"""Backend lifecycle and explicit completion timing; no silent provider fallback."""
from abc import ABC,abstractmethod

class InferenceBackend(ABC):
    @abstractmethod
    def load(self,source):pass
    @abstractmethod
    def infer(self,value):pass
    @abstractmethod
    def synchronize(self):pass
    @abstractmethod
    def close(self):pass
    def warmup(self,value):
        result=self.infer(value);self.synchronize();return result

class TorchBackend(InferenceBackend):
    def __init__(self,device='cpu',precision='fp32',memory_format=None):
        if precision not in ('fp32','fp16'):raise ValueError('Unsupported precision')
        self.device=device;self.precision=precision;self.model=None
        self.memory_format=memory_format
    def load(self,factory):
        import torch
        self.close()
        self.model=factory().eval().to(device=self.device,dtype=torch.float16 if self.precision=='fp16' else torch.float32)
        if self.memory_format is not None:self.model=self.model.to(memory_format=self.memory_format)
        return self
    def infer(self,value):
        import torch
        if self.model is None:raise RuntimeError('Backend is not loaded')
        with torch.inference_mode():return self.model(value)
    def infer_to(self,value,output_size):
        import torch
        if self.model is None:raise RuntimeError('Backend is not loaded')
        with torch.inference_mode():
            if hasattr(self.model,'forward_to'):return self.model.forward_to(value,output_size)
            result=self.model(value)
            return result if tuple(result.shape[-2:])==tuple(output_size) else torch.nn.functional.interpolate(
                result,size=output_size,mode='bilinear',align_corners=False)
    def synchronize(self):
        if str(self.device).startswith('cuda'):
            import torch
            torch.cuda.synchronize(self.device)
    def close(self):self.model=None

class TensorRTBackend(InferenceBackend):
    def load(self,source):
        import torch
        from tensorrt_model import TensorRTModel
        root,name,path,scale=source;self.model=TensorRTModel(root,name,torch,path=path,scale=scale);return self
    def infer(self,value):return self.model(value)
    def prepare(self,shape):self.model.prepare(shape)
    def synchronize(self):
        import torch
        torch.cuda.synchronize()
    def close(self):self.model=None

class OnnxBackend(InferenceBackend):
    def __init__(self,provider='CPUExecutionProvider'):
        self.provider=provider;self.session=None;self.last_result=None
    def load(self,path):
        import onnxruntime as ort
        if self.provider not in ort.get_available_providers():raise RuntimeError(f'Provider unavailable: {self.provider}')
        self.close();options=ort.SessionOptions()
        if self.provider=='DmlExecutionProvider':
            options.enable_mem_pattern=False;options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        self.session=ort.InferenceSession(str(path),sess_options=options,providers=[self.provider])
        return self
    def infer(self,value):
        if self.session is None:raise RuntimeError('Backend is not loaded')
        # session.run returns CPU arrays: completion includes readback, including DML.
        self.last_result=self.session.run(None,{self.session.get_inputs()[0].name:value})[0]
        return self.last_result
    def synchronize(self):pass  # run() has already completed its CPU readback.
    def close(self):self.session=None;self.last_result=None
