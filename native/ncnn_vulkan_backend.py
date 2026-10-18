"""Persistent NCNN Vulkan inference for the official RealESRGAN anime model."""
from pathlib import Path
import numpy as np


class NcnnVulkanBackend:
    def __init__(self, model_dir, model_name='realesr-animevideov3-x2', gpu_index=0):
        import ncnn
        self._ncnn = ncnn
        self.model_dir = Path(model_dir)
        self.model_name = model_name
        self.vulkan_available = ncnn.get_gpu_count() > gpu_index
        if not self.vulkan_available:
            raise RuntimeError('No Vulkan GPU is available to NCNN')
        self.net = ncnn.Net()
        self.net.set_vulkan_device(gpu_index)
        self.net.opt.use_vulkan_compute = True
        self.net.opt.use_fp16_packed = True
        self.net.opt.use_fp16_storage = True
        self.net.opt.use_fp16_arithmetic = True
        param = self.model_dir / f'{model_name}.param'
        binary = self.model_dir / f'{model_name}.bin'
        if not param.is_file() or not binary.is_file():
            raise FileNotFoundError(f'NCNN model files not found for {model_name}')
        if self.net.load_param(str(param)) != 0 or self.net.load_model(str(binary)) != 0:
            raise RuntimeError(f'Could not load NCNN model {model_name}')

    def infer_bytes(self, data, width, height, scale=2):
        if len(data) != width * height * 3:
            raise ValueError('RGB input byte length does not match dimensions')
        pixels = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
        image = self._ncnn.Mat.from_pixels(pixels, self._ncnn.Mat.PixelType.PIXEL_RGB, width, height)
        image.substract_mean_normalize([], [1 / 255.0] * 3)
        ex = self.net.create_extractor()
        ex.set_light_mode(True)
        if ex.input('data', image) != 0:
            raise RuntimeError('NCNN input failed')
        code, output = ex.extract('output')
        if code != 0:
            raise RuntimeError(f'NCNN output failed: {code}')
        array = np.array(output.numpy(format='f'), copy=True)
        ex.clear()
        if array.ndim != 3 or array.shape[0] != 3:
            raise RuntimeError(f'Unexpected NCNN output shape: {array.shape}')
        return np.transpose(array, (1, 2, 0))

    def close(self):
        if getattr(self, 'net', None) is not None:
            self.net.clear()
            self.net = None
