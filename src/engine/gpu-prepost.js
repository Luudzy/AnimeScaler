import { nchwFloatBytes, outputShape2x } from './gpu-utils.js';
import { PREPROCESS_WGSL, EXTERNAL_PREPROCESS_WGSL, POSTPROCESS_WGSL, RENDER_WGSL } from './shaders.js';
import { importVideoTexture, videoCropUniform } from './external-video.js';
import { validateWorkgroup, workgroupShader } from './workgroup.js';

export class GpuPrePost {
  constructor({ device, canvas, workgroup }) {
    if (!device) throw new Error('WebGPU device is required');
    if (!canvas) throw new Error('canvas is required');
    this.device = device;
    this.workgroup = validateWorkgroup(workgroup, device.limits);
    this.canvas = canvas;
    this.context = canvas.getContext('webgpu');
    if (!this.context) throw new Error('WebGPU canvas context is unavailable');
    this.resources = [];
    this.config = null;
  }

  async configure({ sourceWidth, sourceHeight, inputWidth, inputHeight, displayWidth = inputWidth * 2, displayHeight = inputHeight * 2 }) {
    this.destroyResources();
    const outputWidth = inputWidth * 2;
    const outputHeight = inputHeight * 2;
    this.config = { sourceWidth, sourceHeight, inputWidth, inputHeight, outputWidth, outputHeight };

    const format = navigator.gpu.getPreferredCanvasFormat();
    this.canvas.width = displayWidth;
    this.canvas.height = displayHeight;
    this.context.configure({ device: this.device, format, alphaMode: 'opaque' });

    this.sourceTexture=null;this.preBindGroup=null;this.externalPreBindGroup=null;

    this.outputTexture = this.track(this.device.createTexture({
      label: 'AnimeScaler model output texture',
      size: [outputWidth, outputHeight, 1],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    }));

    this.inputBuffer = this.track(this.device.createBuffer({
      label: 'AnimeScaler model input NCHW',
      size: nchwFloatBytes(inputWidth, inputHeight, 16),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    }));

    this.outputBuffer = this.track(this.device.createBuffer({
      label: 'AnimeScaler model output NCHW',
      size: nchwFloatBytes(outputWidth, outputHeight, 16),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    }));

    this.preDimsBuffer = this.createDimsBuffer(inputWidth, inputHeight, 'AnimeScaler preprocess dims');
    if (this.device.importExternalTexture) {
      this.cropBuffer=this.track(this.device.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}));
      if (!this.externalPrePipeline && !this.externalPreUnavailable) {
        try {
          this.externalPrePipeline=await this.device.createComputePipelineAsync({
            label:'AnimeScaler direct video -> NCHW',layout:'auto',
            compute:{module:this.device.createShaderModule({code:workgroupShader(EXTERNAL_PREPROCESS_WGSL,this.workgroup)}),entryPoint:'main'},
          });
        } catch {this.externalPreUnavailable=true;}
      }
    }
    this.postDimsBuffer = this.createDimsBuffer(outputWidth, outputHeight, 'AnimeScaler postprocess dims');
    this.sampler ??= this.device.createSampler({ magFilter: 'linear', minFilter: 'linear' });

    this.prePipeline ??= await this.device.createComputePipelineAsync({
      label: 'AnimeScaler RGB -> NCHW',
      layout: 'auto',
      compute: { module: this.device.createShaderModule({ code: workgroupShader(PREPROCESS_WGSL, this.workgroup) }), entryPoint: 'main' },
    });
    this.postPipeline ??= await this.device.createComputePipelineAsync({
      label: 'AnimeScaler NCHW -> RGB',
      layout: 'auto',
      compute: { module: this.device.createShaderModule({ code: workgroupShader(POSTPROCESS_WGSL, this.workgroup) }), entryPoint: 'main' },
    });
    this.renderPipeline ??= await this.device.createRenderPipelineAsync({
      label: 'AnimeScaler canvas renderer',
      layout: 'auto',
      vertex: { module: this.device.createShaderModule({ code: RENDER_WGSL }), entryPoint: 'vertexMain' },
      fragment: {
        module: this.device.createShaderModule({ code: RENDER_WGSL }),
        entryPoint: 'fragmentMain',
        targets: [{ format }],
      },
      primitive: { topology: 'triangle-list' },
    });

    this.postBindGroup = this.device.createBindGroup({
      layout: this.postPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.outputBuffer } },
        { binding: 1, resource: this.outputTexture.createView() },
        { binding: 2, resource: { buffer: this.postDimsBuffer } },
      ],
    });
    this.renderBindGroup = this.device.createBindGroup({
      layout: this.renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.outputTexture.createView() },
        { binding: 1, resource: this.sampler },
      ],
    });
  }

  track(resource) {
    this.resources.push(resource);
    return resource;
  }

  createDimsBuffer(width, height, label) {
    const buffer = this.track(this.device.createBuffer({
      label,
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    }));
    this.device.queue.writeBuffer(buffer, 0, new Uint32Array([width, height, 0, 0]));
    return buffer;
  }

  uploadVideoFrame(video, sourceRect = null) {
    if (!this.config) throw new Error('GpuPrePost is not configured');
    this.externalPreBindGroup=null;
    const external=this.externalPrePipeline?importVideoTexture(this.device,video):null;
    if (external) {
      this.device.queue.writeBuffer(this.cropBuffer,0,videoCropUniform(video,sourceRect,this.config.sourceWidth,this.config.sourceHeight));
      this.externalPreBindGroup=this.device.createBindGroup({
        layout:this.externalPrePipeline.getBindGroupLayout(0),entries:[
          {binding:0,resource:external},{binding:1,resource:this.sampler},
          {binding:2,resource:{buffer:this.inputBuffer}},{binding:3,resource:{buffer:this.preDimsBuffer}},
          {binding:4,resource:{buffer:this.cropBuffer}},
        ],
      });
      this.captureMethod='external-texture';return;
    }
    this.sourceTexture??=this.track(this.device.createTexture({
      label:'AnimeScaler fallback source texture',size:[this.config.sourceWidth,this.config.sourceHeight,1],
      format:'rgba8unorm',usage:GPUTextureUsage.COPY_DST|GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.RENDER_ATTACHMENT,
    }));
    this.preBindGroup??=this.device.createBindGroup({layout:this.prePipeline.getBindGroupLayout(0),entries:[
      {binding:0,resource:this.sourceTexture.createView()},{binding:1,resource:this.sampler},
      {binding:2,resource:{buffer:this.inputBuffer}},{binding:3,resource:{buffer:this.preDimsBuffer}},
    ]});
    this.captureMethod='texture-copy';
    const source = { source: video };
    if (sourceRect) source.origin = { x: sourceRect.x, y: sourceRect.y };
    this.device.queue.copyExternalImageToTexture(
      source,
      { texture: this.sourceTexture, colorSpace: 'srgb' },
      [this.config.sourceWidth, this.config.sourceHeight],
    );
  }

  preprocess() {
    const { inputWidth, inputHeight } = this.config;
    const encoder = this.device.createCommandEncoder({ label: 'AnimeScaler preprocess encoder' });
    const pass = encoder.beginComputePass({ label: 'AnimeScaler preprocess pass' });
    pass.setPipeline(this.externalPreBindGroup?this.externalPrePipeline:this.prePipeline);
    pass.setBindGroup(0, this.externalPreBindGroup??this.preBindGroup);
    pass.dispatchWorkgroups(Math.ceil(inputWidth / this.workgroup[0]), Math.ceil(inputHeight / this.workgroup[1]));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    this.externalPreBindGroup=null;
  }

  render() {
    const { outputWidth, outputHeight } = this.config;
    const encoder = this.device.createCommandEncoder({ label: 'AnimeScaler output encoder' });
    const compute = encoder.beginComputePass({ label: 'AnimeScaler output conversion' });
    compute.setPipeline(this.postPipeline);
    compute.setBindGroup(0, this.postBindGroup);
    compute.dispatchWorkgroups(Math.ceil(outputWidth / this.workgroup[0]), Math.ceil(outputHeight / this.workgroup[1]));
    compute.end();

    const render = encoder.beginRenderPass({
      label: 'AnimeScaler canvas render',
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });
    render.setPipeline(this.renderPipeline);
    render.setBindGroup(0, this.renderBindGroup);
    render.draw(3);
    render.end();
    this.device.queue.submit([encoder.finish()]);
  }

  getInputBuffer() {
    return this.inputBuffer;
  }

  getOutputBuffer() {
    return this.outputBuffer;
  }

  getInputShape() {
    const { inputWidth, inputHeight } = this.config;
    return [1, 3, inputHeight, inputWidth];
  }

  getOutputShape() {
    const { inputWidth, inputHeight } = this.config;
    return outputShape2x(inputWidth, inputHeight);
  }

  destroyResources() {
    this.externalPreBindGroup=null;
    for (const resource of this.resources.splice(0)) {
      try { resource.destroy?.(); } catch { /* ignore stale GPU resources */ }
    }
  }

  destroy() {
    this.destroyResources();
    this.context.unconfigure?.();
    this.config = null;
  }
}
