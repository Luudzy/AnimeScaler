import {importVideoTexture,videoCropUniform} from './external-video.js';

const SHADER=`
struct Params { crop: vec4f, mode: vec4f, bounds: vec4f }
@group(0) @binding(0) var current: texture_2d<f32>;
@group(0) @binding(1) var ai: texture_2d<f32>;
@group(0) @binding(2) var confidence: texture_2d<f32>;
@group(0) @binding(3) var<storage,read> vectors: array<vec2f>;
@group(0) @binding(4) var linearSampler: sampler;
@group(0) @binding(5) var<uniform> params: Params;
struct Vertex { @builtin(position) position: vec4f, @location(0) uv: vec2f }
@vertex fn vertexMain(@builtin(vertex_index) i:u32)->Vertex {
  var positions=array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3));
  var out:Vertex;let p=positions[i];out.position=vec4f(p,0,1);out.uv=vec2f((p.x+1)*.5,(1-p.y)*.5);return out;
}
@fragment fn fragmentMain(in:Vertex)->@location(0) vec4f {
  if(params.mode.x==1){return vec4f(textureSampleLevel(ai,linearSampler,in.uv,0).rgb,1);}
  let base=textureSampleLevel(current,linearSampler,params.crop.xy+in.uv*params.crop.zw,0);
  if(params.mode.x==0){return vec4f(base.rgb,1);}
  let cell=min(vec2u(in.uv*vec2f(80,45)),vec2u(79,44));
  let tile=cell/8u;let delta=vectors[tile.y*10u+tile.x]/vec2f(80,45);
  let weight=textureSampleLevel(confidence,linearSampler,in.uv,0).a*params.mode.y;
  let enhanced=textureSampleLevel(ai,linearSampler,in.uv+delta,0);
  return vec4f(mix(base.rgb,enhanced.rgb,weight),1);
}`;
const EXTERNAL_SHADER=SHADER.replace('current: texture_2d<f32>','current: texture_external')
  .replace('textureSampleLevel(current,linearSampler,params.crop.xy+in.uv*params.crop.zw,0)',
    'textureSampleBaseClampToEdge(current,linearSampler,clamp(params.crop.xy+in.uv*params.crop.zw,params.bounds.xy,params.bounds.zw))');

export async function createNativeGpuRenderer(canvas) {
  if(!globalThis.navigator?.gpu)return null;
  let device,pipeline,rgbPipeline,externalPipeline;
  const format=navigator.gpu.getPreferredCanvasFormat();
  try {
    const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});
    if(!adapter)return null;
    device=await adapter.requestDevice();
    const module=device.createShaderModule({code:SHADER});
    pipeline=await device.createRenderPipelineAsync({layout:'auto',vertex:{module,entryPoint:'vertexMain'},
      fragment:{module,entryPoint:'fragmentMain',targets:[{format}]},primitive:{topology:'triangle-list'}});
    if(device.importExternalTexture){
      try {
        const externalModule=device.createShaderModule({code:EXTERNAL_SHADER});
        externalPipeline=await device.createRenderPipelineAsync({layout:'auto',vertex:{module:externalModule,entryPoint:'vertexMain'},
          fragment:{module:externalModule,entryPoint:'fragmentMain',targets:[{format}]},primitive:{topology:'triangle-list'}});
      } catch { /* Older devices keep the copied texture path. */ }
    }
    const unpack=device.createShaderModule({code:`
      @group(0) @binding(0) var<storage,read> packed:array<u32>;
      @group(0) @binding(1) var outputImage:texture_storage_2d<rgba8unorm,write>;
      fn byteAt(i:u32)->f32{return f32((packed[i/4u]>>((i%4u)*8u))&255u)/255.;}
      @compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id:vec3u){
        let size=textureDimensions(outputImage);let p=id.x;if(p>=size.x*size.y){return;}
        let i=p*3u;textureStore(outputImage,vec2u(p%size.x,p/size.x),vec4f(byteAt(i),byteAt(i+1u),byteAt(i+2u),1));
      }`});
    const errors=(await unpack.getCompilationInfo()).messages.filter(message=>message.type==='error');
    if(errors.length)throw new Error(errors.map(message=>message.message).join('; '));
    rgbPipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module:unpack,entryPoint:'main'}});
  } catch (error) {console.warn('Native GPU compositor unavailable:',error);device?.destroy();return null;}
  const context=canvas.getContext('webgpu');
  if(!context){device.destroy();return null;}
  context.configure({device,format,alphaMode:'opaque'});
  return new NativeGpuRenderer(canvas,context,device,pipeline,rgbPipeline,externalPipeline);
}

export class NativeGpuRenderer {
  constructor(canvas,context,device,pipeline,rgbPipeline,externalPipeline) {
    Object.assign(this,{canvas,context,device,pipeline,rgbPipeline,externalPipeline});
    this.sampler=device.createSampler({magFilter:'linear',minFilter:'linear'});
    this.mask=this.texture(80,45);
    this.vectors=device.createBuffer({size:10*6*8,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    this.params=device.createBuffer({size:48,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    this.vectorData=new Float32Array(120);this.paramData=new Float32Array([0,0,1,1,0,0,0,0,0,0,1,1]);
    this.current=this.texture(1,1);this.currentWidth=1;this.currentHeight=1;
    this.hasCurrent=false;this.hasAI=false;this.width=0;this.height=0;
    device.lost.then(()=>{this.lost=true;});
    device.addEventListener('uncapturederror',event=>{this.failure=event.error.message;});
  }
  texture(width,height) {
    return this.device.createTexture({size:[width,height],format:'rgba8unorm',
      usage:GPUTextureUsage.COPY_DST|GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.STORAGE_BINDING});
  }
  resize(width,height) {
    if(this.width===width&&this.height===height)return;
    this.width=width;this.height=height;this.canvas.width=width;this.canvas.height=height;
    this.ai?.destroy();this.ai=this.texture(width,height);this.hasAI=false;this.hasCurrent=false;this.bind=null;
    this.rgbBuffer?.destroy();this.rgbBuffer=null;this.rgbBind=null;this.pendingRgb=false;
    this.externalCurrent=null;
  }
  setFrame(frame,crop) {
    this.externalCurrent=this.externalPipeline?importVideoTexture(this.device,frame):null;
    if(this.externalCurrent){
      const values=videoCropUniform(frame,crop,crop.width,crop.height);
      this.paramData.set(values.subarray(0,4));this.paramData.set(values.subarray(4),8);
      this.hasCurrent=true;this.sourceMethod='external-texture';return;
    }
    const w=frame.displayWidth??frame.width,h=frame.displayHeight??frame.height;
    if(w!==this.currentWidth||h!==this.currentHeight){
      this.current.destroy();this.current=this.texture(w,h);this.currentWidth=w;this.currentHeight=h;this.bind=null;
    }
    this.device.queue.copyExternalImageToTexture({source:frame},{texture:this.current},[w,h]);
    this.paramData.set([crop.x/w,crop.y/h,crop.width/w,crop.height/h]);this.hasCurrent=true;this.sourceMethod='texture-copy';
  }
  uploadAI(image) {
    if(!image.rgb)this.pendingRgb=false;
    if(image.rgb) {
      this.rgbBuffer??=this.device.createBuffer({size:Math.ceil(image.rgb.byteLength/4)*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
      const bytes=image.rgb.byteLength%4?new Uint8Array(Math.ceil(image.rgb.byteLength/4)*4):image.rgb;
      if(bytes!==image.rgb)bytes.set(image.rgb);
      this.device.queue.writeBuffer(this.rgbBuffer,0,bytes);
      this.rgbBind??=this.device.createBindGroup({layout:this.rgbPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.rgbBuffer}},{binding:1,resource:this.ai.createView()}]});
      this.pendingRgb=true;
    } else if(image.data)this.device.queue.writeTexture({texture:this.ai},image.data,{bytesPerRow:image.width*4},[image.width,image.height]);
    else this.device.queue.copyExternalImageToTexture({source:image},{texture:this.ai},[image.width,image.height]);
    this.hasAI=true;
  }
  presentAI(image) {this.resize(image.width,image.height);this.uploadAI(image);this.draw(null,true);}
  draw(reuse=null,aiOnly=false,strength=1) {
    if(this.lost||this.failure)throw new Error(this.failure||'Compositor WebGPU desconectado.');
    if(!aiOnly&&!this.hasCurrent)return;
    this.paramData[4]=aiOnly?1:(reuse&&this.hasAI?2:0);
    this.paramData[5]=strength;
    if(this.paramData[4]===2) {
      this.vectorData.fill(0);
      for(const tile of reuse.tiles){const i=((tile.y/8)*10+tile.x/8)*2;this.vectorData[i]=tile.dx;this.vectorData[i+1]=tile.dy;}
      this.device.queue.writeBuffer(this.vectors,0,this.vectorData);
      this.device.queue.writeTexture({texture:this.mask},reuse.mask,{bytesPerRow:80*4},[80,45]);
    }
    this.device.queue.writeBuffer(this.params,0,this.paramData);
    const external=!aiOnly && this.externalCurrent;
    const pipeline=external?this.externalPipeline:this.pipeline;
    const makeBind=()=>this.device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[
      {binding:0,resource:external||this.current.createView()},{binding:1,resource:this.ai.createView()},
      {binding:2,resource:this.mask.createView()},{binding:3,resource:{buffer:this.vectors}},
      {binding:4,resource:this.sampler},{binding:5,resource:{buffer:this.params}},
    ]});
    const bind=external?makeBind():(this.bind??=makeBind());
    const encoder=this.device.createCommandEncoder();
    if(this.pendingRgb) {
      const unpack=encoder.beginComputePass();unpack.setPipeline(this.rgbPipeline);unpack.setBindGroup(0,this.rgbBind);
      unpack.dispatchWorkgroups(Math.ceil(this.width*this.height/256));unpack.end();this.pendingRgb=false;
    }
    const pass=encoder.beginRenderPass({colorAttachments:[{view:this.context.getCurrentTexture().createView(),
      loadOp:'clear',storeOp:'store',clearValue:{r:0,g:0,b:0,a:1}}]});
    pass.setPipeline(pipeline);pass.setBindGroup(0,bind);pass.draw(3);pass.end();
    this.device.queue.submit([encoder.finish()]);
    this.externalCurrent=null;
    if(external)this.hasCurrent=false; // A later draw needs a newly imported source.
  }
  destroy(){this.context.unconfigure();this.device.destroy();}
}
