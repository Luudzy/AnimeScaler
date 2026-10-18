// Conservative stationary-region temporal denoise. Movement falls back to the
// current spatial frame; this is not optical flow or FSR 3 reconstruction.
export const ASR_TEMPORAL_WGSL = /* wgsl */ `
@vertex fn vertexMain(@builtin(vertex_index) index:u32)->@builtin(position) vec4f {
  let points=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.));
  return vec4f(points[index],0.,1.);
}
@group(0) @binding(0) var current:texture_2d<f32>;
@group(0) @binding(1) var rawCurrent:texture_2d<f32>;
@group(0) @binding(2) var rawPrevious:texture_2d<f32>;
@group(0) @binding(3) var history:texture_2d<f32>;
@group(0) @binding(4) var<uniform> parameters:vec4f;
fn at(image:texture_2d<f32>,p:vec2i,size:vec2i)->vec3f {
  return textureLoad(image,clamp(p,vec2i(0),size-vec2i(1)),0).rgb;
}
fn maximum(c:vec3f)->f32 { return max(c.x,max(c.y,c.z)); }
@fragment fn temporalMain(@builtin(position) position:vec4f)->@location(0) vec4f {
  let size=vec2i(textureDimensions(current)); let p=vec2i(position.xy);
  let center=at(current,p,size); let raw=at(rawCurrent,p,size);
  let oldRaw=at(rawPrevious,p,size);
  var guide=raw; var oldGuide=oldRaw;
  var lower=center; var upper=center; var gradientChange=0.;
  let offsets=array<vec2i,4>(vec2i(-1,0),vec2i(1,0),vec2i(0,-1),vec2i(0,1));
  for(var i:u32=0u;i<4u;i++) {
    let near=at(current,p+offsets[i],size);
    lower=min(lower,near); upper=max(upper,near);
    let a=at(rawCurrent,p+offsets[i],size);
    let b=at(rawPrevious,p+offsets[i],size);
    guide+=a; oldGuide+=b;
    gradientChange=max(gradientChange,maximum(abs((a-raw)-(b-oldRaw))));
  }
  let patchDifference=maximum(abs(guide-oldGuide))*.2;
  let pixelDifference=maximum(abs(raw-oldRaw));
  let confidence=(1.-smoothstep(.008,.040,patchDifference))
    *(1.-smoothstep(.035,.10,pixelDifference))
    *(1.-smoothstep(.05,.14,gradientChange));
  // Bound history to today's neighbourhood to limit contrasting old contours
  // in newly exposed background or subtitle regions.
  let old=clamp(at(history,p,size),lower-vec3f(.006),upper+vec3f(.006));
  return vec4f(mix(center,old,parameters.x*confidence),1.);
}`;

export class AsrTemporalFilter {
  constructor(device) { this.device=device; this.valid=false; this.index=0; }
  reset() { this.valid=false; }
  async configure(source,filtered,width,height) {
    this.destroyResources(); this.width=width; this.height=height;
    const usage=globalThis.GPUTextureUsage ?? {TEXTURE_BINDING:2,RENDER_ATTACHMENT:4,COPY_DST:1};
    this.history=[0,1].map(index=>this.device.createTexture({label:`ASR history ${index}`,size:[width,height],format:'rgba16float',usage:usage.TEXTURE_BINDING|usage.RENDER_ATTACHMENT}));
    this.rawPrevious=this.device.createTexture({label:'ASR previous source',size:[width,height],format:'rgba8unorm',usage:usage.TEXTURE_BINDING|usage.COPY_DST});
    const buffers=globalThis.GPUBufferUsage ?? {UNIFORM:1,COPY_DST:2};
    this.parameters=this.device.createBuffer({size:16,usage:buffers.UNIFORM|buffers.COPY_DST});
    if(!this.pipeline) {
      const module=this.device.createShaderModule({code:ASR_TEMPORAL_WGSL});
      this.pipeline=await this.device.createRenderPipelineAsync({layout:'auto',vertex:{module,entryPoint:'vertexMain'},fragment:{module,entryPoint:'temporalMain',targets:[{format:'rgba16float'}]},primitive:{topology:'triangle-list'}});
    }
    const bind=current=>this.history.map((_,index)=>this.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[
      {binding:0,resource:current.createView()},{binding:1,resource:source.createView()},
      {binding:2,resource:this.rawPrevious.createView()},{binding:3,resource:this.history[1-index].createView()},
      {binding:4,resource:{buffer:this.parameters}},
    ]}));
    this.rawBind=bind(source); this.filteredBind=bind(filtered); this.source=source;
    this.index=0; this.reset();
  }
  render(encoder,denoised) {
    const index=this.index, historyAvailable=this.valid;
    this.device.queue.writeBuffer(this.parameters,0,new Float32Array([historyAvailable ? .45 : 0,0,0,0]));
    const pass=encoder.beginRenderPass({label:'ASR temporal denoise',colorAttachments:[{view:this.history[index].createView(),loadOp:'clear',storeOp:'store',clearValue:{r:0,g:0,b:0,a:1}}]});
    pass.setPipeline(this.pipeline); pass.setBindGroup(0,(denoised?this.filteredBind:this.rawBind)[index]); pass.draw(3); pass.end();
    encoder.copyTextureToTexture({texture:this.source},{texture:this.rawPrevious},[this.width,this.height]);
    this.valid=true; this.index=1-index;
    return {index,historyAvailable};
  }
  destroyResources() { this.history?.forEach(texture=>texture.destroy()); this.rawPrevious?.destroy(); this.parameters?.destroy(); this.history=null; this.reset(); }
  destroy() { this.destroyResources(); }
}
