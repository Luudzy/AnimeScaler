// AMD FidelityFX Super Resolution 1 spatial path, adapted for WebGPU.
// EASU (edge-adaptive spatial upscaling) and RCAS (contrast-adaptive
// sharpening) are derived from AMD's MIT-licensed FSR 1 reference.

const VERTEX = /* wgsl */ `
struct VertexOut { @builtin(position) position: vec4f, @location(0) uv: vec2f }
@vertex fn vertexMain(@builtin(vertex_index) index:u32)->VertexOut {
  var points=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.));
  let point=points[index];
  return VertexOut(vec4f(point,0.,1.),vec2f((point.x+1.)*.5,(1.-point.y)*.5));
}`;

export const FSR_EASU_WGSL = /* wgsl */ `
${VERTEX}
@group(0) @binding(0) var source:texture_2d<f32>;
struct Dimensions { input:vec2f, output:vec2f }
@group(0) @binding(2) var<uniform> dimensions:Dimensions;
struct Set { dir:vec2f, len:f32 }
struct Tap { color:vec3f, weight:f32 }
fn sampleAt(p:vec2i, size:vec2i)->vec3f { return textureLoad(source,clamp(p,vec2i(0),size-vec2i(1)),0).rgb; }
fn luma(c:vec3f)->f32 { return c.b*.5+(c.r*.5+c.g); }
fn min3(a:vec3f,b:vec3f,c:vec3f)->vec3f { return min(a,min(b,c)); }
fn max3(a:vec3f,b:vec3f,c:vec3f)->vec3f { return max(a,max(b,c)); }
fn addSet(value:Set, pp:vec2f, weight:f32, a:f32,b:f32,c:f32,d:f32,e:f32)->Set {
  let dx=d-b; let dy=e-a;
  let maxX=max(abs(d-c),abs(c-b)); let maxY=max(abs(e-c),abs(c-a));
  let normalizedX=select(0.,pow(clamp(abs(dx)/maxX,0.,1.),2.),maxX>1e-6);
  let normalizedY=select(0.,pow(clamp(abs(dy)/maxY,0.,1.),2.),maxY>1e-6);
  return Set(value.dir+vec2f(dx,dy)*weight,value.len+(normalizedX+normalizedY)*weight);
}
fn addTap(value:Tap, offset:vec2f, dir:vec2f, shape:vec2f, lobe:f32, clip:f32, color:vec3f)->Tap {
  var rotated=vec2f(dot(offset,dir),dot(offset,vec2f(-dir.y,dir.x)))*shape;
  let distance=min(dot(rotated,rotated),clip);
  let basis=2./5.*distance-1.; let window=lobe*distance-1.;
  let weight=(25./16.*basis*basis-(25./16.-1.))*window*window;
  return Tap(value.color+color*weight,value.weight+weight);
}
@fragment fn easuMain(@builtin(position) position:vec4f)->@location(0) vec4f {
  let sourceSize=vec2i(textureDimensions(source));
  let uv=position.xy/dimensions.output;
  let sourceCoord=uv*dimensions.input-vec2f(.5);
  let base=vec2i(floor(sourceCoord));
  let frac=fract(sourceCoord);
  let b=sampleAt(base+vec2i(0,-1),sourceSize); let c=sampleAt(base+vec2i(1,-1),sourceSize);
  let e=sampleAt(base+vec2i(-1,0),sourceSize); let f=sampleAt(base,sourceSize);
  let g=sampleAt(base+vec2i(1,0),sourceSize); let h=sampleAt(base+vec2i(2,0),sourceSize);
  let i=sampleAt(base+vec2i(-1,1),sourceSize); let j=sampleAt(base+vec2i(0,1),sourceSize);
  let k=sampleAt(base+vec2i(1,1),sourceSize); let l=sampleAt(base+vec2i(2,1),sourceSize);
  let n=sampleAt(base+vec2i(0,2),sourceSize); let o=sampleAt(base+vec2i(1,2),sourceSize);
  var flow=Set(vec2f(0.),0.);
  flow=addSet(flow,frac,(1.-frac.x)*(1.-frac.y),luma(b),luma(e),luma(f),luma(g),luma(j));
  flow=addSet(flow,frac,frac.x*(1.-frac.y),luma(c),luma(f),luma(g),luma(h),luma(k));
  flow=addSet(flow,frac,(1.-frac.x)*frac.y,luma(f),luma(i),luma(j),luma(k),luma(n));
  flow=addSet(flow,frac,frac.x*frac.y,luma(g),luma(j),luma(k),luma(l),luma(o));
  let directionLength=length(flow.dir);
  let direction=select(vec2f(1.,0.),flow.dir/directionLength,directionLength>1./32768.);
  let edge=pow(flow.len*.5,2.);
  let stretch=1./max(abs(direction.x),abs(direction.y));
  let shape=vec2f(1.+(stretch-1.)*edge,1.-.5*edge);
  let lobe=.5+((.25-.04)-.5)*edge;
  let clip=1./lobe;
  var accumulated=Tap(vec3f(0.),0.);
  accumulated=addTap(accumulated,vec2f(0.,-1.)-frac,direction,shape,lobe,clip,b);
  accumulated=addTap(accumulated,vec2f(1.,-1.)-frac,direction,shape,lobe,clip,c);
  accumulated=addTap(accumulated,vec2f(-1.,1.)-frac,direction,shape,lobe,clip,i);
  accumulated=addTap(accumulated,vec2f(0.,1.)-frac,direction,shape,lobe,clip,j);
  accumulated=addTap(accumulated,vec2f(0.,0.)-frac,direction,shape,lobe,clip,f);
  accumulated=addTap(accumulated,vec2f(-1.,0.)-frac,direction,shape,lobe,clip,e);
  accumulated=addTap(accumulated,vec2f(1.,1.)-frac,direction,shape,lobe,clip,k);
  accumulated=addTap(accumulated,vec2f(2.,1.)-frac,direction,shape,lobe,clip,l);
  accumulated=addTap(accumulated,vec2f(2.,0.)-frac,direction,shape,lobe,clip,h);
  accumulated=addTap(accumulated,vec2f(1.,0.)-frac,direction,shape,lobe,clip,g);
  accumulated=addTap(accumulated,vec2f(1.,2.)-frac,direction,shape,lobe,clip,o);
  accumulated=addTap(accumulated,vec2f(0.,2.)-frac,direction,shape,lobe,clip,n);
  let lower=min3(f,g,j); let upper=max3(f,g,j);
  let lower4=min(lower,k); let upper4=max(upper,k);
  let normalized=select(f,accumulated.color/accumulated.weight,abs(accumulated.weight)>1e-6);
  return vec4f(clamp(normalized,lower4,upper4),1.);
}`;

export const FSR_RCAS_WGSL = /* wgsl */ `
${VERTEX}
@group(0) @binding(0) var source:texture_2d<f32>;
fn loadAt(p:vec2i,size:vec2i)->vec3f { return textureLoad(source,clamp(p,vec2i(0),size-vec2i(1)),0).rgb; }
fn min3v(a:vec3f,b:vec3f,c:vec3f)->vec3f{return min(a,min(b,c));}
fn max3v(a:vec3f,b:vec3f,c:vec3f)->vec3f{return max(a,max(b,c));}
@fragment fn rcasMain(@builtin(position) position:vec4f)->@location(0) vec4f {
  let size=vec2i(textureDimensions(source)); let p=vec2i(position.xy);
  let north=loadAt(p+vec2i(0,-1),size); let west=loadAt(p+vec2i(-1,0),size);
  let center=loadAt(p,size); let east=loadAt(p+vec2i(1,0),size); let south=loadAt(p+vec2i(0,1),size);
  let min4=min3v(north,west,east); let max4=max3v(north,west,east);
  let ringMin=min(min4,south); let ringMax=max(max4,south);
  let localMin=min(ringMin,center); let localMax=max(ringMax,center);
  let hitMin=localMin/max(4.*localMax,vec3f(1e-5));
  let hitMax=(vec3f(1.)-localMax)/min(4.*localMin-vec3f(4.),vec3f(-1e-5));
  let lobe=clamp(max(-hitMin,hitMax),vec3f(-3./16.),vec3f(0.))*0.55;
  let denom=max(vec3f(1e-5),vec3f(1.)+4.*lobe);
  return vec4f(clamp((center+lobe*(north+west+east+south))/denom,vec3f(0.),vec3f(1.)),1.);
}`;

// Anime Spatial Refinement: a conservative refinement pass for cel-shaded
// material. It only smooths very flat areas, protects detected line contrast
// and clamps sharpening to the local colour range to avoid ringing/halos.
export const ASR_ANIME_WGSL = /* wgsl */ `
${VERTEX}
@group(0) @binding(0) var source:texture_2d<f32>;
fn asrLoad(p:vec2i,size:vec2i)->vec3f{return textureLoad(source,clamp(p,vec2i(0),size-vec2i(1)),0).rgb;}
fn asrLuma(c:vec3f)->f32{return dot(c,vec3f(.2126,.7152,.0722));}
@fragment fn rcasMain(@builtin(position) position:vec4f)->@location(0) vec4f {
  let size=vec2i(textureDimensions(source)); let p=vec2i(position.xy);
  let north=asrLoad(p+vec2i(0,-1),size); let west=asrLoad(p+vec2i(-1,0),size);
  let center=asrLoad(p,size); let east=asrLoad(p+vec2i(1,0),size); let south=asrLoad(p+vec2i(0,1),size);
  let localMin=min(center,min(min(north,south),min(west,east)));
  let localMax=max(center,max(max(north,south),max(west,east)));
  let lumaMin=min(asrLuma(center),min(min(asrLuma(north),asrLuma(south)),min(asrLuma(west),asrLuma(east))));
  let lumaMax=max(asrLuma(center),max(max(asrLuma(north),asrLuma(south)),max(asrLuma(west),asrLuma(east))));
  let flatness=1.-smoothstep(.018,.095,lumaMax-lumaMin);
  let crossAverage=(north+west+east+south)*.25;
  let denoised=mix(center,crossAverage,flatness*.18);
  let lineStrength=1.-flatness;
  let lobe=-.10*lineStrength;
  let sharpened=(denoised+lobe*(north+west+east+south))/max(vec3f(1e-5),vec3f(1.)+4.*lobe);
  return vec4f(clamp(sharpened,localMin,localMax),1.);
}`;

// Video prefilter: spatial bilateral smoothing at source resolution. No
// temporal history, motion vectors or CPU pixels; colour distance protects
// lineart/subtitles and limits smoothing across compression boundaries.
export const VIDEO_DENOISE_WGSL = /* wgsl */ `
${VERTEX}
@group(0) @binding(0) var source:texture_2d<f32>;
@group(0) @binding(2) var<uniform> parameters:vec4f;
@fragment fn denoiseMain(@builtin(position) position:vec4f)->@location(0) vec4f {
  let size=vec2i(textureDimensions(source)); let p=vec2i(position.xy);
  let center=textureLoad(source,p,0).rgb;
  var sum=vec3f(0.); var weights=0.;
  let inverseVariance=1./max(3.*parameters.x*parameters.x,1e-6);
  for(var y:i32=-2;y<=2;y++) {
    for(var x:i32=-2;x<=2;x++) {
      let offset=vec2i(x,y);
      let sampleColor=textureLoad(source,clamp(p+offset,vec2i(0),size-vec2i(1)),0).rgb;
      let delta=sampleColor-center;
      let weight=exp2(-dot(delta,delta)*inverseVariance-dot(vec2f(offset),vec2f(offset))*.35);
      sum+=sampleColor*weight; weights+=weight;
    }
  }
  return vec4f(mix(center,sum/max(weights,1e-6),parameters.y),1.);
}`;

export const FSR_INPUT_RESIZE_WGSL = /* wgsl */ `
${VERTEX}
@group(0) @binding(0) var source:texture_2d<f32>;
@group(0) @binding(1) var linearSampler:sampler;
struct Dimensions { input:vec2f, output:vec2f }
@group(0) @binding(2) var<uniform> dimensions:Dimensions;
@fragment fn resizeMain(@builtin(position) position:vec4f)->@location(0) vec4f {
  let uv=position.xy/dimensions.input;
  // Four bilinear samples approximate area reduction, avoiding point-like
  // downsampling on lineart when the source is much larger than the input.
  let offset=vec2f(.25)/dimensions.input;
  let a=textureSampleLevel(source,linearSampler,uv+vec2f(-offset.x,-offset.y),0.).rgb;
  let b=textureSampleLevel(source,linearSampler,uv+vec2f(offset.x,-offset.y),0.).rgb;
  let c=textureSampleLevel(source,linearSampler,uv+vec2f(-offset.x,offset.y),0.).rgb;
  let d=textureSampleLevel(source,linearSampler,uv+offset,0.).rgb;
  return vec4f((a+b+c+d)*.25,1.);
}`;
