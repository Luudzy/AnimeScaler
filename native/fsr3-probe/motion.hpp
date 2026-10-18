// Estimated 2D video motion, not engine-provided game motion/depth.
inline constexpr char MotionShader[] = R"HLSL(
Texture2D<float4> Current : register(t0);
Texture2D<float4> Previous : register(t1);
RWTexture2D<float2> Motion : register(u0);
RWTexture2D<float> Reactive : register(u1);
cbuffer Dimensions : register(b0) { uint Width; uint Height; uint History; uint Padding; };
float3 at(Texture2D<float4> image,int2 p) { return image.Load(int3(clamp(p,int2(0,0),int2(Width-1,Height-1)),0)).rgb; }
float difference(float3 a,float3 b) { float3 d=abs(a-b);return max(d.r,max(d.g,d.b)); }
float errorAt(int2 origin,int2 delta) {
  float total=0;
  int2 points[5]={int2(2,2),int2(5,2),int2(2,5),int2(5,5),int2(4,4)};
  [unroll] for(uint i=0;i<5;i++)total+=difference(at(Current,origin+points[i]),at(Previous,origin+points[i]+delta));
  return total*.2;
}
[numthreads(8,8,1)] void main(uint3 id:SV_DispatchThreadID) {
  int2 origin=int2(id.xy)*8;if(origin.x>=Width||origin.y>=Height)return;
  int2 bestDelta=int2(0,0);float bestError=0;
  if(History) {
    bestError=errorAt(origin,bestDelta);float bestCost=bestError;
    // Coarse grid + local refinement. Large/unmatched motion rejects history.
    [loop] for(int dy=-6;dy<=6;dy+=2) [loop] for(int dx=-6;dx<=6;dx+=2) {
      int2 delta=int2(dx,dy);float e=errorAt(origin,delta);float cost=e+.00015*(abs(dx)+abs(dy));
      if(cost<bestCost){bestCost=cost;bestError=e;bestDelta=delta;}
    }
    int2 coarse=bestDelta;
    [unroll] for(int dy=-1;dy<=1;dy++) [unroll] for(int dx=-1;dx<=1;dx++) {
      int2 delta=coarse+int2(dx,dy);float e=errorAt(origin,delta);float cost=e+.00015*(abs(delta.x)+abs(delta.y));
      if(cost<bestCost){bestCost=cost;bestError=e;bestDelta=delta;}
    }
  }
  [loop] for(int y=0;y<8;y++) [loop] for(int x=0;x<8;x++) {
    int2 p=origin+int2(x,y);if(p.x>=Width||p.y>=Height)continue;
    float reject=1;
    if(History) {
      float localError=difference(at(Current,p),at(Previous,p+bestDelta));
      reject=max(smoothstep(.012,.065,bestError),smoothstep(.025,.10,localError));
      // Neighbour gradients detect details not represented by one block vector.
      float3 a=at(Current,p),b=at(Previous,p+bestDelta);
      float edgeError=max(difference(at(Current,p+int2(1,0))-a,at(Previous,p+bestDelta+int2(1,0))-b),difference(at(Current,p+int2(0,1))-a,at(Previous,p+bestDelta+int2(0,1))-b));
      reject=max(reject,smoothstep(.04,.12,edgeError));
    }
    Motion[p]=History ? float2(bestDelta)/float2(Width,Height) : float2(0,0);
    Reactive[p]=reject;
  }
}
)HLSL";
