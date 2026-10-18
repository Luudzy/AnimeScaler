// Backward block matching: a destination pixel reads reference(x+dx,y+dy).
// Bounded thumbnail search, not interpolation or a claim of optical-flow accuracy.
export function motionReuse(reference,current,width,height) {
  const length=width*height*4;
  if(!reference || !current || reference.length!==length || current.length!==length) return null;
  const tiles=[], valid=new Uint8Array(width*height), mask=new Uint8ClampedArray(length);
  const block=8, radius=4;
  // All searched displacements lie on a quarter-pixel grid in [-4.75,4.75].
  // One lazy, bounded scratch buffer per call; scores never cross tiles.
  const scoreSide=39,scoreOffset=19;
  let scores;
  function read(x,y,c) {
    const xx=Math.floor(x), yy=Math.floor(y), fx=x-xx,fy=y-yy;
    if(fx===0&&fy===0)return reference[(yy*width+xx)*4+c];
    const x1=Math.min(width-1,xx+1), y1=Math.min(height-1,yy+1);
    return (reference[(yy*width+xx)*4+c]*(1-fx)+reference[(yy*width+x1)*4+c]*fx)*(1-fy)
      +(reference[(y1*width+xx)*4+c]*(1-fx)+reference[(y1*width+x1)*4+c]*fx)*fy;
  }
  for(let y=0;y<height;y+=block)for(let x=0;x<width;x+=block) {
    const w=Math.min(block,width-x),h=Math.min(block,height-y);
    const calculateScore=(dx,dy)=>{
      if(x+dx<0||y+dy<0||x+w-1+dx>=width||y+h-1+dy>=height) return Infinity;
      let error=0,n=0;
      if(Number.isInteger(dx)&&Number.isInteger(dy)) {
        for(let yy=y;yy<y+h;yy+=2)for(let xx=x;xx<x+w;xx+=2) {
          const a=(yy*width+xx)*4,b=((yy+dy)*width+xx+dx)*4;
          error+=Math.abs(current[a]-reference[b])+Math.abs(current[a+1]-reference[b+1])+Math.abs(current[a+2]-reference[b+2]);n+=3;
        }
        return error/n;
      }
      for(let yy=y;yy<y+h;yy+=2)for(let xx=x;xx<x+w;xx+=2)for(let c=0;c<3;c++) {
        error+=Math.abs(current[(yy*width+xx)*4+c]-read(xx+dx,yy+dy,c));n++;
      }
      return error/n;
    };
    const score=(dx,dy)=>{
      const key=(dy*4+scoreOffset)*scoreSide+dx*4+scoreOffset;
      const cached=scores[key];
      if(!Number.isNaN(cached))return cached;
      return scores[key]=calculateScore(dx,dy);
    };
    const stationary=calculateScore(0,0);
    let best=stationary, second=Infinity,dx=0,dy=0;
    if(stationary>2) {
      scores??=new Float64Array(scoreSide*scoreSide);
      scores.fill(NaN);
      scores[scoreOffset*scoreSide+scoreOffset]=stationary;
      const candidates=[{dx:0,dy:0,error:stationary}];
      for(let sy=-radius;sy<=radius;sy++)for(let sx=-radius;sx<=radius;sx++) {
        if(!sx&&!sy)continue;
        const error=score(sx,sy);
        if(candidates.length<4||error<candidates.at(-1).error){
          candidates.push({dx:sx,dy:sy,error});candidates.sort((a,b)=>a.error-b.error);if(candidates.length>4)candidates.pop();
        }
        if(error<best){second=best;best=error;dx=sx;dy=sy;}
        else second=Math.min(second,error);
      }
      // A blended/subpixel edge may not have the best integer score.
      if(best>32)continue;
      const refined=[];
      for(const candidate of candidates) {
        let rx=candidate.dx,ry=candidate.dy,local=candidate.error;
        for(const step of [.5,.25]) {
          // A single pass can stop on the slope of a neighboring seed's
          // minimum. Allow one bounded follow-up without expanding the
          // original maximum displacement (radius + .5 + .25).
          for(let pass=0;pass<2;pass++) {
            const cx=rx,cy=ry;
            for(let sy=-1;sy<=1;sy++)for(let sx=-1;sx<=1;sx++) {
              const nx=cx+sx*step,ny=cy+sy*step;
              if(Math.abs(nx)>radius+.75||Math.abs(ny)>radius+.75)continue;
              const error=score(nx,ny);
              if(error<local){local=error;rx=nx;ry=ny;}
            }
            if(rx===cx&&ry===cy)break;
          }
        }
        refined.push({dx:rx,dy:ry,error:local});
        if(local<best){best=local;dx=rx;dy=ry;}
      }
      // Also compare refined scores. Multiple integer seeds may converge to
      // one displacement; only distinct refined solutions compete.
      // Inspect every seed even after a perfect match: periodic subpixel
      // patterns can have another equally good direction elsewhere.
      for(const candidate of refined)if(candidate.dx!==dx||candidate.dy!==dy)
        second=Math.min(second,candidate.error);
      if((dx||dy)&&second-best<1)continue;
    }
    if(best>=12)continue;
    tiles.push({x,y,width:w,height:h,dx,dy});
    // Validate every pixel after the sparse search, including new occlusions.
    for(let yy=y;yy<y+h;yy++)for(let xx=x;xx<x+w;xx++) {
      let error=0;
      for(let c=0;c<3;c++)error=Math.max(error,Math.abs(current[(yy*width+xx)*4+c]-read(xx+dx,yy+dy,c)));
      // Continuous confidence avoids full-image flashing around the old 6/8
      // code-value thresholds. Strong mismatches still have exactly zero weight.
      valid[yy*width+xx]=Math.round(255*Math.max(0,Math.min(1,(12-error)/10)));
    }
  }
  for(let y=0;y<height;y++)for(let x=0;x<width;x++) {
    let ok=valid[y*width+x];
    for(let dy=-1;dy<=1&&ok;dy++)for(let dx=-1;dx<=1;dx++) {
      const xx=x+dx,yy=y+dy;
      if(xx>=0&&xx<width&&yy>=0&&yy<height)ok=Math.min(ok,valid[yy*width+xx]);
    }
    const i=(y*width+x)*4;mask[i]=mask[i+1]=mask[i+2]=255;mask[i+3]=ok;
  }
  return {tiles,mask};
}
