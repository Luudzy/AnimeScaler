// CPU-only thumbnail diagnostics; residuals are not a measure of AI ghosting.
export function analyzeMotion(reference,current,width,height,reuse) {
  const n=width*height, raw=new Float64Array(n), errors=new Float64Array(n);
  errors.fill(NaN);
  const read=(x,y,c)=>{
    const x0=Math.floor(x),y0=Math.floor(y),fx=x-x0,fy=y-y0;
    const x1=Math.min(width-1,x0+1),y1=Math.min(height-1,y0+1);
    return (reference[(y0*width+x0)*4+c]*(1-fx)+reference[(y0*width+x1)*4+c]*fx)*(1-fy)
      +(reference[(y1*width+x0)*4+c]*(1-fx)+reference[(y1*width+x1)*4+c]*fx)*fy;
  };
  for(const t of reuse.tiles)for(let y=t.y;y<t.y+t.height;y++)for(let x=t.x;x<t.x+t.width;x++) {
    let error=0;
    for(let c=0;c<3;c++)error=Math.max(error,Math.abs(current[(y*width+x)*4+c]-read(x+t.dx,y+t.dy,c)));
    errors[y*width+x]=error;
    raw[y*width+x]=Math.round(255*Math.max(0,Math.min(1,(12-error)/10)))/255;
  }
  let opacity=0,before=0,weightedError=0,maxError=0,supportedPixels=0;
  for(let p=0;p<n;p++) {
    before+=raw[p];
    const alpha=reuse.mask[p*4+3]/255;opacity+=alpha;
    if(alpha>0) {
      if(!Number.isFinite(errors[p]))throw new Error('Mask supports a pixel with no matching tile');
      supportedPixels++;weightedError+=alpha*errors[p];maxError=Math.max(maxError,errors[p]);
    }
  }
  return {coverage:opacity/n,beforeErosionCoverage:before/n,erosionLoss:(before-opacity)/n,
    supportedPixels,supportedMaxChannelErrorMean:opacity?weightedError/opacity:null,
    supportedMaxChannelErrorMax:opacity?maxError:null,
    acceptedTiles:reuse.tiles.length,movedTiles:reuse.tiles.filter(t=>t.dx||t.dy).length};
}
