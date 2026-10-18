export class ViewerFullscreen {
  constructor(windows,windowId,initial = {state:'normal'}) {
    this.windows=windows;this.windowId=windowId;this.previous=initial;this.busy=false;
  }
  async change(exitOnly = false) {
    if(this.busy)return;
    this.busy=true;
    try {
      const current=await this.windows.get(this.windowId);
      if(current.state==='fullscreen') {
        const state=this.previous.state==='maximized'?'maximized':'normal';
        await this.windows.update(this.windowId,{state});
        if(state==='normal') {
          const bounds={};
          for(const key of ['left','top','width','height']) if(Number.isFinite(this.previous[key]))bounds[key]=this.previous[key];
          if(Object.keys(bounds).length)await this.windows.update(this.windowId,bounds);
        }
      } else if(!exitOnly) {
        this.previous=current;
        await this.windows.update(this.windowId,{state:'fullscreen'});
      }
    } finally {this.busy=false;}
  }
  toggle(){return this.change();}
  exit(){return this.change(true);}
}
