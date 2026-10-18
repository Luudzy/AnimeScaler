import { findCaptureTargetElement } from './capture-target.js';

// Never reparent an iframe: moving it can reload the player and lose playback.
export class ExclusiveSource {
  constructor(root = document) { this.document = root; this.saved = []; this.sessionId = null; }
  override(element, properties) {
    if (!element?.style) return;
    for (const [property, value] of Object.entries(properties)) {
      const entry={element, property, value:element.style.getPropertyValue(property), priority:element.style.getPropertyPriority(property)};
      element.style.setProperty(property, value, 'important');
      // CSSOM normalizes shorthands (notably animation and transition).
      // Compare against the value actually installed when restoring later.
      entry.applied=element.style.getPropertyValue(property);
      this.saved.push(entry);
    }
  }
  enter(sessionId, captureFrame) {
    if (this.sessionId===sessionId && this.target?.isConnected) return {ok:true};
    this.exit();
    const target = findCaptureTargetElement(this.document, captureFrame);
    if (!target) return {ok:false, message:'Player não encontrado para tela cheia exclusiva.'};
    this.sessionId = sessionId;
    this.target = target;
    let ancestor = target.parentElement ?? target.getRootNode?.().host;
    while (ancestor) {
      this.override(ancestor, {transform:'none', translate:'none',rotate:'none',scale:'none','will-change':'auto',animation:'none',transition:'none', filter:'none', perspective:'none', contain:'none',
        overflow:'visible', 'clip-path':'none', clip:'auto', 'content-visibility':'visible', opacity:'1', visibility:'visible', isolation:'auto', 'z-index':'auto'});
      ancestor = ancestor.parentElement ?? ancestor.getRootNode?.().host;
    }
    this.override(this.document.documentElement, {overflow:'hidden', background:'black'});
    this.override(target, {position:'fixed', inset:'0px', width:'100vw', height:'100vh',
      'min-width':'0px','min-height':'0px','max-width':'none','max-height':'none',margin:'0px',padding:'0px',border:'0px',
      transform:'none',translate:'none',rotate:'none',scale:'none','will-change':'auto',animation:'none',transition:'none','object-fit':'contain','object-position':'50% 50%', 'aspect-ratio':'auto',
      'clip-path':'none',clip:'auto','border-radius':'0px','z-index':'2147483647',background:'black',visibility:'visible',opacity:'1'});
    return {ok:true};
  }
  exit(sessionId) {
    if (sessionId && this.sessionId !== sessionId) return {ok:true};
    for (const entry of this.saved.reverse()) {
      // Keep changes the site made after our override instead of erasing them.
      if (entry.element.style.getPropertyValue(entry.property) !== entry.applied) continue;
      if (entry.value) entry.element.style.setProperty(entry.property,entry.value,entry.priority);
      else entry.element.style.removeProperty(entry.property);
    }
    this.saved = []; this.sessionId = null; this.target = null;
    return {ok:true};
  }
}
