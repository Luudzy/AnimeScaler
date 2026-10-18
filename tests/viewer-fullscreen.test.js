import test from 'node:test';
import assert from 'node:assert/strict';
import {ViewerFullscreen} from '../viewer/fullscreen.js';

test('fullscreen returns to the previous normal window bounds',async()=>{
  let current={id:7,state:'normal',left:40,top:50,width:1000,height:700};
  const windows={async get(){return {...current};},async update(id,change){assert.equal(id,7);current={...current,...change};return current;}};
  const control=new ViewerFullscreen(windows,7);
  await control.toggle();assert.equal(current.state,'fullscreen');
  current={...current,left:0,top:0,width:1920,height:1080};
  await control.exit();
  assert.deepEqual(current,{id:7,state:'normal',left:40,top:50,width:1000,height:700});
});

test('fullscreen restores maximized state and ignores repeated exit',async()=>{
  let current={id:8,state:'maximized'},calls=0;
  const windows={async get(){return {...current};},async update(id,change){calls++;current={...current,...change};}};
  const control=new ViewerFullscreen(windows,8);
  await control.toggle();await control.exit();await control.exit();
  assert.equal(current.state,'maximized');assert.equal(calls,2);
});
