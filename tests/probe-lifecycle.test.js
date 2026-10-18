import test from 'node:test';
import assert from 'node:assert/strict';
import {cleanupOwnedProbe} from '../diagnostics/probe-lifecycle.cjs';

function resources() {
  const calls=[],reports=[];
  const service={exitCode:null,signalCode:null,kill(){calls.push('kill');this.signalCode='SIGTERM';return true;}};
  const server={listening:true,close(callback){calls.push('server');this.listening=false;callback();},closeAllConnections(){}};
  return {calls,reports,options:{service,serviceClosed:Promise.resolve(),server,
    browser:{async close(){calls.push('browser');}},
    logClosers:[()=>calls.push('stdout'),()=>calls.push('stderr')],
    restoreConnection(){calls.push('restore');return true;},record(report){reports.push(report);},timeoutMs:10}};
}

test('owned resources and credentials are released after a successful probe',async()=>{
  const {options,calls,reports}=resources();await cleanupOwnedProbe(options);
  assert.deepEqual(calls,['kill','browser','server','stdout','stderr','restore']);
  assert.equal(reports[0].ownedNativeExited,true);assert.equal(reports[0].connectionRestored,true);
  assert.deepEqual(reports[0].errors,[]);
});

test('unconfirmed native shutdown fails instead of silently authorizing success',async()=>{
  const {options,calls,reports}=resources();
  options.service.kill=()=>false;options.serviceClosed=new Promise(()=>{});
  await assert.rejects(cleanupOwnedProbe(options),/cleanup failed/i);
  assert.equal(reports[0].ownedNativeExited,false);
  assert.ok(calls.includes('restore'),'credentials still restored on shutdown failure');
});

test('browser close failure cannot skip server, descriptors or connection restoration',async()=>{
  const {options,calls,reports}=resources();options.browser.close=async()=>{throw new Error('browser failure');};
  await assert.rejects(cleanupOwnedProbe(options),/cleanup failed/i);
  assert.deepEqual(calls,['kill','server','stdout','stderr','restore']);
  assert.equal(reports[0].connectionRestored,true);
  assert.equal(reports[0].browserClosed,false);
});

test('descriptor failure does not skip other descriptors or connection restoration',async()=>{
  const {options,calls}=resources();options.logClosers[0]=()=>{throw new Error('descriptor failure');};
  await assert.rejects(cleanupOwnedProbe(options),/cleanup failed/i);
  assert.ok(calls.includes('stderr'));assert.ok(calls.includes('restore'));
});

test('failure to restore connection is recorded and returned as failure',async()=>{
  const {options,reports}=resources();options.restoreConnection=()=>false;
  await assert.rejects(cleanupOwnedProbe(options),/cleanup failed/i);
  assert.equal(reports[0].connectionRestored,false);
});
