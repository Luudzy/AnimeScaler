function boundedWait(promise,ms,label) {
  let timer;
  return Promise.race([promise,new Promise((_,reject)=>{
    timer=setTimeout(()=>reject(new Error(label+' timed out')),ms);
  })]).finally(()=>clearTimeout(timer));
}

async function cleanupOwnedProbe({service,serviceClosed,browser,server,logClosers=[],restoreConnection,record,timeoutMs=5000}) {
  const report={date:new Date().toISOString(),ownedNativeExited:!service,browserClosed:!browser,
    serverClosed:!server||!server.listening,connectionRestored:false,errors:[]};
  const attempt=async(step,operation)=>{
    try {await operation();}
    catch(error){report.errors.push({step,message:error.message});}
  };
  await attempt('native',async()=>{
    if(!service)return;
    if(service.exitCode===null&&service.signalCode===null)service.kill();
    await boundedWait(serviceClosed,timeoutMs,'Owned native shutdown');
    report.ownedNativeExited=service.exitCode!==null||service.signalCode!==null;
    if(!report.ownedNativeExited)throw new Error('Native termination unconfirmed');
  });
  await attempt('browser',async()=>{
    if(browser)await boundedWait(browser.close(),timeoutMs,'Owned browser shutdown');
    report.browserClosed=true;
  });
  await attempt('server',async()=>{
    if(server?.listening)await boundedWait(new Promise((resolve,reject)=>{
      server.close(error=>error?reject(error):resolve());server.closeAllConnections();
    }),timeoutMs,'Owned diagnostic shutdown');
    report.serverClosed=true;
  });
  for(let i=0;i<logClosers.length;i++)await attempt(`log-${i}`,logClosers[i]);
  // Restoration is reached even if every previous resource operation fails.
  await attempt('connection',async()=>{
    report.connectionRestored=await restoreConnection();
    if(!report.connectionRestored)throw new Error('Connection restoration unconfirmed');
  });
  if(record)await attempt('report',()=>record(report));
  if(report.errors.length)throw new AggregateError(report.errors.map(row=>new Error(row.message)),
    'Probe cleanup failed: '+report.errors.map(row=>row.step).join(', '));
  return report;
}

module.exports={boundedWait,cleanupOwnedProbe};
