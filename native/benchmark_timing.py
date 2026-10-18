"""Timing utilities requiring an explicit backend completion barrier."""
import time
import numpy as np

def fill_distinct_input(value,index,seed=929):
    import torch
    generator=torch.Generator(device=value.device).manual_seed(seed+index)
    value.copy_(torch.rand(value.shape,dtype=value.dtype,device=value.device,generator=generator))

def timed_iterations(operation,synchronize,warmup=50,iterations=500,prepare=lambda i:None):
    if warmup<0 or iterations<1:raise ValueError('Invalid iteration count')
    for i in range(warmup):prepare(i);operation();synchronize()
    submit=[];complete=[]
    for i in range(iterations):
        prepare(i+warmup);synchronize()
        start=time.perf_counter();operation();submitted=time.perf_counter();synchronize();finished=time.perf_counter()
        submit.append((submitted-start)*1000);complete.append((finished-start)*1000)
    return {'warmup':warmup,'iterations':iterations,'cpuSubmitMeanMs':float(np.mean(submit)),
            'meanMs':float(np.mean(complete)),'medianMs':float(np.median(complete)),'p50Ms':float(np.percentile(complete,50)),
            'p95Ms':float(np.percentile(complete,95)),'p99Ms':float(np.percentile(complete,99)),
            'minMs':min(complete),'maxMs':max(complete),'fps':1000/float(np.mean(complete))}
