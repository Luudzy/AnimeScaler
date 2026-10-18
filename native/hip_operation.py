"""Bounded clean-process HIP operation retries; complete results are never merged."""
import subprocess
from pathlib import Path
from artifact_io import atomic_json_write


def is_hip_infra_failure(stderr):
    # ROCm PyTorch uses the CUDA API/error prefix even on AMD.
    return 'unspecified launch failure' in stderr.lower()


def save_json(path,value):
    atomic_json_write(value,path)


def run_operation(base,command_factory,validate_result,start_attempt=1,timeout=600):
    base=Path(base)
    if start_attempt not in (1,2,3):raise ValueError('Original plus at most two retries')
    base.mkdir(parents=True,exist_ok=True)
    status=base/'operation.json'
    if status.exists():raise FileExistsError('Preserve previous operation')
    report={'state':'RUNNING','scope':'operation only; no model invalidation','attempts':[],
            'selectedAttempt':None,'result':None,'releaseApproved':False}
    save_json(status,report)
    for attempt in range(start_attempt,4):
        out=base/f'attempt-{attempt}'
        out.mkdir()
        try:command=list(map(str,command_factory(attempt,out)))
        except Exception as error:
            report['attempts'].append({'attempt':attempt,'pid':None,'output':str(out),
                'state':'FAIL','reason':'Preflight criterion: '+str(error),'exitCode':None})
            report.update(state='FAIL',reason='operation preflight/integrity criterion failed')
            save_json(status,report);return report
        stdout=out/'stdout.log';stderr=out/'stderr.log'
        with stdout.open('wb') as standard_out,stderr.open('wb') as standard_error:
            try:process=subprocess.Popen(command,stdout=standard_out,stderr=standard_error)
            except OSError as error:
                report['attempts'].append({'attempt':attempt,'pid':None,'command':command,'output':str(out),
                    'state':'INCONCLUSIVE','reason':str(error),'exitCode':None})
                report.update(state='NOT_MEASURED',reason='process launch infrastructure failure')
                save_json(status,report);return report
            entry={'attempt':attempt,'pid':process.pid,'command':command,'output':str(out),
                   'stdout':str(stdout),'stderr':str(stderr),'state':'RUNNING'}
            report['attempts'].append(entry);save_json(status,report)
            try:
                entry['exitCode']=process.wait(timeout=timeout)
            except BaseException as error:
                process.terminate()
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                entry.update(exitCode=process.returncode,state='INCONCLUSIVE',reason=type(error).__name__)
                report.update(state='NOT_MEASURED',reason='operation interrupted or timed out')
                save_json(status,report)
                if not isinstance(error,subprocess.TimeoutExpired):raise
                return report
        # Child has fully exited before any retry. No HIP context lives in this module.
        text=stderr.read_text(encoding='utf-8',errors='replace')
        if entry['exitCode']!=0:
            if is_hip_infra_failure(text):
                entry['state']='HIP_INFRA_FAILURE'
                save_json(status,report)
                continue
            entry['state']='INCONCLUSIVE'
            report.update(state='NOT_MEASURED',reason='non-HIP process failure; cause not established')
            save_json(status,report);return report
        try:
            result=validate_result(out)
        except Exception as error:
            entry.update(state='FAIL',reason=str(error))
            report.update(state='FAIL',reason='completed process returned invalid operation result')
            save_json(status,report);return report
        entry['state']='PASS'
        report.update(state='PASS',selectedAttempt=attempt,result=result)
        save_json(status,report);return report
    report.update(state='NOT_MEASURED',reason='recurrent HIP infrastructure failure')
    save_json(status,report)
    return report
