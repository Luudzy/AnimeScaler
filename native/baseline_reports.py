"""Preserve screening results only across matching performance reruns."""
import json


def performance_identity(report):
    rows=[]
    for row in report.get('cases',[]):
        artifact=row.get('artifactSha256')
        # Older Torch reports already hash the actual loaded checkpoint.
        if artifact is None and report.get('backend')=='torch':artifact=row.get('sha256')
        if not artifact:return None
        rows.append([row.get(k) for k in ('model','nativeScale','width','height','outputWidth','outputHeight')]+[artifact])
    if not rows:return None
    return (report.get('backend'),report.get('provider'),report.get('precision'),
            tuple(sorted(json.dumps(row) for row in rows)))


def merge_baseline_report(previous,performance):
    result={'performance':performance,'quality':'PENDING same-input visual/perceptual baseline comparison',
            'realVideoValidation':'NOT EXECUTED','zenkaiTrainingApproved':False}
    quality=previous.get('quality')
    identity=performance_identity(performance)
    compatible=identity is not None and identity==performance_identity(previous.get('performance',{}))
    # Quality currently executes Torch on ROCm. Matching source weights alone
    # cannot certify an ONNX export or a different precision/provider.
    if isinstance(quality,dict):
        expected={row['model']:row['sha256'] for row in performance.get('cases',[])}
        actual={name:meta.get('sha256') for name,meta in quality.get('models',{}).items()}
        compatible=compatible and performance.get('backend')=='torch' and expected==actual
        compatible=compatible and quality.get('backend')==performance.get('provider') and quality.get('precision')==performance.get('precision')
        if compatible:
            result['quality']=quality
            for key in ('syntheticTemporal','pending'):
                if key in previous:result[key]=previous[key]
        else:
            result['qualityInvalidation']='Prior screening invalidated: model artifact, backend, precision, or benchmark cases differ or provenance is incomplete.'
    elif 'syntheticTemporal' in previous:
        result['qualityInvalidation']='Prior temporal screening invalidated: matching quality provenance is unavailable.'
    return result
