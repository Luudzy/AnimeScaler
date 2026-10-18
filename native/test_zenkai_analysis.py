import copy
import unittest
from zenkai_analysis import validate_matched_configs, pareto_dominators, analyze_results

MODELS=['zenkai-a','zenkai-b','adore-lite-v3','realesr-anime','hfa2k-cugan']


def fixtures():
    quality={'state':'complete','backend':'ROCm','inferenceBackend':'torch','precision':'fp16','gpu':'RX 5500 XT',
             'models':{name:{'sha256':name} for name in MODELS},'cases':{},'sampleCounts':{}}
    for case in ['clean','heavy','codec-intra-h264','codec-intra-h265','codec-intra-av1']:
        quality['sampleCounts'][case]=32
        quality['cases'][case]={name:{'mae':.01,'edgeMae':.02,'lpipsSqueeze':.10+index*.01,
                                     'ssim':.95,'msSsim':.98,'hfError':.02,'laplacianMae':.03}
                                for index,name in enumerate(MODELS)}
    performance={'complete':True,'backend':'torch','provider':'ROCm','precision':'fp16','gpu':'RX 5500 XT','cases':[]}
    for name in MODELS:
        for width,height,ow,oh in [(640,360,1280,720),(1280,720,1920,1080)]:
            performance['cases'].append({'model':name,'artifactSha256':name,'width':width,'height':height,
                'outputWidth':ow,'outputHeight':oh,'medianMs':10.,'p95Ms':11.,'meanMs':10.1,'p99Ms':12.,
                'peakAllocatedMiB':100.,'warmup':50,'iterations':500})
    return quality,performance


class ZenkaiAnalysisTests(unittest.TestCase):
    def test_matching_all_fields_except_architecture_depth_and_run_identity(self):
        a={'family':'Zenkai','channels':32,'nativeScale':2,'depth':5,'run':'a','parameters':100,'seed':929,'loss':'frequency'}
        b={**a,'depth':6,'run':'b','parameters':120}
        self.assertTrue(validate_matched_configs(a,b))
        for changed in [{**b,'seed':1},{**b,'loss':'other'},{**b,'extra':True},{**b,'depth':5}]:
            with self.assertRaises(ValueError):validate_matched_configs(a,changed)

    def test_pareto_requires_one_strict_improvement_and_uses_same_workload(self):
        points={'a':{'latency':10.,'error':.10},'b':{'latency':12.,'error':.12},
                'c':{'latency':8.,'error':.15},'d':{'latency':10.,'error':.10}}
        result=pareto_dominators(points)
        self.assertEqual(result['b'],['a','d'])
        self.assertEqual(result['a'],[]);self.assertEqual(result['c'],[]);self.assertEqual(result['d'],[])

    def test_regressions_are_visible_even_when_mean_lpips_improves(self):
        quality,performance=fixtures()
        quality['cases']['heavy']['zenkai-a']['lpipsSqueeze']=.15
        result=analyze_results(quality,performance)
        self.assertFalse(result['releaseApproved'])
        self.assertEqual(result['decision'],'REVIEW_REQUIRED')
        self.assertTrue(any(row['case']=='heavy' and row['metric']=='lpipsSqueeze'
                            for row in result['models']['zenkai-a']['regressionsVsV3']))
        self.assertEqual(len(result['workloads']),2)

    def test_changed_hash_precision_device_or_case_invalidates_comparison(self):
        quality,performance=fixtures()
        variants=[]
        changed=copy.deepcopy(performance);changed['cases'][0]['artifactSha256']='wrong';variants.append(changed)
        changed=copy.deepcopy(performance);changed['precision']='fp32';variants.append(changed)
        changed=copy.deepcopy(performance);changed['gpu']='other';variants.append(changed)
        changed=copy.deepcopy(performance);changed['backend']='onnx';variants.append(changed)
        changed=copy.deepcopy(performance);changed['cases'][0]['medianMs']=float('nan');variants.append(changed)
        changed=copy.deepcopy(performance);changed['cases'].pop();variants.append(changed)
        for changed in variants:
            with self.assertRaises(ValueError):analyze_results(quality,changed)
        quality['cases']['clean']['zenkai-a']['lpipsSqueeze']=float('nan')
        with self.assertRaises(ValueError):analyze_results(quality,performance)


if __name__=='__main__':unittest.main()
