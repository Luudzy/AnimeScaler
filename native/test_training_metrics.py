import unittest
from training_metrics import quality_report

class QualityReports(unittest.TestCase):
    def test_bilinear_independent_of_baseline_and_test(self):
        scores={'candidate':{'mae':2.,'edgeMae':2.},'previous':{'mae':1.,'edgeMae':1.},'bilinear':{'mae':3.,'edgeMae':3.}}
        report=quality_report({'validation':scores})
        self.assertTrue(report['improvesOverBilinear'])
        self.assertFalse(report['passesValidationGate'])
        better={**scores,'candidate':{'mae':.5,'edgeMae':.5}}
        self.assertTrue(quality_report({'validation':better,'test':scores})['passesValidationGate'])
        self.assertAlmostEqual(report['comparisons']['validation']['maeVsBilinearPercent'],-100/3)

if __name__=='__main__':unittest.main()
