import unittest
import torch
from train_zenkai import validate_options, learning_rate, update_composite


class ZenkaiTrainingConfigTests(unittest.TestCase):
    def test_only_reviewed_runs_and_no_resume_or_overwrite_is_implicit(self):
        for options in [{'depth':5,'steps':5000,'run':'zenkai-lite-v1-a-5k'},
                        {'depth':6,'steps':5000,'run':'zenkai-lite-v1-b-5k'}]:
            validate_options(**options)
        for options in [{'depth':4,'steps':5000,'run':'a'},
                        {'depth':5,'steps':0,'run':'a'},
                        {'depth':5,'steps':5000,'run':'../unsafe'}]:
            with self.assertRaises(ValueError):validate_options(**options)

    def test_lr_schedule_is_matched_for_both_depths(self):
        self.assertEqual(learning_rate(1,5000),learning_rate(1,5000))
        self.assertAlmostEqual(learning_rate(1,5000),2e-4/300,places=10)
        self.assertAlmostEqual(learning_rate(5000,5000),1e-5,places=10)

    def test_composite_reports_each_domain_and_tracks_individual_regression(self):
        candidate={'mae':.01,'edgeMae':.02,'laplacianMae':.03,'ssim':.9,'lpipsSqueeze':.2,'msSsim':.9}
        previous={'mae':.02,'edgeMae':.04,'laplacianMae':.06,'ssim':.8,'lpipsSqueeze':.4,'msSsim':.8}
        report={'clean':{'candidate':candidate,'adore-lite-v3':previous},
                'heavy':{'candidate':{**candidate,'lpipsSqueeze':.3},'adore-lite-v3':previous}}
        summary=update_composite(report)
        self.assertEqual(summary['byCase']['clean']['mae'],.01)
        self.assertEqual(summary['byCase']['heavy']['lpipsSqueeze'],.3)
        self.assertGreater(summary['composite'],0)
        report['heavy']['candidate']['lpipsSqueeze']=.8
        worse=update_composite(report)
        self.assertNotEqual(summary['byCase']['heavy']['lpipsSqueeze'],worse['byCase']['heavy']['lpipsSqueeze'])


if __name__=='__main__':unittest.main()
