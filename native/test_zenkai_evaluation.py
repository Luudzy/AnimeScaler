import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from zenkai_evaluation import score_prediction, ZenkaiFixedEvaluator
from zenkai_validation_manifest import write_validation_manifest


class ZenkaiEvaluationTests(unittest.TestCase):
    def test_metrics_capture_reconstruction_edges_and_frequency(self):
        target=torch.zeros(1,3,16,16);target[...,8:]=1
        mae,metrics=score_prediction(target,target)
        self.assertEqual(float(mae),0)
        self.assertEqual(metrics['mae'],0)
        self.assertEqual(metrics['hfError'],0)
        self.assertEqual(metrics['dctError'],0)
        wrong=torch.zeros_like(target)
        _,metrics=score_prediction(wrong,target)
        self.assertGreater(metrics['mae'],0)
        self.assertGreater(metrics['edgeMae'],0)
        self.assertGreater(metrics['dctError'],0)

    def test_evaluator_uses_the_requested_fixed_cases_and_model_output(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temporary:
            root=Path(temporary);image=root/'scene.png';Image.new('RGB',(32,32),'gray').save(image)
            dataset=root/'dataset';dataset.mkdir();row={'file':str(image),'sha256':hashlib.sha256(image.read_bytes()).hexdigest(),
                'split':'validation','domain':'anime','cluster':'c0'}
            (dataset/'manifest.jsonl').write_text(json.dumps(row)+'\n',encoding='utf-8')
            allowed=[root]
            frozen=write_validation_manifest(dataset,root/'fixed.json',allowed_roots=allowed,patch_lr=8,per_image=1)
            evaluator=ZenkaiFixedEvaluator(root/'fixed.json',per_case=1,allowed_roots=allowed)
            model=torch.nn.Upsample(scale_factor=2,mode='nearest')
            report=evaluator.evaluate({'model':model},device='cpu')
            self.assertEqual(report['validationManifestSha256'],hashlib.sha256((root/'fixed.json').read_bytes()).hexdigest())
            self.assertEqual(report['sampleCounts'],{'clean':1,'heavy':1})
            self.assertEqual(report['cases']['clean']['model']['mae'],0)
            self.assertEqual(report['cases']['heavy']['model']['mae'],0)

    def test_manifest_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temporary:
            path=Path(temporary)/'bad.json';path.write_text(json.dumps({'samples':[]}),encoding='utf-8')
            with self.assertRaises(ValueError):ZenkaiFixedEvaluator(path)


if __name__=='__main__':unittest.main()
