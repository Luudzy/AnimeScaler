import unittest
import torch
from evaluate_real_video import paired_temporal_metrics, verified_pairs, evaluate, candidate_identity
from prepare_video_degradation import ROOT, file_hash
from PIL import Image
from torch.nn import functional as F
from pathlib import Path
import tempfile
import json


class RealVideoTests(unittest.TestCase):
    def test_all_frame_review_preserves_metrics_and_saves_intermediate_frames(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'diagnostics') as folder:
            folder=Path(folder);rows=[]
            for index,color in enumerate((80,85,90,95,100)):
                row={'index':index}
                for kind,size in (('hr',(8,8)),('lr',(4,4))):
                    path=folder/f'{kind}-{index}.png'
                    Image.new('RGB',size,(color,color,color)).save(path)
                    row[kind]=path.name;row[kind+'Sha256']=file_hash(path)
                rows.append(row)
            manifest=folder/'complete.json'
            manifest.write_text(json.dumps({'syntheticSource':False,'frameCount':5,'frames':rows,
                'hrSize':[8,8],'lrSize':[4,4],'sourceSha256':'fixture','fps':'24/1'}))
            models={'fixture':lambda x:F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False)}
            normal=evaluate([manifest],models,folder/'normal',{})
            try:all_frames=evaluate([manifest],models,folder/'all',{},save_all_frames=True)
            except TypeError:self.fail('All-frame review API missing; intermediate frames cannot be inspected in motion')
            self.assertFalse((folder/'normal'/folder.name/'0001-fixture.png').exists())
            for index in range(5):
                self.assertTrue((folder/'all'/folder.name/f'{index:04d}-fixture.png').is_file())
            for key in ('mae','referenceCompensatedTemporalMae','includedTransitions'):
                self.assertEqual(normal['sequences'][0]['models']['fixture'][key],all_frames['sequences'][0]['models']['fixture'][key])
            self.assertEqual(all_frames['previewPolicy'],'all frames')

    def test_c_identity_is_preserved_and_unknown_probe_is_rejected(self):
        run='zenkai-lite-v1-c-codec50-5k'
        config={'run':run,'depth':5,'family':'ZenkaiIdentitySkipProbe'}
        self.assertEqual(candidate_identity({'config':config}),run)
        for wrong in ({**config,'run':'zenkai-lite-v1-c-unreviewed-5k'},
                      {**config,'family':'Zenkai'}, {**config,'depth':6}):
            with self.assertRaises(ValueError):candidate_identity({'config':wrong})

    def test_ablation_is_not_mislabeled_as_original_a(self):
        run='zenkai-ablation-frequency-3k-continuation'
        self.assertEqual(candidate_identity({'config':{'run':run,'depth':5}}),run)
        with self.assertRaises(ValueError):candidate_identity({'config':{'run':'zenkai-ablation-../../escape','depth':5}})

    def test_b_is_not_mislabeled_as_a(self):
        self.assertEqual(candidate_identity({'config': {'depth': 6}}), 'zenkai-b')
        self.assertEqual(candidate_identity({'config': {'depth': 5}}), 'zenkai-a')
        with self.assertRaises(ValueError):
            candidate_identity({'config': {'depth': 7}})

    def test_reference_motion_is_not_called_flicker(self):
        old = torch.zeros(1, 3, 4, 4)
        new = torch.ones_like(old) * .2
        result = paired_temporal_metrics(new, old, new, old)
        self.assertEqual(result['referenceCompensatedDeltaMae'], 0)
        self.assertAlmostEqual(result['referenceDeltaMae'], .2, places=6)

    def test_alternating_model_bias_is_detected(self):
        ref = torch.ones(1, 3, 4, 4) * .5
        result = paired_temporal_metrics(ref + .1, ref - .1, ref, ref)
        self.assertAlmostEqual(result['referenceCompensatedDeltaMae'], .2, places=6)

    def test_bad_shapes_and_nonfinite_outputs_rejected(self):
        x = torch.zeros(1, 3, 4, 4)
        for bad in (x[:, :, :2], x + float('nan')):
            with self.assertRaises(ValueError):
                paired_temporal_metrics(bad, x, x, x)

    def test_synthetic_manifest_and_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'complete.json'
            for report in ({'syntheticSource': True}, {'syntheticSource': False,
                'frameCount': 2, 'frames': [{'index': 0, 'hr': '../escape', 'lr': 'low'}]}):
                path.write_text(json.dumps(report))
                with self.assertRaises(ValueError):
                    verified_pairs(path)

    def test_complete_pair_evaluation_and_tamper_detection(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'diagnostics') as folder:
            folder = Path(folder)
            rows = []
            for index, color in enumerate((80, 90)):
                row = {'index': index}
                for kind, size in (('hr', (8, 8)), ('lr', (4, 4))):
                    path = folder / f'{kind}-{index}.png'
                    Image.new('RGB', size, (color, color, color)).save(path)
                    row[kind] = path.name
                    row[kind + 'Sha256'] = file_hash(path)
                rows.append(row)
            manifest = folder / 'complete.json'
            manifest.write_text(json.dumps({'syntheticSource': False, 'frameCount': 2,
                'frames': rows, 'hrSize': [8, 8], 'lrSize': [4, 4],
                'sourceSha256': 'fixture', 'fps': '24/1'}))
            result = evaluate([manifest], {'fixture': lambda x: F.interpolate(x, scale_factor=2,
                mode='bilinear', align_corners=False)}, folder / 'results', {})
            self.assertEqual(result['state'], 'complete')
            self.assertEqual(result['sequences'][0]['models']['fixture']['includedTransitions'], 1)
            self.assertLess(result['sequences'][0]['models']['fixture']['mae'], 1e-6)
            self.assertTrue((folder / 'results' / folder.name / '0000-fixture.png').exists())
            (folder / rows[0]['lr']).write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'hash'):
                verified_pairs(manifest)


if __name__ == '__main__':
    unittest.main()
