import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import artifact_io
from artifact_io import atomic_torch_save, atomic_torch_save_new, atomic_json_write, atomic_json_create, publish_training_checkpoint


class AtomicArtifactTests(unittest.TestCase):
    def test_json_create_refuses_existing_report(self):
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'report.json';target.write_text('{"owner":"existing"}')
            with self.assertRaisesRegex(FileExistsError,'Refusing to overwrite report'):
                atomic_json_create({'owner':'new'},target)
            self.assertEqual(json.loads(target.read_text()),{'owner':'existing'})

    def test_json_create_preserves_concurrent_writer(self):
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'report.json'
            original_link=__import__('os').link
            def competing_writer(source,path):
                Path(path).write_text('{"owner":"concurrent"}')
                return original_link(source,path)
            with patch('artifact_io.os.link',side_effect=competing_writer):
                with self.assertRaisesRegex(FileExistsError,'Refusing to overwrite report'):
                    atomic_json_create({'owner':'new'},target)
            self.assertEqual(json.loads(target.read_text()),{'owner':'concurrent'})

    def test_checkpoint_publication_refuses_to_overwrite_existing_state(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);checkpoint=root/'step-2.pt';checkpoint.write_bytes(b'protected')
            index_path=root/'checkpoint-index.json';index={'checkpoints':{},'lastCompleteStep':1,'lastCheckpoint':'step-1.pt'}
            atomic_json_write(index,index_path)
            with self.assertRaisesRegex(FileExistsError,'Refusing to overwrite'):
                publish_training_checkpoint({'step':2},checkpoint,index_path,index,'step-2.pt',2,'step-2.pt')
            self.assertEqual(checkpoint.read_bytes(),b'protected')
            self.assertEqual(json.loads(index_path.read_text()),index)

    def test_checkpoint_publication_preserves_file_created_after_initial_check(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);checkpoint=root/'step-2.pt';index_path=root/'checkpoint-index.json'
            index={'checkpoints':{},'lastCompleteStep':1,'lastCheckpoint':'step-1.pt'}
            atomic_json_write(index,index_path)
            original_link=__import__('os').link
            def competing_writer(source,path):
                Path(path).write_bytes(b'concurrent checkpoint')
                return original_link(source,path)
            with patch('artifact_io.os.link',side_effect=competing_writer):
                with self.assertRaisesRegex(FileExistsError,'Refusing to overwrite'):
                    publish_training_checkpoint({'step':2},checkpoint,index_path,index,'step-2.pt',2,'step-2.pt')
            self.assertEqual(checkpoint.read_bytes(),b'concurrent checkpoint')
            self.assertEqual(json.loads(index_path.read_text()),index)

    def test_checkpoint_publication_updates_hash_index_after_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);checkpoint=root/'step-2.pt';index_path=root/'checkpoint-index.json';report_path=root/'report.json'
            index={'checkpoints':{'step-1.pt':'old-digest'},'lastCompleteStep':1,'lastCheckpoint':'step-1.pt'}
            atomic_json_write(index,index_path);atomic_json_write({'stepsExecuted':1},report_path)
            updated=publish_training_checkpoint({'step':2},checkpoint,index_path,index,'step-2.pt',2,'step-2.pt',
                report_path,{'stepsExecuted':2})
            self.assertEqual(updated['lastCompleteStep'],2)
            self.assertEqual(updated['checkpoints']['step-2.pt'],__import__('hashlib').sha256(checkpoint.read_bytes()).hexdigest())
            self.assertEqual(json.loads(index_path.read_text())['lastCheckpoint'],'step-2.pt')
            self.assertEqual(json.loads(report_path.read_text()),{'stepsExecuted':2})

    def test_index_failure_keeps_previous_index_and_removes_unpublished_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);checkpoint=root/'step-2.pt';index_path=root/'checkpoint-index.json';report_path=root/'report.json'
            previous={'checkpoints':{'step-1.pt':'old-digest'},'lastCompleteStep':1,'lastCheckpoint':'step-1.pt'}
            atomic_json_write(previous,index_path);atomic_json_write({'stepsExecuted':1},report_path)
            with patch('artifact_io.atomic_json_write',side_effect=PermissionError('index locked')):
                with self.assertRaisesRegex(PermissionError,'index locked'):
                    publish_training_checkpoint({'step':2},checkpoint,index_path,previous,'step-2.pt',2,'step-2.pt',
                        report_path,{'stepsExecuted':2})
            self.assertEqual(json.loads(index_path.read_text()),previous)
            self.assertEqual(json.loads(report_path.read_text()),{'stepsExecuted':1})
            self.assertFalse(checkpoint.exists())

    def test_checkpoint_round_trip_and_atomic_replacement(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'last.pt'
            target.write_bytes(b'old')
            atomic_torch_save({'step': 2, 'model': torch.tensor([1., 2.])}, target)
            saved = torch.load(target, map_location='cpu', weights_only=True)
            self.assertEqual(saved['step'], 2)
            self.assertTrue(torch.equal(saved['model'], torch.tensor([1., 2.])))
            self.assertEqual([file.name for file in Path(folder).iterdir()], ['last.pt'])

    def test_new_checkpoint_save_refuses_to_replace_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'step-2.pt'
            target.write_bytes(b'previous complete checkpoint')
            with self.assertRaisesRegex(FileExistsError, 'Refusing to overwrite checkpoint'):
                atomic_torch_save_new({'step': 2}, target)
            self.assertEqual(target.read_bytes(), b'previous complete checkpoint')
            self.assertEqual([file.name for file in Path(folder).iterdir()], ['step-2.pt'])

    def test_atomic_checkpoint_copy_creates_new_file_and_refuses_overwrite(self):
        copy_new = getattr(artifact_io, 'atomic_copy_file_new', None)
        self.assertIsNotNone(copy_new)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'attempt' / 'step-1000-training-state.pt'
            source.parent.mkdir()
            source.write_bytes(b'complete-checkpoint')
            target = root / 'step-1000-training-state.pt'
            copy_new(source, target)
            self.assertEqual(target.read_bytes(), b'complete-checkpoint')
            with self.assertRaisesRegex(FileExistsError, 'Refusing to overwrite checkpoint'):
                copy_new(source, target)
            self.assertEqual(target.read_bytes(), b'complete-checkpoint')

    def test_atomic_checkpoint_copy_replaces_existing_file(self):
        copy_replace = getattr(artifact_io, 'atomic_copy_file', None)
        self.assertIsNotNone(copy_replace)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'source.pt'
            target = root / 'target.pt'
            source.write_bytes(b'new-checkpoint')
            target.write_bytes(b'old-checkpoint')
            copy_replace(source, target)
            self.assertEqual(target.read_bytes(), b'new-checkpoint')
            self.assertEqual(sorted(path.name for path in root.iterdir()), ['source.pt', 'target.pt'])

    def test_interrupted_serialization_keeps_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'last.pt'
            target.write_bytes(b'previous complete checkpoint')
            def interrupted(value, stream):
                stream.write(b'partial checkpoint')
                raise OSError('disk full')
            with patch('torch.save', side_effect=interrupted):
                with self.assertRaisesRegex(OSError, 'disk full'):
                    atomic_torch_save({'step': 3}, target)
            self.assertEqual(target.read_bytes(), b'previous complete checkpoint')
            self.assertEqual([file.name for file in Path(folder).iterdir()], ['last.pt'])

    def test_failed_replace_keeps_previous_report_and_removes_temporary(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'report.json'
            target.write_text('{"step": 1}')
            with patch('artifact_io.os.replace', side_effect=PermissionError('file locked')):
                with self.assertRaisesRegex(PermissionError, 'file locked'):
                    atomic_json_write({'step': 2}, target)
            self.assertEqual(json.loads(target.read_text()), {'step': 1})
            self.assertEqual([file.name for file in Path(folder).iterdir()], ['report.json'])

    def test_nonfinite_report_is_rejected_without_corrupting_previous(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'report.json'
            atomic_json_write({'score': 1}, target)
            with self.assertRaises(ValueError):
                atomic_json_write({'score': float('nan')}, target)
            self.assertEqual(json.loads(target.read_text()), {'score': 1})

    def test_json_round_trip_preserves_unicode_and_complete_payload(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'report.json'
            value = {'name': 'avaliação', 'steps': [1, 2], 'promoted': False}
            atomic_json_write(value, target)
            self.assertEqual(json.loads(target.read_text(encoding='utf-8')), value)


if __name__ == '__main__':
    unittest.main()
