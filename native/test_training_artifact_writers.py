import ast
from pathlib import Path
import unittest


class TrainingArtifactWriterTests(unittest.TestCase):
    def test_trainers_publish_json_metadata_atomically(self):
        expected_calls = {
            'train_lite_v2.py': 1,
            'train_lite_v3.py': 3,
            'train_lite_v31.py': 2,
            'continue_real_pairs.py': 2,
            'train_real_pairs_clean.py': 2,
            'train_zenkai_realesr.py': 4,
            'zenkai_factor_screens.py': 2,
        }
        root = Path(__file__).parent
        for filename, expected in expected_calls.items():
            with self.subTest(trainer=filename):
                tree = ast.parse((root / filename).read_text(encoding='utf-8'))
                atomic_calls = [node for node in ast.walk(tree)
                                if isinstance(node, ast.Call)
                                and isinstance(node.func, ast.Name)
                                and node.func.id == 'atomic_json_write']
                direct_writes = [node for node in ast.walk(tree)
                                 if isinstance(node, ast.Call)
                                 and isinstance(node.func, ast.Attribute)
                                 and node.func.attr == 'write_text']
                self.assertEqual(len(atomic_calls), expected)
                self.assertFalse(direct_writes)

    def test_screen_checkpoints_use_atomic_writer_except_existing_pending_protocol(self):
        expected_atomic = {
            'zenkai_scheduler_screen.py': 3,
            'zenkai_v2_dataset_screen.py': 3,
            'zenkai_c_screen.py': 3,
            'zenkai_activity.py': 1,
        }
        root = Path(__file__).parent
        for filename, expected in expected_atomic.items():
            with self.subTest(screen=filename):
                tree = ast.parse((root / filename).read_text(encoding='utf-8'))
                atomic_calls = [node for node in ast.walk(tree)
                                if isinstance(node, ast.Call)
                                and isinstance(node.func, ast.Name)
                                and node.func.id == 'atomic_torch_save']
                atomic_json_calls = [node for node in ast.walk(tree)
                                     if isinstance(node, ast.Call)
                                     and isinstance(node.func, ast.Name)
                                     and node.func.id == 'atomic_json_create']
                raw_calls = [node for node in ast.walk(tree)
                             if isinstance(node, ast.Call)
                             and isinstance(node.func, ast.Attribute)
                             and isinstance(node.func.value, ast.Name)
                             and node.func.value.id == 'torch'
                             and node.func.attr == 'save']
                self.assertEqual(len(atomic_calls), expected)
                self.assertEqual(len(raw_calls), 1 if filename != 'zenkai_activity.py' else 0)
                if filename == 'zenkai_activity.py':
                    self.assertEqual(len(atomic_json_calls), 1)
                    self.assertFalse(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                                         and node.func.attr == 'write_text' for node in ast.walk(tree)))

    def test_training_checkpoint_writers_are_atomic(self):
        expected = {
            'continue_real_pairs.py': (0, 2),
            'train_lite_v2.py': (1, 0),
            'train_lite_v3.py': (2, 0),
            'train_lite_v31.py': (2, 1),
            'train_real_pairs_clean.py': (0, 2),
            'train_zenkai_realesr.py': (0, 3),
            'zenkai_domain_screen.py': (2, 1),
        }
        root = Path(__file__).parent
        for filename, (replace_count, create_count) in expected.items():
            with self.subTest(trainer=filename):
                tree = ast.parse((root / filename).read_text(encoding='utf-8'))
                calls = {name: [node for node in ast.walk(tree)
                                if isinstance(node, ast.Call)
                                and isinstance(node.func, ast.Name)
                                and node.func.id == name]
                         for name in ('atomic_torch_save', 'atomic_torch_save_new')}
                direct_saves = [node for node in ast.walk(tree)
                                if isinstance(node, ast.Call)
                                and isinstance(node.func, ast.Attribute)
                                and isinstance(node.func.value, ast.Name)
                                and node.func.value.id == 'torch'
                                and node.func.attr == 'save']
                self.assertEqual(len(calls['atomic_torch_save']), replace_count)
                self.assertEqual(len(calls['atomic_torch_save_new']), create_count)
                self.assertFalse(direct_saves)

    def test_domain_screen_publishes_final_checkpoint_copy_atomically(self):
        tree = ast.parse((Path(__file__).parent / 'zenkai_domain_screen.py').read_text(encoding='utf-8'))
        atomic_copies = [node for node in ast.walk(tree)
                         if isinstance(node, ast.Call)
                         and isinstance(node.func, ast.Name)
                         and node.func.id == 'atomic_copy_file']
        direct_copies = [node for node in ast.walk(tree)
                         if isinstance(node, ast.Call)
                         and isinstance(node.func, ast.Attribute)
                         and isinstance(node.func.value, ast.Name)
                         and node.func.value.id == 'shutil'
                         and node.func.attr == 'copyfile']
        self.assertEqual(len(atomic_copies), 1)
        self.assertEqual(len(direct_copies), 1)
        self.assertNotIn('step-1000-training-state.pt', ast.unparse(direct_copies[0]))

    def test_recovery_flows_publish_checkpoint_copies_atomically(self):
        expected = {
            'zenkai_codec_resume.py': (1, 1),
            'zenkai_c_screen.py': (3, 1),
            'zenkai_scheduler_screen.py': (1, 0),
            'zenkai_smoke_resume.py': (0, 1),
            'zenkai_v2_dataset_screen.py': (1, 0),
        }
        root = Path(__file__).parent
        for filename, (replace_count, create_count) in expected.items():
            with self.subTest(flow=filename):
                tree = ast.parse((root / filename).read_text(encoding='utf-8'))
                atomic_copies = {name: [node for node in ast.walk(tree)
                                        if isinstance(node, ast.Call)
                                        and isinstance(node.func, ast.Name)
                                        and node.func.id == name]
                                 for name in ('atomic_copy_file', 'atomic_copy_file_new')}
                direct_checkpoint_copies = [node for node in ast.walk(tree)
                                            if isinstance(node, ast.Call)
                                            and isinstance(node.func, ast.Attribute)
                                            and isinstance(node.func.value, ast.Name)
                                            and node.func.value.id == 'shutil'
                                            and node.func.attr == 'copyfile'
                                            and len(node.args) > 1
                                            and '.pt' in ast.unparse(node.args[1])]
                self.assertEqual(len(atomic_copies['atomic_copy_file']), replace_count)
                self.assertEqual(len(atomic_copies['atomic_copy_file_new']), create_count)
                self.assertFalse(direct_checkpoint_copies)


if __name__ == '__main__':
    unittest.main()
