import json
import tempfile
import unittest
from pathlib import Path


class ZenkaiRealESRRecipeTests(unittest.TestCase):
    def test_recipe_is_fixed_and_serializable(self):
        from zenkai_realesr_recipe import make_recipe
        recipe = make_recipe({'seed': 929, 'datasetManifestSha256': 'abc'})
        self.assertEqual(recipe['experimentKind'], 'zenkai-realesr-lite-v1')
        self.assertEqual(recipe['channels'], 32)
        self.assertEqual(recipe['blocks'], 6)
        self.assertEqual(recipe['steps'], 500)
        json.dumps(recipe)

    def test_recipe_rejects_wrong_identity_or_budget(self):
        from zenkai_realesr_recipe import make_recipe
        with self.assertRaises(ValueError):
            make_recipe({}, run='zenkai-realesr-lite-v1-smoke-501')
        with self.assertRaises(ValueError):
            make_recipe({}, steps=501)
        self.assertEqual(make_recipe({}, run='zenkai-realesr-lite-v1-smoke-500-retry1')['run'],
                         'zenkai-realesr-lite-v1-smoke-500-retry1')

    def test_run_directory_never_overwrites(self):
        from zenkai_realesr_recipe import create_run_dir, make_recipe
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recipe = make_recipe({'seed': 929})
            created = create_run_dir(root, recipe)
            self.assertTrue((created / 'config.json').is_file())
            with self.assertRaises(FileExistsError):
                create_run_dir(root, recipe)

    def test_training_runner_keeps_perceptual_metric_for_composite(self):
        source = Path(__file__).with_name('train_zenkai_realesr.py').read_text(encoding='utf-8')
        self.assertIn('smoke_composite', source)
        self.assertIn("'lpipsSqueeze': 'NOT MEASURED'", source)
        self.assertIn("'msSsim': 'NOT MEASURED'", source)


if __name__ == '__main__':
    unittest.main()
