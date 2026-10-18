import unittest
import tempfile
import hashlib
from pathlib import Path


class ArtifactProvenanceTests(unittest.TestCase):
    def test_changed_initial_file_cannot_fabricate_weight_delta(self):
        from artifact_provenance import validate_checkpoint_file
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'initial.pt';path.write_bytes(b'original checkpoint')
            pin=hashlib.sha256(path.read_bytes()).hexdigest()
            validate_checkpoint_file(path,pin)
            path.write_bytes(b'replaced checkpoint')
            with self.assertRaisesRegex(ValueError,'checkpoint.*hash'):
                validate_checkpoint_file(path,pin)
    def test_comparisons_reject_replaced_or_missing_baselines(self):
        from artifact_provenance import validate_artifact_pins
        expected={'fixedSkip':'a'*64,'videoSkip':'a'*64,'fixedV3':'b'*64,'videoV3':'b'*64}
        actual={key:{'sha256':value} for key,value in expected.items()}
        validate_artifact_pins(actual,expected)
        for key in expected:
            bad={**actual,key:{'sha256':'c'*64}}
            with self.assertRaises(ValueError):validate_artifact_pins(bad,expected)
            bad={k:v for k,v in actual.items() if k!=key}
            with self.assertRaises(ValueError):validate_artifact_pins(bad,expected)


if __name__=='__main__':unittest.main()
