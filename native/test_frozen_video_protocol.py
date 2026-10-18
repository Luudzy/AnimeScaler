import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from frozen_video_protocol import validate_protocol


class FrozenProtocolTests(unittest.TestCase):
    def test_counts_and_manifest_bytes_are_pinned(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            sequences=[]
            for index in range(3):
                path=root/f'{index}.json'
                path.write_text(json.dumps({'frameCount':48}),encoding='utf-8')
                sequences.append({'manifest':str(path),'frameCount':48,
                    'manifestSha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            report={'state':'complete','sequences':sequences}
            self.assertEqual(len(validate_protocol(report)),3)
            sequences[0]['frameCount']=47
            with self.assertRaisesRegex(ValueError,'144'):validate_protocol(report)
            sequences[0]['frameCount']=48
            Path(sequences[0]['manifest']).write_text('{}',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'hash'):validate_protocol(report)


if __name__=='__main__':unittest.main()
