import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import ctypes
import errno

from prepare_real_video_pairs import main, partition_clip_rows
from split_pair_manifest import main as split_pair_manifest_main


class RealVideoPairSplitTests(unittest.TestCase):
    def test_generic_split_cli_partitions_whole_source_groups_and_hashes_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'source'
            source.mkdir()
            rows = [{'file': f'frames/{clip}-{frame}.png',
                     'sha256': f'{clip * 10 + frame + 1:064x}', 'source': f'clip-{clip}'}
                    for clip in range(5) for frame in range(3)]
            (source / 'manifest.jsonl').write_text(
                '\n'.join(json.dumps(row) for row in rows) + '\n', encoding='utf-8')
            output = root / 'generated'
            with patch('sys.argv', ['split_pair_manifest.py', '--input', str(source),
                                    '--output', str(output), '--holdout-every', '3']):
                split_pair_manifest_main()

            train_bytes = (output / 'manifest.jsonl').read_bytes()
            holdout_bytes = (output / 'holdout.jsonl').read_bytes()
            train = [json.loads(line) for line in train_bytes.splitlines()]
            holdout = [json.loads(line) for line in holdout_bytes.splitlines()]
            metadata = json.loads((output / 'metadata.json').read_text(encoding='utf-8'))
            train_groups = {row['splitGroup'] for row in train}
            holdout_groups = {row['splitGroup'] for row in holdout}
            self.assertTrue(train_groups.isdisjoint(holdout_groups))
            self.assertTrue(all(row['split'] == 'train' for row in train))
            self.assertTrue(all(row['split'] == 'holdout' for row in holdout))
            self.assertEqual(metadata['sourceSha256'], hashlib.sha256(train_bytes).hexdigest())
            self.assertEqual(metadata['holdoutSha256'], hashlib.sha256(holdout_bytes).hexdigest())
            self.assertEqual(metadata['splitStrategy'],
                             'whole source clips; frames from one clip never cross splits')
            self.assertEqual(metadata['sourceGroups'], 5)

    def test_generic_split_cli_rejects_rows_without_source_groups(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'source'
            source.mkdir()
            (source / 'manifest.jsonl').write_text(
                json.dumps({'file': 'frame.png', 'sha256': 'a' * 64}) + '\n', encoding='utf-8')
            output = root / 'generated'
            with patch('sys.argv', ['split_pair_manifest.py', '--input', str(source), '--output', str(output)]):
                with self.assertRaisesRegex(ValueError, 'Missing source clip group'):
                    split_pair_manifest_main()
            self.assertFalse(output.exists())

    def test_all_frames_from_source_clip_stay_in_one_split(self):
        rows=[{'source':f'clip-{clip}','sha256':f'{clip}-{frame}'}
              for clip in range(8) for frame in range(5)]
        train,holdout=partition_clip_rows(rows,holdout_every=4)
        split_by_source={}
        for row in train:split_by_source.setdefault(row['source'],set()).add(row['split'])
        for row in holdout:split_by_source.setdefault(row['source'],set()).add(row['split'])
        self.assertTrue(all(len(splits)==1 for splits in split_by_source.values()))
        self.assertEqual({source for source,splits in split_by_source.items() if splits=={'holdout'}},
                         {'clip-0','clip-4'})
        self.assertEqual(len(train)+len(holdout),len(rows))

    def test_split_is_deterministic_and_requires_distinct_source_groups(self):
        rows=[{'source':f'clip-{clip}','sha256':str(clip)} for clip in range(3)]
        first=partition_clip_rows(rows,2)
        second=partition_clip_rows(list(reversed(rows)),2)
        self.assertEqual(first,second)
        with self.assertRaisesRegex(ValueError,'two source clips'):
            partition_clip_rows([{'source':'only','sha256':'x'}],2)

    def _make_clip_tree(self, root, clip_count, frames_per_clip, missing_pair=None):
        for clip_index in range(clip_count):
            clip = root / f'clip-{clip_index:02d}-360-to-720'
            hr = clip / 'hr'
            lr = clip / 'lr'
            hr.mkdir(parents=True)
            lr.mkdir()
            for frame_index in range(frames_per_clip):
                name = f'frame-{frame_index:03d}.png'
                (hr / name).write_bytes(f'hr-{clip_index}-{frame_index}'.encode())
                if missing_pair != (clip_index, frame_index):
                    (lr / name).write_bytes(f'lr-{clip_index}-{frame_index}'.encode())

    def _run_cli(self, clips_root, output):
        with patch('sys.argv', ['prepare_real_video_pairs.py', '--clips-root', str(clips_root),
                                '--output', str(output)]):
            main()

    def test_cli_writes_hashed_train_holdout_manifests_and_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            clips_root = root / 'clips'
            clips_root.mkdir()
            output = root / 'generated'
            self._make_clip_tree(clips_root, clip_count=7, frames_per_clip=25)

            self._run_cli(clips_root, output)

            manifest_bytes = (output / 'manifest.jsonl').read_bytes()
            holdout_bytes = (output / 'holdout.jsonl').read_bytes()
            train = [json.loads(line) for line in manifest_bytes.splitlines()]
            holdout = [json.loads(line) for line in holdout_bytes.splitlines()]
            metadata = json.loads((output / 'metadata.json').read_text(encoding='utf-8'))
            self.assertEqual(len(train), 125)
            self.assertEqual(len(holdout), 50)
            self.assertTrue(all(row['split'] == 'train' and row['splitGroup'] == row['source']
                                for row in train))
            self.assertTrue(all(row['split'] == 'holdout' and row['splitGroup'] == row['source']
                                for row in holdout))
            self.assertEqual({row['source'] for row in holdout},
                             {'clip-00-360-to-720', 'clip-04-360-to-720'})
            for row in train + holdout:
                self.assertFalse(Path(row['file']).is_absolute())
                self.assertFalse(Path(row['pairedLr']).is_absolute())
                self.assertEqual(hashlib.sha256((output / row['file']).read_bytes()).hexdigest(), row['sha256'])
                self.assertEqual(hashlib.sha256((output / row['pairedLr']).read_bytes()).hexdigest(),
                                 row['pairedLrSha256'])
            self.assertEqual(metadata['state'], 'complete')
            self.assertEqual(metadata['trainPairs'], len(train))
            self.assertEqual(metadata['holdoutPairs'], len(holdout))
            self.assertEqual(metadata['sourceSha256'], hashlib.sha256(manifest_bytes).hexdigest())
            self.assertEqual(metadata['holdoutSha256'], hashlib.sha256(holdout_bytes).hexdigest())
            self.assertEqual(metadata['splitStrategy'],
                             'whole source clips; frames from one clip never cross splits')
            self.assertEqual(metadata['sourceGroups'], 7)

    def test_cli_rejects_missing_lr_pair_without_creating_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            clips_root = root / 'clips'
            clips_root.mkdir()
            output = root / 'generated'
            self._make_clip_tree(clips_root, clip_count=2, frames_per_clip=1,
                                 missing_pair=(1, 0))

            with self.assertRaises(FileNotFoundError):
                self._run_cli(clips_root, output)

            self.assertFalse(output.exists())

    def test_cli_rejects_existing_output_before_writing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            clips_root = root / 'clips'
            clips_root.mkdir()
            output = root / 'generated'
            output.mkdir()
            sentinel = output / 'keep.txt'
            sentinel.write_text('preserve', encoding='utf-8')

            with self.assertRaises(FileExistsError):
                self._run_cli(clips_root, output)

            self.assertEqual(sentinel.read_text(encoding='utf-8'), 'preserve')
            self.assertEqual(list(output.iterdir()), [sentinel])

    def test_cli_cleans_staging_when_manifest_write_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            clips_root = root / 'clips'
            clips_root.mkdir()
            output = root / 'generated'
            self._make_clip_tree(clips_root, clip_count=7, frames_per_clip=25)
            original_write_text = Path.write_text

            def fail_metadata_write(path, *args, **kwargs):
                if path.name == 'metadata.json':
                    raise OSError('injected metadata write failure')
                return original_write_text(path, *args, **kwargs)

            with patch.object(Path, 'write_text', fail_metadata_write):
                with self.assertRaisesRegex(OSError, 'injected metadata write failure'):
                    self._run_cli(clips_root, output)

            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob('.generated.*.pending')), [])

    def test_cli_preserves_concurrent_destination_at_publish(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            clips_root = root / 'clips'
            clips_root.mkdir()
            output = root / 'generated'
            self._make_clip_tree(clips_root, clip_count=7, frames_per_clip=25)
            original_rename = os.rename

            def create_racing_destination(source, destination):
                destination = Path(destination)
                destination.mkdir()
                sentinel = destination / 'concurrent.txt'
                sentinel.write_text('keep', encoding='utf-8')
                return original_rename(source, destination)

            with patch.object(os, 'rename', side_effect=create_racing_destination):
                with self.assertRaises((FileExistsError, OSError)):
                    self._run_cli(clips_root, output)

            self.assertEqual((output / 'concurrent.txt').read_text(encoding='utf-8'), 'keep')
            self.assertEqual(list(root.glob('.generated.*.pending')), [])

    def test_cli_does_not_replace_empty_destination_created_during_posix_publish(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            clips_root = root / 'clips'
            clips_root.mkdir()
            output = root / 'generated'
            self._make_clip_tree(clips_root, clip_count=7, frames_per_clip=25)
            original_rename = os.rename
            created = {}

            class RenameAt2:
                def renameat2(self, source_dir, source, destination_dir, destination, flags):
                    created['flags'] = flags
                    target = Path(os.fsdecode(destination))
                    target.mkdir()
                    created['stat'] = target.stat()
                    ctypes.set_errno(errno.EEXIST)
                    return -1

            def simulated_posix_overwrite(source, destination):
                target = Path(destination)
                target.mkdir()
                created['stat'] = target.stat()
                target.rmdir()
                return original_rename(source, destination)

            with patch.object(sys, 'platform', 'linux'), \
                    patch.object(ctypes, 'CDLL', return_value=RenameAt2()), \
                    patch.object(os, 'rename', side_effect=simulated_posix_overwrite):
                with self.assertRaises(OSError):
                    self._run_cli(clips_root, output)

            self.assertEqual(created['flags'], 1)
            self.assertTrue(output.is_dir())
            self.assertEqual(list(output.iterdir()), [])
            self.assertEqual(output.stat().st_ino, created['stat'].st_ino)
            self.assertEqual(list(root.glob('.generated.*.pending')), [])

    def test_cli_requires_at_least_100_training_pairs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            clips_root = root / 'clips'
            clips_root.mkdir()
            output = root / 'generated'
            self._make_clip_tree(clips_root, clip_count=5, frames_per_clip=25)

            with self.assertRaisesRegex(ValueError, 'at least 100 training pairs'):
                self._run_cli(clips_root, output)

            self.assertFalse(output.exists())


if __name__=='__main__':unittest.main()
