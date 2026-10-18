import copy
import unittest
from prepare_zenkai_dataset import partition_rows


class ZenkaiSplitTests(unittest.TestCase):
    def test_preserves_groups_hidden_and_original_rows(self):
        rows=[{'file':f'F:/treino/hfa2k/{i}.png','sha256':str(i),'cluster':f'g{i//2}',
               'split':'train' if i<8 else 'validation' if i<10 else 'final-hidden','domain':'anime'} for i in range(12)]
        original=copy.deepcopy(rows)
        result=partition_rows(rows,development_target=2)
        self.assertEqual(rows,original)
        self.assertEqual([r['sha256'] for r in result if r['split']=='final-hidden'],['10','11'])
        self.assertEqual([r['sha256'] for r in result if r['split']=='validation'],['8','9'])
        groups={}
        for row in result:groups.setdefault(row['cluster'],set()).add(row['split'])
        self.assertTrue(all(len(splits)==1 for splits in groups.values()))
        self.assertEqual(sum(r['split']=='development-test' for r in result),2)
        self.assertEqual({r['sha256']:r['split'] for r in result},
                         {r['sha256']:r['split'] for r in partition_rows(list(reversed(rows)),development_target=2)})

    def test_rejects_cross_split_cluster(self):
        rows=[{'file':'F:/treino/a.png','sha256':'a','cluster':'same','domain':'anime','split':split} for split in ['train','final-hidden']]
        with self.assertRaisesRegex(ValueError,'cluster'):partition_rows(rows,development_target=1)


if __name__=='__main__':unittest.main()
