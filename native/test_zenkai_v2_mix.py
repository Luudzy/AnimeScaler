import unittest
from prepare_zenkai_v2_mix import merge_train_only


class V2MixLeakageTests(unittest.TestCase):
    def test_union_preserves_all_holdouts_and_rejects_overlap_or_shared_group(self):
        base=[{'sha256':'a','cluster':'train-a','split':'train'},
              {'sha256':'v','cluster':'validation-v','split':'validation'},
              {'sha256':'h','cluster':'hidden-h','split':'final-hidden'}]
        extra=[{'sha256':'new','cluster':'new-illustration','split':'train'}]
        result=merge_train_only(base,extra)
        self.assertEqual(result[:3],base);self.assertEqual(len(result),4)
        for r in ({'sha256':'v','cluster':'new','split':'train'},
                  {'sha256':'new','cluster':'hidden-h','split':'train'},
                  {'sha256':'new','cluster':'new','split':'validation'},
                  {'sha256':'a','cluster':'train-a','split':'train'}):
            with self.assertRaises(ValueError):merge_train_only(base,[r])


if __name__=='__main__':unittest.main()
