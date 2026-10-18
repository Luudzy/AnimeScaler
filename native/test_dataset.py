import unittest
from prepare_dataset import eligible, split_for


class DatasetTests(unittest.TestCase):
    def test_filters(self):
        row={'rating':'s','file_ext':'png','file_url':'https://cdn.donmai.us/original/a.png','image_width':1280,'image_height':720}
        self.assertTrue(eligible(row))
        for change in [{'rating':'e'},{'file_ext':'gif'},{'image_height':719},{'file_url':'https://localhost/a.png'},{'file_url':'http://cdn.donmai.us/a.png'}]:
            self.assertFalse(eligible({**row,**change}))

    def test_content_split_is_deterministic(self):
        self.assertEqual(split_for('00000000'+'a'*56),'test')
        self.assertEqual(split_for('0000000a'+'a'*56),'validation')
        self.assertEqual(split_for('00000014'+'a'*56),'train')
