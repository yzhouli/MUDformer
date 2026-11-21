import json,tempfile,unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mudformer.io import read_jsonl

class JsonlTests(unittest.TestCase):
    def test_unicode_separators_remain_inside_records(self):
        rows=[{'description':'first\u2028second\u2029third\u0085fourth'}, {'text':'another behavior'}]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'history.jsonl'
            path.write_text('\n'.join(json.dumps(row,ensure_ascii=False) for row in rows)+'\n',encoding='utf-8')
            self.assertEqual(read_jsonl(path),rows)
    def test_blank_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'users.jsonl';path.write_text('\n'+json.dumps({'user_id':'u0'})+'\n\n',encoding='utf-8')
            self.assertEqual(read_jsonl(path),[{'user_id':'u0'}])
if __name__=='__main__':unittest.main()
