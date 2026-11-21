import unittest,json,tempfile,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mudformer.protocol import stratified_split,validate_splits,check_resume
class ProtocolTests(unittest.TestCase):
    def test_sizes_and_disjointness(self):
        for n0,n1 in [(343,342),(4728,5272)]:
            rows=[{'user_id':f'u{i}','label':int(i>=n0),'path':f'u{i}.npz','behavior_count':50} for i in range(n0+n1)]
            split=stratified_split(rows,42)
            self.assertEqual(sum(map(len,split.values())),n0+n1)
            self.assertEqual(len(split["train"]),int((n0+n1)*0.8))
            self.assertEqual(split,stratified_split(rows[::-1],42))
            sets=[{r['user_id'] for r in group} for group in split.values()]
            self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
            with tempfile.TemporaryDirectory() as tmp:
                files=[]
                for name,group in split.items():
                    p=Path(tmp)/name;p.write_text(json.dumps({'records':group}));files.append(p)
                identity=validate_splits(*files);check_resume({'split_identity':identity},identity)
                with self.assertRaises(ValueError):check_resume({'split_identity':'other'},identity)
                with self.assertRaises(ValueError):check_resume({},identity)
                with self.assertRaises(ValueError):validate_splits(files[0],files[0],files[2])
    def test_reject_legacy_ratio(self):
        with tempfile.TemporaryDirectory() as tmp:
            files=[]
            for j,counts in enumerate([(80,80),(10,10),(20,20)]):
                rows=[{'user_id':f'{j}-{label}-{i}','label':label} for label,n in enumerate(counts) for i in range(n)]
                p=Path(tmp)/str(j);p.write_text(json.dumps(rows));files.append(p)
            with self.assertRaises(ValueError):validate_splits(*files)
if __name__=='__main__':unittest.main()
