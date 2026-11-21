#!/usr/bin/env python3
"""Generate derived manifests, keeping the public raw dataset unchanged."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mudformer.io import read_jsonl
from mudformer.protocol import stratified_split,validate_splits

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--users',type=Path,required=True,help='Raw users.jsonl')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    raw=read_jsonl(a.users)
    rows=[{'user_id':str(r['user_id']),'label':int(r['label']),'path':str(r['user_id'])+'.npz','behavior_count':int(r['behavior_count'])} for r in raw]
    splits=stratified_split(rows,a.seed)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    for name,rows in splits.items():
        (a.output_dir/(name+'.json')).write_text(json.dumps({'seed':a.seed,'split':name,'ratios':[.8,.1,.1],'split_unit':'user','stratified_by':'label','records':rows},indent=2))
    identity=validate_splits(*(a.output_dir/(n+'.json') for n in ('train','validation','test')))
    print(json.dumps({'counts':{k:len(v) for k,v in splits.items()},'split_identity':identity},indent=2))
if __name__=='__main__':main()
