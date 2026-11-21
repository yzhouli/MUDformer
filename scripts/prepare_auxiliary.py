#!/usr/bin/env python3
"""Adapt public profiles/relations to the original auxiliary-feature pipeline."""
import argparse,json,csv,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mudformer.io import read_jsonl
import numpy as np
from mudformer.protocol import records,validate_splits

PROFILE_KEYS=('followers_count','friends_count','statuses_count','verified','verified_type','mbrank','mbtype')
RELATIONS=('follow','retweet','ori_uid','reply')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root',type=Path,required=True)
    p.add_argument('--splits',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();identity=validate_splits(*(a.splits/(n+'.json') for n in ('train','validation','test')))
    users=read_jsonl(a.raw_root/'users.jsonl')
    users.sort(key=lambda r:str(r['user_id']));ids=[str(u['user_id']) for u in users];lookup={uid:i for i,uid in enumerate(ids)}
    partitions={n:records(a.splits/(n+'.json')) for n in ('train','validation','test')}
    combined=[r for rows in partitions.values() for r in rows]
    if {r['user_id']:int(r['label']) for r in combined}!={u['user_id']:int(u['label']) for u in users}:raise ValueError('Users/labels differ between splits and raw data')
    train_ids={r['user_id'] for r in partitions['train']};train=np.asarray([uid in train_ids for uid in ids])
    numeric=np.zeros((len(users),8),np.float32)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    with (a.output_dir/'nodes.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['user_id','description','historical_profile_available']);writer.writeheader()
        for i,u in enumerate(users):
            profile=u.get('profile') or {};numeric[i,-1]=not bool(profile)
            for j,key in enumerate(PROFILE_KEYS):
                try:value=float(profile.get(key) or 0);numeric[i,j]=value if np.isfinite(value) else 0
                except (TypeError,ValueError):numeric[i,j]=0
            writer.writerow({'user_id':u['user_id'],'description':str(profile.get('description') or ''),'historical_profile_available':int(bool(profile))})
    edges=[];types=[];weights=[]
    with (a.raw_root/'relations.jsonl').open() as f:
        for line in f:
            if not line.strip():continue
            r=json.loads(line);source=r['source_user_id'];target=r['target_user_id'];kind=r['relation_type']
            if source in lookup and target in lookup and kind in RELATIONS:
                edges.append((lookup[source],lookup[target]));types.append(RELATIONS.index(kind));weights.append(int(r.get('observation_count',1)))
    graph=a.output_dir/'graph.npz'
    np.savez_compressed(graph,user_ids=np.asarray(ids),labels=np.asarray([u['label'] for u in users]),train_mask=train,test_mask=~train,profile_features=numeric,profile_feature_names=np.asarray(PROFILE_KEYS+('profile_missing',)),edge_index=np.asarray(edges,dtype=np.int64).reshape(-1,2).T,edge_type=np.asarray(types,dtype=np.int64),edge_weight=np.asarray(weights,dtype=np.int64),edge_type_names=np.asarray(RELATIONS))
    subprocess.run([sys.executable,str(Path(__file__).with_name('build_robust_aux_features.py')),'--graph',str(graph),'--nodes',str(a.output_dir/'nodes.csv'),'--output',str(a.output_dir/'auxiliary.npz'),'--seed',str(a.seed)],check=True)
    (a.output_dir/'protocol.json').write_text(json.dumps({'split_identity':identity,'normalization_fit':'train only','graph_coordinates':'transductive label-free full-cohort topology','profile_dim':72,'graph_dim':51,'public_profile_fields':'released description and numeric fields; unavailable fields empty'},indent=2))
if __name__=='__main__':main()
