#!/usr/bin/env python3
"""Rank five chronological history blocks using the trained audit decision margin."""
import argparse,json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from mudformer.data import UserFeatureDataset,AuxiliaryFeatureStore,collate_users
from mudformer.runtime import load_model_checkpoint,move_batch,seed_everything
from mudformer.temporal_localization import temporal_groups

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--feature-root',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--aux-features',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-length',type=int,default=16384)
    p.add_argument('--device',default='cuda')
    a=p.parse_args();seed_everything(42);device=torch.device(a.device)
    model=load_model_checkpoint(a.checkpoint,device).eval()
    for param in model.parameters():param.requires_grad_(False)
    auxiliary=AuxiliaryFeatureStore(a.aux_features)
    dataset=UserFeatureDataset(a.manifest,a.feature_root,a.max_length,training=False,auxiliary_features=auxiliary)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w') as f:
        for batch in DataLoader(dataset,batch_size=1,collate_fn=collate_users):
            batch=move_batch(batch,device);uid=batch['user_ids'][0]
            n=int(batch['sequence_mask'].sum())
            if not bool(batch['behavior_present'][0]) or n<5:
                f.write(json.dumps({'user_id':uid,'status':'insufficient_history'})+'\n');continue
            text=batch['text'].detach().requires_grad_(True);image=batch['image'].detach().requires_grad_(True)
            kw={k:batch[k] for k in ('profile','graph','profile_present','graph_present','behavior_present') if k in batch}
            outputs=model(text,image,batch['sequence_mask'],batch['image_present'],sample_latent=False,**kw)
            logits=outputs['audit_logits'];gradient=torch.autograd.grad(logits[0,1]-logits[0,0],(text,image))
            scores=(gradient[0]*text).sum(-1)[0]+(gradient[1]*image).sum(-1)[0]*batch['image_present'][0]
            groups=temporal_groups(n);blocks=[float(scores[group].sum().detach()) for group in groups]
            row={'user_id':uid,'retained_history_length':n,'audit_probability':float(logits.softmax(-1)[0,1].detach()),'block_scores':blocks,'highest_risk_block':int(max(range(5),key=lambda i:blocks[i])),'block_index_base':0}
            f.write(json.dumps(row)+'\n')
if __name__=='__main__':main()
