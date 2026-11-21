#!/usr/bin/env python3
"""Encode public raw histories with locally supplied BERT and ViT encoders."""
import argparse,json,hashlib,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mudformer.io import read_jsonl
import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer,AutoModel,AutoImageProcessor
from tqdm import tqdm

def local_path(root,relative):
    p=(root/relative).resolve()
    if not p.is_relative_to(root.resolve()):
        raise ValueError('Dataset path escapes its root')
    return p

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--text-model',type=Path,required=True)
    p.add_argument('--image-model',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--max-text-length',type=int,default=128)
    p.add_argument('--device',default='cuda')
    a=p.parse_args()
    if a.batch_size<1:raise ValueError('batch size must be positive')
    torch.set_num_threads(2)
    device=torch.device(a.device)
    tokenizer=AutoTokenizer.from_pretrained(a.text_model,local_files_only=True)
    text_model=AutoModel.from_pretrained(a.text_model,local_files_only=True).eval().to(device)
    processor=AutoImageProcessor.from_pretrained(a.image_model,local_files_only=True,use_fast=False)
    image_model=AutoModel.from_pretrained(a.image_model,local_files_only=True).eval().to(device)
    dim=text_model.config.hidden_size
    if dim!=image_model.config.hidden_size:raise ValueError('Text and image embedding dimensions differ')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    # Hash encoder files to prevent mixing incompatible cached user features.
    identity=hashlib.sha256()
    for folder in (a.text_model,a.image_model):
        for f in sorted(folder.rglob('*')):
            if f.is_file():
                identity.update(str(f.relative_to(folder)).encode())
                with f.open('rb') as stream:
                    for chunk in iter(lambda:stream.read(8*1024*1024),b''):identity.update(chunk)
    settings={'encoder_sha256':identity.hexdigest(),'text_length':a.max_text_length,'pooling':'CLS','image_policy':'first available image','dim':dim,'history_order':'published chronological order'}
    settings_path=a.output_dir/'preprocessing.json'
    if settings_path.exists() and json.loads(settings_path.read_text())!=settings:raise ValueError('Output contains different encoder settings')
    settings_path.write_text(json.dumps(settings,indent=2))
    users=read_jsonl(a.raw_root/'users.jsonl')
    for user in tqdm(users,desc='users'):
        uid=str(user['user_id'])
        dest=local_path(a.output_dir,uid+'.npz')
        history=local_path(a.raw_root,user['history_path'])
        source_sha=hashlib.sha256(history.read_bytes()).hexdigest()
        receipt=dest.with_suffix('.json')
        if dest.exists() and receipt.exists():
            saved=json.loads(receipt.read_text())
            if saved.get('history_sha256')!=source_sha:raise ValueError('History changed for cached user')
            continue
        started=time.perf_counter()
        events=read_jsonl(history)
        if len(events)!=int(user['behavior_count']):raise ValueError('History count mismatch')
        n=len(events);size=max(1,n)
        text=np.zeros((size,dim),np.float16);image=np.zeros_like(text);present=np.zeros(size,bool)
        with torch.inference_mode():
            for start in range(0,n,a.batch_size):
                batch=events[start:start+a.batch_size]
                tokens=tokenizer([str(e.get('text') or '') for e in batch],padding=True,truncation=True,max_length=a.max_text_length,return_tensors='pt').to(device)
                text[start:start+len(batch)]=text_model(**tokens).last_hidden_state[:,0].float().cpu().numpy()
                images=[];indices=[]
                for j,e in enumerate(batch):
                    for name in e.get('image_paths',[]):
                        try:
                            with Image.open(local_path(a.raw_root,name)) as opened:img=opened.convert('RGB')
                            images.append(img);indices.append(start+j);break
                        except (OSError,ValueError):continue
                if images:
                    pixels=processor(images=images,return_tensors='pt')['pixel_values'].to(device)
                    image[indices]=image_model(pixel_values=pixels).last_hidden_state[:,0].float().cpu().numpy()
                    present[indices]=True
                    for img in images:img.close()
        if not np.isfinite(text).all() or not np.isfinite(image).all():raise ValueError('Non-finite embedding')
        temporary=dest.with_suffix('.tmp.npz')
        np.savez_compressed(temporary,text=text,image=image,image_present=present,behavior_present=np.asarray(n>0))
        temporary.replace(dest)
        receipt.write_text(json.dumps({'behaviors':n,'image_behaviors':int(present.sum()),'history_sha256':source_sha,'seconds':time.perf_counter()-started},indent=2))
if __name__=='__main__':main()
