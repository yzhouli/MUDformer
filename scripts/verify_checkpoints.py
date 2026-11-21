from pathlib import Path
import sys,json,hashlib,torch,numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from mudformer.runtime import load_model_checkpoint,seed_everything
torch.set_num_threads(2);seed_everything(42)
checks=[]
for ds in ['Weibo-25-plus','MORSE']:
 d=root/'checkpoints'/ds
 c=torch.load(d/'classification.pt',weights_only=True,map_location='cpu');a=torch.load(d/'audit.pt',weights_only=True,map_location='cpu')
 assert c['config']==a['config'];keys=[k for k in c['model'] if not k.startswith('temporal_risk_head.')]
 assert all(torch.equal(c['model'][k],a['model'][k]) for k in keys)
 assert any(not torch.equal(c['model'][k],a['model'][k]) for k in c['model'] if k.startswith('temporal_risk_head.'))
 base=load_model_checkpoint(d/'classification.pt',torch.device('cpu')).eval();audit=load_model_checkpoint(d/'audit.pt',torch.device('cpu')).eval()
 text=torch.randn(1,128,768);image=torch.randn_like(text);mask=torch.ones(1,128,dtype=torch.bool);present=mask.clone();present[:,::3]=False
 kw={'profile':torch.randn(1,72),'graph':torch.randn(1,51),'profile_present':torch.ones(1,dtype=torch.bool),'graph_present':torch.ones(1,dtype=torch.bool),'behavior_present':torch.ones(1,dtype=torch.bool)}
 with torch.inference_mode():
  x=base(text,image,mask,present,sample_latent=False,**kw);y=audit(text,image,mask,present,sample_latent=False,**kw)
  z=audit(text,image,mask,present,sample_latent=False,**kw)
 assert torch.equal(x['user_logits'],y['user_logits'])
 assert torch.equal(y['audit_logits'],z['audit_logits'])
 assert y['temporal_risk_logits'].shape==(1,128)
 for value in y.values():
  if isinstance(value,torch.Tensor):assert torch.isfinite(value).all()
 for name in ('classification.pt','audit.pt'):
  receipts=[]
  for receipt in (root/'checkpoints').glob('verification_*.json'):
   if isinstance(json.loads(receipt.read_text()),list):receipts.extend(json.loads(receipt.read_text()))
  expected=next(x['export_sha256'] for x in receipts if x.get('dataset')==ds and x.get('task')==name[:-3])
  assert hashlib.sha256((d/name).read_bytes()).hexdigest()==expected
 checks.append({'dataset':ds,'strict_state_load':True,'non_audit_keys_bitwise_equal':len(keys),'classification_logits_bitwise_equal':True,'deterministic_audit_logits':True,'parameters':sum(p.numel() for p in base.parameters()),'audit_trainable_parameters':sum(p.numel() for p in audit.temporal_risk_head.parameters())})
(root/'checkpoints/verification.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
