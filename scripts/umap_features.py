"""Recreate the four-panel paper-style UMAP from actual trained embeddings.

Requires optional umap-learn and matplotlib. Uses target essays ONLY after
both checkpoints are trained and frozen, never for tuning model weights.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from odaa_aes.config import load_config
from odaa_aes.data import read_essays, make_collate_fn, SegmentedEssayDataset, normalize, spec_for
from odaa_aes.train import load_checkpoint, make_tokenizer, device_batch
from torch.utils.data import DataLoader
try:
    import umap
except ImportError as e:
    raise SystemExit('Install umap-learn first: pip install umap-learn') from e
p=argparse.ArgumentParser()
p.add_argument('--data',required=True)
p.add_argument('--config',required=True)
p.add_argument('--encoder-only',required=True)
p.add_argument('--odaa',required=True)
p.add_argument('--out',required=True)
p.add_argument('--seed',type=int,default=42)
p.add_argument('--limit',type=int,default=1500)
a=p.parse_args()
cfg=load_config(a.config); records=read_essays(a.data,cfg)
rng=np.random.default_rng(a.seed)
if len(records)>a.limit:records=[records[i] for i in sorted(rng.choice(len(records),a.limit,replace=False))]
labels=np.asarray([normalize(r.score,spec_for(cfg,r.prompt_id)) for r in records]); prompts=[r.prompt_id for r in records]
def embeddings(folder):
    model,ckcfg,transform,device=load_checkpoint(folder)
    tok=make_tokenizer(ckcfg)
    loader=DataLoader(SegmentedEssayDataset(records,tok,ckcfg),batch_size=4,
                      collate_fn=make_collate_fn(tok))
    out=[];model.eval()
    with torch.no_grad():
        for batch in loader:
            batch=device_batch(batch,device)
            z=model(batch['input_ids'],batch['attention_mask'],batch['segment_mask'],adversarial=False)['z']
            out.append(z.cpu().numpy())
    return np.concatenate(out)
Z1=embeddings(a.encoder_only);Z2=embeddings(a.odaa)
# Same UMAP hyperparameters / seed, independently fitted 2D projections (not same coordinates).
proj1=umap.UMAP(n_neighbors=15,min_dist=.1,random_state=a.seed).fit_transform(Z1)
proj2=umap.UMAP(n_neighbors=15,min_dist=.1,random_state=a.seed).fit_transform(Z2)
fig,ax=plt.subplots(2,2,figsize=(10,8))
from sklearn.preprocessing import LabelEncoder
pclass=LabelEncoder().fit_transform(prompts)
for axy,points,name in [(ax[0,0],proj1,'Encoder only / prompt'),(ax[0,1],proj2,'ODAA / prompt')]:
    axy.scatter(points[:,0],points[:,1],c=pclass,s=5,cmap='tab20',alpha=.75)
    axy.set_title(name)
for axy,points,name in [(ax[1,0],proj1,'Encoder only / score'),(ax[1,1],proj2,'ODAA / score')]:
    ss=axy.scatter(points[:,0],points[:,1],c=labels,s=5,cmap='viridis',vmin=0,vmax=1,alpha=.75)
    axy.set_title(name)
fig.colorbar(ss,ax=ax[1].tolist(),label='Rubric-relative score',fraction=.045)
for axy in ax.flat:axy.set_xticks([]);axy.set_yticks([])
fig.subplots_adjust(wspace=.16,hspace=.20,right=.88)
Path(a.out).parent.mkdir(parents=True,exist_ok=True)
fig.savefig(a.out,dpi=240)
print('Saved UMAP visualization; interpret qualitatively, not as proof of prompt invariance:',a.out)
