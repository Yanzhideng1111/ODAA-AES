"""Full source-only nested CV grid sweep: actually reruns, never inserts manuscript numbers."""
import argparse
import copy
import json
from pathlib import Path
import pandas as pd
from odaa_aes.config import load_config
from odaa_aes.data import read_essays
from odaa_aes.train import run_outer_fold
p=argparse.ArgumentParser()
p.add_argument('--data',required=True);p.add_argument('--config',required=True)
p.add_argument('--out',required=True)
p.add_argument('--target',required=True,help='One held-out prompt; repeat for all prompts to macro-average')
p.add_argument('--kind',required=True,choices=['k','sigma','backbone','anchors','mapping'])
p.add_argument('--seed',type=int,default=42)
a=p.parse_args()
cfg=load_config(a.config);data=read_essays(a.data,cfg)
options={
  'k':[3,4,5,6,7,9],
  'sigma':[.05,.10,.15,.20,.30],
  'backbone':['bert-base-uncased','roberta-base','microsoft/deberta-v3-base'],
  'anchors':['uniform','quantile','fixed_nonuniform','density_adaptive'],
  'mapping':['minmax','source_zscore','source_percentile','source_piecewise']
}[a.kind]
rows=[]
for v in options:
    c=copy.deepcopy(cfg)
    if a.kind=='k':c['model']['num_anchors']=v
    if a.kind=='sigma':c['model']['sigma']=v
    if a.kind=='backbone':c['model']['backbone']=v
    if a.kind=='anchors':
        c['model']['anchor_strategy']=v
        if v=='fixed_nonuniform':c['model']['anchors']=[0,.10,.50,.90,1.]
    if a.kind=='mapping':c['training']['score_transform']=v
    name=str(v).replace('/','_')
    row=run_outer_fold(data,a.target,c,a.seed,str(Path(a.out)/name))
    rows.append({a.kind:v,**row['metrics']})
Path(a.out).mkdir(parents=True,exist_ok=True)
pd.DataFrame(rows).to_csv(Path(a.out)/'measured_sweep.csv',index=False)
