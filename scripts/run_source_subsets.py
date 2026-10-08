"""Actual source-count/composition study, without selecting sources by target QWK.

Use the default all-source study in cli.py for the principal paper benchmark.
This script adds the 1/3/5/7 source-prompt setting and subset SD.
"""
import argparse
import itertools
from pathlib import Path
import json
import numpy as np
import pandas as pd
from odaa_aes.config import load_config
from odaa_aes.data import read_essays,unique_prompts
from odaa_aes.train import run_outer_fold
p=argparse.ArgumentParser()
p.add_argument('--data',required=True);p.add_argument('--config',required=True)
p.add_argument('--target',required=True);p.add_argument('--out',required=True)
p.add_argument('--sizes',default='1,3,5,7');p.add_argument('--max-combinations',type=int,default=6)
p.add_argument('--seeds',default='42,43,44,45,46');p.add_argument('--selection-seed',type=int,default=2026)
a=p.parse_args()
cfg=load_config(a.config); all_essays=read_essays(a.data,cfg)
source=[x for x in unique_prompts(all_essays) if x!=str(a.target)]
if len(source)<1:raise SystemExit('No source prompts')
output=Path(a.out);output.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(a.selection_seed)
rows=[];manifest=[]
for size in [int(x) for x in a.sizes.split(',')]:
    if size>len(source):continue
    choices=list(itertools.combinations(source,size))
    selected=sorted(rng.choice(len(choices),min(len(choices),a.max_combinations),replace=False))
    for idx in selected:
        chosen=choices[int(idx)];genre={x.genre for x in all_essays if x.prompt_id in chosen}
        manifest.append({'size':size,'chosen_source_prompts':chosen,'genre_set':sorted(genre)})
        # Restrict the training input to prespecified source prompts and one target.
        restricted=[x for x in all_essays if x.prompt_id in chosen or x.prompt_id==a.target]
        for seed in [int(x) for x in a.seeds.split(',')]:
            fold_dir=output/f'count_{size}'/f'subset_{idx}'
            result=run_outer_fold(restricted,a.target,cfg,seed,str(fold_dir))
            rows.append({'source_count':size,'source_prompt_ids':'-'.join(chosen),
                         'genre_count':len(genre),'seed':seed,**result['metrics']})
(output/'source_subset_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
pd.DataFrame(rows).to_csv(output/'measured_source_subsets.csv',index=False)
print('Wrote measured_source_subsets.csv. All choices made without observing target test results.')
