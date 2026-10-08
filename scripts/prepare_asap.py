"""Convert official ASAP training_set_rel3.tsv into config-compatible CSV.

Usage:
  python scripts/prepare_asap.py --input /path/to/training_set_rel3.tsv --output data/asap.csv
The script performs NO score imputation or train/test splitting.
"""
import argparse
from pathlib import Path
import pandas as pd
p=argparse.ArgumentParser()
p.add_argument('--input',required=True);p.add_argument('--output',required=True)
a=p.parse_args()
df=pd.read_csv(a.input,sep='\t',encoding='latin1')
columns=['essay_id','essay_set','essay','domain1_score']
miss=[c for c in columns if c not in df.columns]
if miss:raise ValueError(f'Missing ASAP columns: {miss}')
assert not df.essay_id.duplicated().any()
assert not df[columns].isna().any().any()
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
df[columns].to_csv(a.output,index=False,encoding='utf-8')
print(f'Prepared {len(df)} essays; no held-out target prompt was split or inspected for score statistics')
