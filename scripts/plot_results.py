"""Create sensitivity curves and calibration figures from measured output CSVs.

No paper performance numbers are hardcoded.
"""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
p=argparse.ArgumentParser()
p.add_argument('--csv',required=True)
p.add_argument('--x',required=True)
p.add_argument('--y',default='qwk')
p.add_argument('--out',required=True)
p.add_argument('--title',default='Measured ODAA-AES performance')
a=p.parse_args()
df=pd.read_csv(a.csv)
if a.x not in df or a.y not in df: raise ValueError(f'Expected {a.x} and {a.y} columns')
fig,ax=plt.subplots(figsize=(6.8,4.4))
ax.plot(df[a.x],df[a.y],marker='o',linewidth=2)
ax.set_xlabel(a.x);ax.set_ylabel(a.y.upper());ax.set_title(a.title)
ax.grid(True,alpha=.2);fig.tight_layout()
Path(a.out).parent.mkdir(parents=True,exist_ok=True)
fig.savefig(a.out,dpi=240);print(a.out)
