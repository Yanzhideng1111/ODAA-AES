"""Benchmark inference parameters, GPU peak and latency, using measured runs."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from odaa_aes.train import load_checkpoint, make_tokenizer, device_batch
from odaa_aes.data import read_essays, SegmentedEssayDataset, make_collate_fn
from torch.utils.data import DataLoader
p=argparse.ArgumentParser()
p.add_argument('--checkpoint',required=True)
p.add_argument('--data',required=True)
p.add_argument('--out',required=True)
p.add_argument('--repeats',type=int,default=4)
p.add_argument('--max-examples',type=int,default=64)
a=p.parse_args()
model,cfg,_,device=load_checkpoint(a.checkpoint)
tok=make_tokenizer(cfg)
records=read_essays(a.data,cfg,require_labels=False)[:a.max_examples]
loader=DataLoader(SegmentedEssayDataset(records,tok,cfg),batch_size=cfg['training']['eval_batch_size'],
                  collate_fn=make_collate_fn(tok))
# Excludes tokenization/disk I/O. Includes only model forward pass.
batches=[device_batch(b,device) for b in loader]
with torch.no_grad():
    for b in batches[:min(2,len(batches))]:model(b['input_ids'],b['attention_mask'],b['segment_mask'],False)
if device.type=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
times=[]
with torch.no_grad():
    for r in range(a.repeats):
        start=time.perf_counter()
        for b in batches:model(b['input_ids'],b['attention_mask'],b['segment_mask'],False)
        if device.type=='cuda':torch.cuda.synchronize()
        times.append(time.perf_counter()-start)
train_params=sum(p.numel() for p in model.parameters())
inference_params=sum(p.numel() for n,p in model.named_parameters() if not n.startswith(('gpaa_head','ocaa_heads','ordinal_head')))
res={'samples':len(records),'repeats':a.repeats,'training_parameter_count':train_params,
     'inference_parameter_count':inference_params,'latency_ms_per_essay':np.median(times)*1000/max(1,len(records)),
     'device':str(device),'gpu_peak_gb':torch.cuda.max_memory_allocated()/1e9 if device.type=='cuda' else None,
     'includes_tokenization':False,'note':'Actual saved model, measurement of inference only; not training GPU peak.'}
Path(a.out).parent.mkdir(parents=True,exist_ok=True)
Path(a.out).write_text(json.dumps(res,indent=2),encoding='utf-8')
print(json.dumps(res,indent=2))
