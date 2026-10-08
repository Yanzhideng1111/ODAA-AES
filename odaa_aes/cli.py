from __future__ import annotations
import argparse
import copy
import itertools
import json
from pathlib import Path
import pandas as pd
from .config import load_config
from .data import read_essays, unique_prompts
from .train import (run_outer_fold, run_cross_corpus, load_checkpoint,
                    predict_records, make_tokenizer)
from .analysis import (summarize_runs, paired_prompt_cluster_bootstrap,
                       calibration_and_subgroups)


def parse():
    p=argparse.ArgumentParser(description='ODAA-AES reproducible source-only CLI')
    cmd=p.add_subparsers(dest='command',required=True)
    train=cmd.add_parser('train',help='Outer LOPO with source-only nested tuning')
    train.add_argument('--data',required=True);train.add_argument('--config',required=True)
    train.add_argument('--out',required=True);train.add_argument('--target',default='all')
    train.add_argument('--seeds',default=None,help='Comma separated seeds. Default 5 config seeds.')
    train.add_argument('--epochs',type=int,default=None)
    abl=cmd.add_parser('ablate',help='2x2x2 and optional conditioning matched controls')
    abl.add_argument('--data',required=True);abl.add_argument('--config',required=True)
    abl.add_argument('--out',required=True);abl.add_argument('--target',default='all')
    abl.add_argument('--seeds',default=None);abl.add_argument('--kind',choices=['factorial','conditioning'],default='factorial')
    cross=cmd.add_parser('cross-corpus',help='Frozen zero-target-adaptation cross-corpus eval')
    for name in ('source-data','source-config','target-data','target-config','out'):
        cross.add_argument('--'+name,required=True)
    cross.add_argument('--seed',type=int,default=42)
    pre=cmd.add_parser('predict',help='Apply frozen checkpoint, without target adaptation')
    for name in ('checkpoint','input','output'):
        pre.add_argument('--'+name,required=True)
    pre.add_argument('--target-config',help='Optional rubric specs for a different corpus')
    pre.add_argument('--no-labels',action='store_true',help='For blind prediction files without score')
    report=cmd.add_parser('report',help='Aggregate actual fold predictions')
    for name in ('root','config','out'):
        report.add_argument('--'+name,required=True)
    stats=cmd.add_parser('paired-stats',help='Paired prompt-cluster bootstrap')
    stats.add_argument('--a',required=True);stats.add_argument('--b',required=True)
    stats.add_argument('--out',required=True);stats.add_argument('--iterations',type=int,default=10000)
    groups=cmd.add_parser('groups',help='Score calibration, genre/length/region errors')
    groups.add_argument('--predictions',required=True)
    groups.add_argument('--config',required=True);groups.add_argument('--out',required=True)
    return p.parse_args()


def selected_seeds(cfg,seeds):
    if seeds:
        return [int(x) for x in seeds.split(',')]
    return [int(s) for s in cfg['training'].get('seeds',[cfg['training']['seed']])]


def run_cv(essays,cfg,out,target,seeds):
    targets=unique_prompts(essays) if target=='all' else [str(target)]
    for t in targets:
        if t not in unique_prompts(essays):
            raise ValueError(f'Target {t} not in dataset')
        for seed in seeds:
            row=run_outer_fold(essays,t,cfg,seed,out)
            print(json.dumps({'event':'finished_outer_fold',**row},ensure_ascii=False),flush=True)


def main():
    args=parse()
    if args.command in ('train','ablate'):
        cfg=load_config(args.config)
        if getattr(args,'epochs',None):cfg['training']['epochs']=args.epochs
        essays=read_essays(args.data,cfg)
        seeds=selected_seeds(cfg,args.seeds)
        if args.command=='train':
            run_cv(essays,cfg,args.out,args.target,seeds)
        else:
            if args.kind=='factorial':
                variants=list(itertools.product([False,True],repeat=3))
                for gp,oc,os in variants:
                    candidate=copy.deepcopy(cfg)
                    candidate['model'].update(gpaa=gp,ocaa=oc,ospl=os)
                    name=f'G{int(gp)}_C{int(oc)}_S{int(os)}'
                    print('ABLATION',name,flush=True)
                    run_cv(essays,candidate,str(Path(args.out)/name),args.target,seeds)
            else:
                controls=[('none',False,'ordinal'),('hard',True,'hard'),
                          ('nonordinal_soft',True,'nonordinal_soft'),
                          ('order_destroyed',True,'shuffled'),('ordinal',True,'ordinal')]
                for name,oc,condition in controls:
                    candidate=copy.deepcopy(cfg)
                    candidate['model'].update(gpaa=True,ospl=True,ocaa=oc,conditioning=condition)
                    print('CONDITIONING',name,flush=True)
                    run_cv(essays,candidate,str(Path(args.out)/name),args.target,seeds)
    elif args.command=='cross-corpus':
        c1=load_config(args.source_config);c2=load_config(args.target_config)
        s=read_essays(args.source_data,c1);t=read_essays(args.target_data,c2)
        res=run_cross_corpus(s,t,c1,c2,args.seed,args.out)
        print(json.dumps(res,indent=2))
    elif args.command=='predict':
        model,cfg,transform,device=load_checkpoint(args.checkpoint)
        if args.target_config:
            target_cfg=load_config(args.target_config)
            cfg['data']=target_cfg['data']
        essays=read_essays(args.input,cfg,require_labels=not args.no_labels)
        tokenizer=make_tokenizer(cfg)
        res=predict_records(model,essays,tokenizer,cfg,transform,0,device)
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(res).to_csv(args.output,index=False)
        print(f'Predictions written to {args.output}')
    elif args.command=='report':
        print(json.dumps(summarize_runs(args.root,load_config(args.config),args.out),indent=2))
    elif args.command=='paired-stats':
        print(json.dumps(paired_prompt_cluster_bootstrap(args.a,args.b,args.out,args.iterations),indent=2))
    elif args.command=='groups':
        print(json.dumps(calibration_and_subgroups(args.predictions,load_config(args.config),args.out),indent=2))

if __name__=='__main__':
    main()
