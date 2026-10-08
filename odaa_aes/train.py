from __future__ import annotations
import copy
import hashlib
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from .config import dump_json
from .data import (Essay, SegmentedEssayDataset, SourceScoreTransform,
                   make_collate_fn, normalize, spec_for, unique_prompts)
from .model import ODAA_AES, compute_losses, source_anchors
from .metrics import per_prompt_metrics, aggregate_prompt_metrics, scores_from_normalized


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_tokenizer(cfg: dict):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(cfg['model']['backbone'], use_fast=True)


def score_transform_for(source_records: list[Essay], cfg: dict) -> SourceScoreTransform:
    ys = [normalize(x.score, spec_for(cfg, x.prompt_id)) for x in source_records]
    return SourceScoreTransform(cfg['training'].get('score_transform', 'minmax')).fit(np.array(ys))


def make_model(cfg: dict, source: list[Essay], transform: SourceScoreTransform,
               encoder=None) -> tuple[ODAA_AES, dict[str, int]]:
    domains = {p: i for i, p in enumerate(unique_prompts(source))}
    norm = np.array([normalize(x.score, spec_for(cfg, x.prompt_id)) for x in source])
    transformed = transform.transform(norm)
    mc = cfg['model']
    anchors = source_anchors(transformed, mc.get('anchor_strategy', 'uniform'),
                             int(mc['num_anchors']), mc.get('anchors'))
    model = ODAA_AES(cfg, num_domains=len(domains), encoder=encoder, anchors=anchors)
    return model, domains


def make_loader(source, tokenizer, cfg, seed: int, training: bool, transform=None):
    dset = SegmentedEssayDataset(source, tokenizer, cfg, label_transform=transform)
    tr = cfg['training']
    batch_size = tr['batch_size'] if training else tr.get('eval_batch_size', tr['batch_size'])
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(dset, batch_size=batch_size, shuffle=training,
                      num_workers=int(tr.get('num_workers',0)),
                      collate_fn=make_collate_fn(tokenizer), generator=gen,
                      pin_memory=torch.cuda.is_available())


def device_batch(batch: dict, device: torch.device):
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k,v in batch.items()}


def configure_optimizer(model: ODAA_AES, cfg: dict):
    t = cfg['training']
    groups = []
    for group_name, parameters, lr in [
        ('pretrained', model.encoder.named_parameters(), float(t['lr_backbone'])),
        ('new', ((n,p) for n,p in model.named_parameters() if not n.startswith('encoder.')),
         float(t['lr_new']))]:
        wd = [p for n,p in parameters if p.requires_grad and not (n.endswith('.bias') or 'norm' in n.lower())]
        # Generate second pass because named_parameters generator was exhausted.
        src = model.encoder.named_parameters() if group_name == 'pretrained' else (
            (n,p) for n,p in model.named_parameters() if not n.startswith('encoder.'))
        no_wd = [p for n,p in src if p.requires_grad and (n.endswith('.bias') or 'norm' in n.lower())]
        groups += [{'params': wd, 'lr':lr, 'weight_decay':float(t['weight_decay'])},
                   {'params': no_wd, 'lr':lr, 'weight_decay':0.0}]
    return torch.optim.AdamW(groups)


def configure_scheduler(optimizer, cfg, total_steps: int):
    warmup = int(round(total_steps * float(cfg['training']['warmup_ratio'])))
    def factor(step):
        if step < warmup:
            return float(step+1) / max(1, warmup)
        return max(0., float(total_steps-step) / max(1, total_steps-warmup))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


@torch.no_grad()
def predict_records(model, records: list[Essay], tokenizer, cfg: dict,
                    transform: SourceScoreTransform, seed: int, device):
    model.eval()
    loader = make_loader(records, tokenizer, cfg, seed, False, transform=None)
    preds = []
    for b in loader:
        b = device_batch(b, device)
        out = model(b['input_ids'], b['attention_mask'], b['segment_mask'], adversarial=False)
        z = out['score'].cpu().numpy()
        norm_preds = np.clip(transform.inverse(z), 0, 1)
        for i, yp in enumerate(norm_preds):
            spec = spec_for(cfg, b['prompt_id'][i])
            raw_pred = float(scores_from_normalized(np.array([yp]), spec, cfg['training']['rounding'])[0])
            row = {'essay_id':b['essay_id'][i], 'prompt_id':b['prompt_id'][i],
                   'genre':b['genre'][i], 'num_words':b['num_words'][i],
                   'predicted_normalized': float(yp), 'predicted_rubric_score': raw_pred}
            if b['raw_score'][i] is not None:
                row['human_score'] = float(b['raw_score'][i])
            preds.append(row)
    return preds


def metrics_by_prompt(rows, cfg):
    frame = pd.DataFrame(rows)
    if 'human_score' not in frame:
        return {}
    groups = {}
    for p, frame_p in frame.groupby('prompt_id'):
        groups[str(p)] = per_prompt_metrics(frame_p.human_score.to_numpy(),
            frame_p.predicted_normalized.to_numpy(), spec_for(cfg, str(p)),
            cfg['training']['rounding'])
    return groups


def train_once(train_records: list[Essay], cfg: dict, seed: int, epochs: int,
               validation: list[Essay] | None = None, output: Path | None = None,
               tokenizer=None, encoder=None):
    if not train_records:
        raise ValueError('Training source has no essays')
    if output is not None:
        output = Path(output); output.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    transform = score_transform_for(train_records, cfg)
    tokenizer = tokenizer or make_tokenizer(cfg)
    model, prompt_to_domain = make_model(cfg, train_records, transform, encoder=encoder)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    train_loader = make_loader(train_records, tokenizer, cfg, seed, True, transform)
    optimizer = configure_optimizer(model, cfg)
    accumulation = int(cfg['training']['accumulation_steps'])
    total_updates = math.ceil(len(train_loader)/accumulation)*epochs
    scheduler = configure_scheduler(optimizer, cfg, total_updates)
    use_amp = bool(cfg['training'].get('amp',True)) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda',enabled=use_amp)
    log, best, best_epoch = [], float('-inf'), 1
    assert validation is None or ({r.essay_id for r in train_records}.isdisjoint({r.essay_id for r in validation}))
    for epoch in range(1, epochs+1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        cum_loss = 0.; batches = len(train_loader)
        for ib, b in enumerate(train_loader):
            b = device_batch(b, device)
            domains = torch.tensor([prompt_to_domain[p] for p in b['prompt_id']], dtype=torch.long, device=device)
            progress = ((epoch-1) * batches + ib) / max(1, epochs*batches - 1)
            group_size = min(accumulation, batches - (ib//accumulation)*accumulation)
            with torch.autocast(device_type='cuda', enabled=use_amp) if use_amp else nullcontext():
                out = model(b['input_ids'], b['attention_mask'], b['segment_mask'], adversarial=True)
                loss, _ = compute_losses(out, b['score'], domains, model, cfg, progress)
            cum_loss += float(loss.detach().cpu())
            scaler.scale(loss/group_size).backward()
            if (ib+1) % accumulation == 0 or (ib+1) == batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg['training']['grad_clip']))
                scaler.step(optimizer); scaler.update()
                scheduler.step(); optimizer.zero_grad(set_to_none=True)
        trace = {'epoch': epoch, 'train_loss': cum_loss / max(1,batches)}
        if validation:
            pred = predict_records(model, validation, tokenizer, cfg, transform, seed, device)
            vals = metrics_by_prompt(pred, cfg)
            qwk = aggregate_prompt_metrics(vals)['qwk']
            trace['validation_qwk'] = qwk
            if np.isfinite(qwk) and qwk > best:
                best = qwk; best_epoch = epoch
        log.append(trace)
        print(json.dumps({'event':'epoch', **trace, 'source_prompts':list(prompt_to_domain)},ensure_ascii=False), flush=True)
    if output:
        torch.save({k: v.detach().cpu() for k,v in model.state_dict().items()},output/'model_state.pt')
        dump_json({'config':cfg,'source_prompt_to_domain':prompt_to_domain,
                   'score_transform':transform.dump(),'seed':seed,'trained_epochs':epochs,
                   'source_ids_sha256':hashlib.sha256('\n'.join(sorted(r.essay_id for r in train_records)).encode()).hexdigest(),
                   'source_size': len(train_records), 'version':'independent-reconstruction-v0.1'}, output/'metadata.json')
        pd.DataFrame(log).to_csv(output/'train_history.csv', index=False)
    return {'model':model, 'transform':transform, 'tokenizer':tokenizer, 'device':device,
            'best_epoch':best_epoch, 'best_qwk':best, 'history':log, 'prompt_to_domain':prompt_to_domain}


def nested_source_select(source_records: list[Essay], cfg: dict, seed: int, tokenizer=None):
    source_prompts = unique_prompts(source_records)
    if len(source_prompts) < 2:
        # A single source cannot support leave-one-SOURCE-PROMPT-out. This is a
        # separate declared policy, not equivalent to multi-source inner CV.
        rng = np.random.default_rng(seed)
        ids = rng.permutation(len(source_records))
        cut = max(1, min(len(ids)-1, int(.8*len(ids))))
        subtrain = [source_records[i] for i in ids[:cut]]
        val = [source_records[i] for i in ids[cut:]]
        inner_folds = [('within_source_80_20',subtrain,val)]
    else:
        inner_folds = [(p, [r for r in source_records if r.prompt_id != p],
                        [r for r in source_records if r.prompt_id == p]) for p in source_prompts]
    max_folds = cfg['training'].get('inner_max_folds')
    if max_folds is not None:
        inner_folds = inner_folds[:int(max_folds)]
    grid = cfg['training'].get('hyperparam_grid') or [{}]
    report = []; tokenizer = tokenizer or make_tokenizer(cfg)
    for config_index, parameters in enumerate(grid):
        trial_cfg = copy.deepcopy(cfg); trial_cfg['training'].update(parameters)
        vals, epochs = [], []
        for fold, inner_train, inner_val in inner_folds:
            print(f'INNER seed={seed} trial={config_index} validation_source={fold}', flush=True)
            res = train_once(inner_train, trial_cfg, seed, int(trial_cfg['training']['epochs']),
                             validation=inner_val, tokenizer=tokenizer)
            vals.append(res['best_qwk'])
            epochs.append(res['best_epoch'])
            del res['model']
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        result = {'trial':config_index, 'params':parameters,'inner_folds':[f[0] for f in inner_folds],
                  'best_qwks':vals, 'best_epochs':epochs,
                  'mean_inner_qwk':float(np.nanmean(vals)),
                  'median_best_epoch':int(max(1,round(float(np.median(epochs)))))}
        report.append(result)
    chosen = max(report, key=lambda r: r['mean_inner_qwk'] if np.isfinite(r['mean_inner_qwk']) else float('-inf'))
    final_cfg = copy.deepcopy(cfg); final_cfg['training'].update(chosen['params'])
    return final_cfg, int(chosen['median_best_epoch']), report


def run_outer_fold(all_essays: list[Essay], target: str, cfg: dict, seed: int, out_root: str,
                   tokenizer=None):
    from .data import outer_split
    source, holdout = outer_split(all_essays, target)
    # Only source is passed to model selection. The outer holdout is touched
    # only in predict_records after all hyperparameters/checkpoints are fixed.
    final_cfg, final_epochs, candidates = nested_source_select(source,cfg,seed,tokenizer=tokenizer)
    out = Path(out_root) / f'target_{target}' / f'seed_{seed}'
    run = train_once(source, final_cfg, seed, final_epochs, output=out, tokenizer=tokenizer)
    pred = predict_records(run['model'], holdout, run['tokenizer'], final_cfg,
                           run['transform'], seed, run['device'])
    pd.DataFrame(pred).to_csv(out/'predictions.csv',index=False)
    perf = metrics_by_prompt(pred, final_cfg)
    dump_json({'target':target,'seed':seed,'final_epochs':final_epochs,
               'validation_trials':candidates,'per_prompt_metrics':perf,
               'macro':aggregate_prompt_metrics(perf)}, out/'results.json')
    return {'target':target,'seed':seed,'metrics':perf[target], 'output':str(out)}


def run_cross_corpus(train_essays: list[Essay], test_essays: list[Essay],
                     train_cfg: dict, test_cfg: dict, seed: int, output: str):
    # Train/model select on SOURCE CORPUS only. Model frozen before opening
    # external corpus for evaluation. Rubric metadata of target corpus is known.
    final_cfg, final_epochs, report = nested_source_select(train_essays, train_cfg, seed)
    output = Path(output)
    run = train_once(train_essays, final_cfg, seed, final_epochs, output=output)
    # test_cfg supplies ONLY published target score metadata, not any fitted distribution.
    eval_cfg = copy.deepcopy(final_cfg)
    eval_cfg['data'] = copy.deepcopy(test_cfg['data'])
    rows = predict_records(run['model'], test_essays,run['tokenizer'],eval_cfg,
                           run['transform'],seed,run['device'])
    pd.DataFrame(rows).to_csv(output/'predictions_external.csv',index=False)
    pm = metrics_by_prompt(rows,eval_cfg)
    dump_json({'seed':seed,'final_epochs':final_epochs,'inner_trials':report,
               'target_macro':aggregate_prompt_metrics(pm),'target_per_prompt':pm},
               output/'external_results.json')
    return pm


def load_checkpoint(folder: str, device=None):
    folder = Path(folder)
    meta = json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    cfg = meta['config']
    num_domains = len(meta['source_prompt_to_domain'])
    st = torch.load(folder/'model_state.pt', map_location='cpu', weights_only=True)
    model = ODAA_AES(cfg,num_domains=num_domains,anchors=st['anchors'])
    model.load_state_dict(st,strict=True)
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    return model, cfg, SourceScoreTransform.restore(meta['score_transform']), device
