from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from .metrics import per_prompt_metrics
from .data import spec_for


def summarize_runs(root: str, config: dict, output: str):
    """Read REAL fold prediction files and compute per-prompt metrics/seed variability."""
    paths = sorted(Path(root).rglob('predictions.csv'))
    if not paths:
        raise FileNotFoundError(f'No predictions.csv found under {root}')
    rows = []
    for fp in paths:
        metadata = json.loads((fp.parent/'metadata.json').read_text(encoding='utf-8'))
        df = pd.read_csv(fp)
        for prompt, g in df.groupby('prompt_id'):
            met = per_prompt_metrics(g.human_score.to_numpy(),g.predicted_normalized.to_numpy(),
                                     spec_for(config,str(prompt)),config['training']['rounding'])
            rows.append({'target':str(prompt),'seed':metadata['seed'], **met,'run_dir':str(fp.parent)})
    all_runs = pd.DataFrame(rows)
    output = Path(output);output.mkdir(parents=True, exist_ok=True)
    all_runs.to_csv(output/'per_prompt_seed_metrics.csv',index=False)
    prompt = all_runs.groupby('target',as_index=False).agg(qwk=('qwk','mean'),
                         qwk_seed_sd=('qwk','std'),mae=('mae','mean'),rmse=('rmse','mean'),
                         spearman=('spearman','mean'),n=('n','mean'),seeds=('seed','nunique'))
    prompt.to_csv(output/'per_prompt_summary.csv',index=False)
    overall = {'prompt_macro_qwk':float(prompt.qwk.mean()),
               'prompt_macro_mae':float(prompt.mae.mean()),
               'prompt_macro_rmse':float(prompt.rmse.mean()),
               'prompt_macro_spearman':float(prompt.spearman.mean()),
               'target_prompts':len(prompt),'runs':len(all_runs),
               'note':'Seed repeats are not independent target domains. No target folds used for tuning.'}
    (output/'overall.json').write_text(json.dumps(overall,indent=2),encoding='utf-8')
    return overall


def paired_prompt_cluster_bootstrap(a_csv: str, b_csv: str, output: str,
                                    iterations: int = 10000, seed: int = 2026):
    """Compare saved per-prompt/seed metrics from two different REAL model runs.

    Paired by (held-out prompt, seed), averages paired seeds within prompt,
    then resamples *prompts* as the independence unit.
    """
    a = pd.read_csv(a_csv)[['target','seed','qwk']].rename(columns={'qwk':'a'})
    b = pd.read_csv(b_csv)[['target','seed','qwk']].rename(columns={'qwk':'b'})
    data = a.merge(b,on=['target','seed'],how='inner',validate='one_to_one')
    if len(data) == 0:
        raise ValueError('No paired (prompt, seed) records')
    paired = data.assign(delta=lambda x: x.a-x.b).groupby('target').delta.mean()
    if len(paired) < 2:
        raise ValueError('Need at least two held-out prompts for prompt-cluster interval')
    arr = paired.to_numpy(); rng = np.random.default_rng(seed)
    draws = rng.choice(arr,size=(iterations,len(arr)),replace=True).mean(axis=1)
    ci = np.percentile(draws,[2.5,97.5])
    p = float(wilcoxon(arr,alternative='two-sided').pvalue)
    summary = {'matched_prompts':len(paired),'paired_prompt_effect':float(arr.mean()),
               '95pct_prompt_cluster_bootstrap_ci':ci.tolist(), 'wilcoxon_p_value':p,
               'bootstrap_samples':iterations,
               'note':'Prompt-level resampling; does not claim independent seeds as additional domains.'}
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary


def calibration_and_subgroups(predictions_csv: str, config: dict, output: str,
                              bins: int = 5):
    df = pd.read_csv(predictions_csv)
    if 'human_score' not in df:
        raise ValueError('Group analysis requires evaluation human scores')
    df['true_norm'] = [float((r.human_score-spec_for(config,str(r.prompt_id))['min']) /
                       (spec_for(config,str(r.prompt_id))['max']-spec_for(config,str(r.prompt_id))['min']))
                       for r in df.itertuples()]
    df['abs_error'] = (df['predicted_normalized']-df['true_norm']).abs()
    df['bin'] = pd.cut(df['predicted_normalized'], bins=np.linspace(0,1,bins+1),
                       labels=False, include_lowest=True)
    cal = df.groupby('bin',dropna=False).agg(n=('essay_id','count'),
        predicted=('predicted_normalized','mean'),human=('true_norm','mean')).reset_index()
    cal['absolute_gap'] = (cal['predicted']-cal['human']).abs()
    by_genre = df.groupby('genre').agg(n=('essay_id','count'),mae=('abs_error','mean')).reset_index()
    df['length_group'] = pd.qcut(df['num_words'],q=3,duplicates='drop').astype(str)
    by_length = df.groupby('length_group').agg(n=('essay_id','count'),mae=('abs_error','mean')).reset_index()
    df['region'] = pd.cut(df['true_norm'],bins=[-1e-9,1/3,2/3,1.000001],labels=['low','middle','high'])
    by_region = df.groupby('region',observed=True).agg(n=('essay_id','count'),mae=('abs_error','mean')).reset_index()
    output = Path(output);output.mkdir(parents=True,exist_ok=True)
    for name,data in [('calibration',cal),('genre_errors',by_genre),
                      ('length_errors',by_length),('score_region_errors',by_region)]:
        data.to_csv(output/f'{name}.csv',index=False)
    cases = df.sort_values('abs_error',ascending=False).head(30)
    cases[['essay_id','prompt_id','genre','num_words','human_score',
           'predicted_rubric_score','predicted_normalized','abs_error']].to_csv(
        output/'largest_errors.csv',index=False)
    return {'calibration_bins':len(cal), 'subgroups':len(by_genre)+len(by_length)+len(by_region)}
