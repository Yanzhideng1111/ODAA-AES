from __future__ import annotations
import math
import numpy as np
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, mean_squared_error, accuracy_score
from scipy.stats import spearmanr
from .data import denormalize, normalize


def round_half_away_from_zero(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.floor(np.abs(x) + .5)


def to_rubric_index(norm_predictions: np.ndarray, spec: dict, rounding: str = 'half_away_from_zero') -> np.ndarray:
    """Map normalized [0,1] score to indices of published rubric categories."""
    raw = np.array([denormalize(y, spec) for y in np.clip(norm_predictions, 0, 1)])
    step, start = float(spec['step']), float(spec['min'])
    index_float = (raw - start)/step
    if rounding == 'half_away_from_zero':
        ind = round_half_away_from_zero(index_float)
    elif rounding == 'floor':
        ind = np.floor(index_float)
    elif rounding == 'ceil':
        ind = np.ceil(index_float)
    else:
        raise ValueError(rounding)
    n = int(round((spec['max'] - spec['min']) / step))
    return np.clip(ind.astype(int), 0, n)


def scores_from_normalized(pred: np.ndarray, spec: dict, rounding: str = 'half_away_from_zero') -> np.ndarray:
    return spec['min'] + spec['step'] * to_rubric_index(pred, spec, rounding)


def per_prompt_metrics(true_score, pred_score, spec, rounding='half_away_from_zero'):
    """true_score are RAW ratings; pred_score are rubric-relative normalized scores."""
    tr = np.asarray(true_score, dtype=float)
    pred = np.asarray(pred_score, dtype=float)
    if len(tr) != len(pred) or not len(tr):
        raise ValueError('Mismatched labels/predictions')
    tn = (tr-spec['min']) / (spec['max']-spec['min'])
    true_indices = to_rubric_index(tn, spec, rounding)
    pred_indices = to_rubric_index(pred, spec, rounding)
    nclass = int(round((spec['max']-spec['min'])/spec['step']))+1
    labels = list(range(nclass))
    # QWK may be undefined for constant labels (one class always predicted); do not silently invent it.
    qwk = cohen_kappa_score(true_indices, pred_indices, labels=labels, weights='quadratic')
    sp = (float(spearmanr(tn, pred).statistic) if len(tn) > 1 and np.std(tn) > 1e-12 and np.std(pred) > 1e-12 else float('nan'))
    return {'n': len(tr), 'qwk': float(qwk), 'mae': float(mean_absolute_error(tn,pred)),
            'rmse': float(np.sqrt(mean_squared_error(tn,pred))), 'spearman': float(sp),
            'accuracy': float(accuracy_score(true_indices,pred_indices))}


def aggregate_prompt_metrics(groups: dict[str, dict]) -> dict:
    """Macro average across held-out prompts, not micro average across essays."""
    keys = ('qwk', 'mae', 'rmse', 'spearman', 'accuracy')
    out = {}
    for x in keys:
        values = np.asarray([v[x] for v in groups.values()], dtype=float)
        out[x] = float(np.nanmean(values)) if np.isfinite(values).any() else float('nan')
    out['prompt_count'] = len(groups)
    return out
