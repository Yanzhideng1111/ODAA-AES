from __future__ import annotations
import copy
import json
from pathlib import Path
from typing import Any
import yaml


def load_config(path: str | Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    validate(cfg)
    return cfg


def validate(cfg: dict) -> None:
    model = cfg['model']; tr = cfg['training']
    assert 0 <= model['overlap'] < model['max_tokens'] - 2, 'Invalid window overlap'
    assert model['num_anchors'] >= 2 and model['max_segments'] >= 1
    assert model['sigma'] > 0
    assert tr['batch_size'] > 0 and tr['accumulation_steps'] > 0
    assert tr['epochs'] > 0
    assert model['conditioning'] in {'ordinal', 'hard', 'shuffled', 'nonordinal_soft'}
    assert tr.get('score_transform', 'minmax') in {'minmax', 'source_percentile', 'source_zscore', 'source_piecewise'}


def updated(cfg: dict, **model_kwargs: Any) -> dict:
    c = copy.deepcopy(cfg)
    c['model'].update(model_kwargs)
    validate(c)
    return c


def dump_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
