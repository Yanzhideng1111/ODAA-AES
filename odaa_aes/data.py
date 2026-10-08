from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Essay:
    essay_id: str
    prompt_id: str
    text: str
    score: float | None
    genre: str = 'unknown'


def spec_for(cfg: dict, prompt: str) -> dict:
    d = cfg['data']; specs = d.get('score_specs', {})
    s = specs.get(str(prompt), d.get('default_score_spec'))
    if s is None:
        raise ValueError(f'No PUBLISHED rubric score range for prompt={prompt}. Add to config; do not infer from held-out labels.')
    s = dict(s)
    if s['max'] <= s['min'] or s['step'] <= 0:
        raise ValueError(f'Invalid score spec {s}')
    return s


def normalize(score: float, spec: dict) -> float:
    return (float(score) - float(spec['min'])) / (float(spec['max']) - float(spec['min']))


def denormalize(score: float, spec: dict) -> float:
    return float(spec['min']) + float(score) * (float(spec['max']) - float(spec['min']))


def read_essays(path: str, cfg: dict, require_labels: bool = True) -> list[Essay]:
    d = cfg['data']; sep = '\t' if d.get('sep') == '\\t' else d.get('sep', ',')
    frame = pd.read_csv(path, sep=sep, encoding='utf-8-sig', keep_default_na=False)
    needed = [d['id_col'], d['prompt_col'], d['text_col']]
    if require_labels:
        needed.append(d['score_col'])
    missing = [x for x in needed if x not in frame.columns]
    if missing:
        raise ValueError(f'Missing columns {missing}. Available columns: {list(frame.columns)}')
    records = []
    seen = set()
    for _, r in frame.iterrows():
        pid = str(r[d['prompt_col']]).strip()
        spec = spec_for(cfg, pid)  # public rubric, never inferred from test distribution
        essay_id = str(r[d['id_col']]).strip()
        if essay_id in seen:
            raise ValueError(f'Duplicate essay_id in dataset: {essay_id}')
        seen.add(essay_id)
        text = str(r[d['text_col']]).strip()
        if not text:
            raise ValueError(f'Empty essay text for {essay_id}')
        raw = r.get(d['score_col'], None)
        y = None
        if require_labels:
            key = str(raw).strip()
            raw = d.get('label_mapping', {}).get(key, raw)
            try:
                y = float(raw)
            except (TypeError, ValueError) as ex:
                raise ValueError(f'Invalid score {raw} for essay {essay_id}') from ex
            if not (spec['min'] - 1e-6 <= y <= spec['max'] + 1e-6):
                raise ValueError(f'Score {y} outside published rubric for {pid}: {spec}')
        genre = str(r[d['genre_col']]) if d.get('genre_col') and d['genre_col'] in frame.columns else spec.get('genre', 'unknown')
        records.append(Essay(essay_id=essay_id, prompt_id=pid, text=text, score=y, genre=genre))
    if not records:
        raise ValueError('No essays were loaded')
    return records


def unique_prompts(essays: list[Essay]) -> list[str]:
    return sorted({r.prompt_id for r in essays})


def outer_split(essays: list[Essay], target: str) -> tuple[list[Essay], list[Essay]]:
    sources = [r for r in essays if r.prompt_id != str(target)]
    targets = [r for r in essays if r.prompt_id == str(target)]
    if not sources or not targets:
        raise ValueError('Outer target/source split is empty')
    assert {r.essay_id for r in sources}.isdisjoint({r.essay_id for r in targets})
    return sources, targets


class SegmentedEssayDataset(Dataset):
    def __init__(self, essays: list[Essay], tokenizer, cfg: dict, label_transform=None):
        self.essays = essays
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.label_transform = label_transform
        m = cfg['model']
        self.max_tokens = int(m['max_tokens'])
        self.max_segments = int(m['max_segments'])
        self.overlap = int(m['overlap'])
        self.truncation = m.get('truncation', 'first')

    def __len__(self):
        return len(self.essays)

    def _tokenize(self, text: str):
        tok = self.tokenizer
        ids = tok(text, add_special_tokens=False, truncation=False)['input_ids']
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        special = tok.num_special_tokens_to_add(pair=False)
        window = self.max_tokens - special
        if window <= 0 or self.overlap >= window:
            raise ValueError('Invalid token window/overlap')
        step = window - self.overlap
        starts = list(range(0, max(1, len(ids)), step))
        if len(starts) > self.max_segments:
            if self.truncation == 'head_tail':
                a = (self.max_segments + 1) // 2
                starts = starts[:a] + starts[-(self.max_segments-a):]
            else:
                starts = starts[:self.max_segments]
        segments = []
        for start in starts:
            chunk = ids[start:start + window]
            full = tok.build_inputs_with_special_tokens(chunk)
            segments.append(full)
            if start + window >= len(ids):
                break
        if not segments:
            segments = [tok.build_inputs_with_special_tokens([])]
        return segments

    def __getitem__(self, idx):
        r = self.essays[idx]; s = spec_for(self.cfg, r.prompt_id)
        y = None if r.score is None else normalize(r.score, s)
        if y is not None and self.label_transform is not None:
            y = float(self.label_transform.transform(np.array([y]))[0])
        return {'essay_id': r.essay_id, 'prompt_id': r.prompt_id, 'genre': r.genre,
                'input_ids': self._tokenize(r.text), 'score': y,
                'raw_score': r.score, 'num_words': len(r.text.split())}


def make_collate_fn(tokenizer):
    pad = tokenizer.pad_token_id
    if pad is None:
        raise ValueError('Tokenizer must have a pad_token_id')

    def collate(rows):
        b = len(rows)
        segments = max(len(r['input_ids']) for r in rows)
        length = max(len(x) for r in rows for x in r['input_ids'])
        inputs = torch.full((b, segments, length), int(pad), dtype=torch.long)
        masks = torch.zeros((b, segments, length), dtype=torch.long)
        valid = torch.zeros((b, segments), dtype=torch.bool)
        for i, r in enumerate(rows):
            for j, seg in enumerate(r['input_ids']):
                inputs[i, j, :len(seg)] = torch.tensor(seg, dtype=torch.long)
                masks[i, j, :len(seg)] = 1
                valid[i, j] = True
        if all(r['score'] is not None for r in rows):
            score = torch.tensor([r['score'] for r in rows], dtype=torch.float32)
        else:
            score = None
        return {'input_ids': inputs, 'attention_mask': masks, 'segment_mask': valid,
                'score': score, 'essay_id': [r['essay_id'] for r in rows],
                'prompt_id': [r['prompt_id'] for r in rows],
                'genre': [r['genre'] for r in rows],
                'raw_score': [r['raw_score'] for r in rows],
                'num_words': [r['num_words'] for r in rows]}
    return collate


class SourceScoreTransform:
    """Fits only on source-prompt labels and never inspects held-out target labels."""
    def __init__(self, kind: str = 'minmax'):
        self.kind = kind
        self.x = None

    def fit(self, source_scores: np.ndarray):
        self.x = np.sort(np.asarray(source_scores, dtype=float))
        if not len(self.x):
            raise ValueError('Empty source labels')
        self.mean = float(np.mean(self.x)); self.std = max(float(np.std(self.x)), 1e-6)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.kind == 'minmax':
            return x
        if self.kind == 'source_percentile':
            return np.searchsorted(self.x, x, side='right') / len(self.x)
        if self.kind == 'source_zscore':
            # bounded monotonic z-score transform; inverse uses only source summary statistics
            return 1 / (1 + np.exp(-((x-self.mean)/self.std)))
        if self.kind == 'source_piecewise':
            # prespecified monotonic curve with exact target-free inverse
            return np.where(x <= 0.5, x*x*2, 1-(1-x)**2*2)
        raise ValueError(self.kind)

    def inverse(self, z: np.ndarray) -> np.ndarray:
        z = np.clip(np.asarray(z, dtype=float), 0, 1)
        if self.kind == 'minmax':
            return z
        if self.kind == 'source_percentile':
            return np.quantile(self.x, z)
        if self.kind == 'source_zscore':
            return np.clip(self.mean + self.std * np.log(np.clip(z, 1e-6, 1-1e-6) / np.clip(1-z, 1e-6, 1-1e-6)), 0, 1)
        if self.kind == 'source_piecewise':
            return np.where(z <= 0.5, np.sqrt(z/2), 1-np.sqrt((1-z)/2))
        raise ValueError(self.kind)

    def dump(self) -> dict:
        return {'kind': self.kind, 'source_scores': self.x.tolist() if self.x is not None else None}

    @classmethod
    def restore(cls, value: dict):
        x = cls(value['kind'])
        if value.get('source_scores') is not None:
            x.fit(np.array(value['source_scores']))
        return x
