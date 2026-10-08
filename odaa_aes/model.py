from __future__ import annotations
import math
import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F


class _GradientReverse(Function):
    @staticmethod
    def forward(ctx, tensor, scale):
        ctx.scale = float(scale)
        return tensor.view_as(tensor)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.scale * grad, None


def grad_reverse(tensor: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return _GradientReverse.apply(tensor, scale)


def sigmoid_grl_schedule(progress: float) -> float:
    """Paper Eq. (14): lambda(p) = 2/(1+exp(-10p)) - 1."""
    progress = max(0., min(1., float(progress)))
    return 2. / (1. + math.exp(-10. * progress)) - 1.


class ODAA_AES(nn.Module):
    def __init__(self, cfg: dict, num_domains: int, encoder: nn.Module | None = None, anchors=None):
        super().__init__()
        self.cfg = cfg
        mc = cfg['model']
        self.num_domains = num_domains
        if encoder is None:
            try:
                from transformers import AutoModel
            except ImportError as e:
                raise RuntimeError('Install requirements.txt for Hugging Face AutoModel') from e
            encoder = AutoModel.from_pretrained(mc['backbone'])
        self.encoder = encoder
        encoder_dim = getattr(self.encoder.config, 'hidden_size', None)
        if encoder_dim is None:
            raise ValueError('Backbone must expose config.hidden_size')
        zdim = mc['hidden_size']
        self.segment_attention = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim), nn.Tanh(), nn.Linear(encoder_dim, 1)
        )
        self.proj = nn.Linear(encoder_dim, zdim)
        self.norm = nn.LayerNorm(zdim)
        self.dropout = nn.Dropout(float(mc['dropout']))
        self.score_head = nn.Linear(zdim, 1)
        self.ordinal_head = nn.Linear(zdim, 1)
        self.use_gpaa = bool(mc['gpaa']) and num_domains >= 2
        self.use_ocaa = bool(mc['ocaa']) and num_domains >= 2
        self.use_ospl = bool(mc['ospl'])
        self.num_anchors = int(mc['num_anchors'])
        self.sigma = float(mc['sigma'])
        if anchors is None:
            anchors = torch.linspace(0, 1, self.num_anchors)
        anchors = torch.as_tensor(anchors, dtype=torch.float32)
        if anchors.shape != (self.num_anchors,):
            raise ValueError(f'anchors expected [{self.num_anchors}], got {anchors.shape}')
        self.register_buffer('anchors', anchors)
        # Same discriminator capacity for hard/ordinal/nonordinal soft variants.
        def discriminator():
            return nn.Sequential(nn.Linear(zdim, zdim), nn.ReLU(),
                                 nn.Dropout(float(mc['dropout'])), nn.Linear(zdim, num_domains))
        self.gpaa_head = discriminator() if self.use_gpaa else None
        self.ocaa_heads = nn.ModuleList(discriminator() for _ in range(self.num_anchors)) if self.use_ocaa else nn.ModuleList()

    def forward(self, input_ids, attention_mask, segment_mask, adversarial: bool = True):
        b, s, length = input_ids.shape
        flat_mask = segment_mask.reshape(-1)
        if not bool(flat_mask.any()):
            raise ValueError('Empty segment mask')
        ids = input_ids.reshape(b*s, length)[flat_mask]
        mask = attention_mask.reshape(b*s, length)[flat_mask]
        states = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0, :]
        states_all = states.new_zeros((b*s, states.shape[-1]))
        states_all = states_all.index_copy(0, flat_mask.nonzero(as_tuple=True)[0], states)
        segment_states = states_all.reshape(b, s, -1)
        weights = self.segment_attention(segment_states).squeeze(-1)
        weights = weights.masked_fill(~segment_mask, torch.finfo(weights.dtype).min)
        weights = weights.softmax(-1)
        essay = torch.einsum('bs,bsh->bh', weights, segment_states)
        z = self.dropout(self.norm(self.proj(essay)))
        scores = torch.sigmoid(self.score_head(z).squeeze(-1))
        latent = self.ordinal_head(z).squeeze(-1)
        out = {'score': scores, 'latent': latent, 'z': z, 'segment_weights': weights}
        if adversarial and self.use_gpaa:
            out['gpaa_logits'] = self.gpaa_head(grad_reverse(z))
        if adversarial and self.use_ocaa:
            out['ocaa_logits'] = torch.stack([head(grad_reverse(z)) for head in self.ocaa_heads], dim=1)
        return out


def source_anchors(scores: list[float], strategy: str, k: int, fixed=None):
    import numpy as np
    y = np.asarray(scores, dtype=float)
    if not len(y):
        raise ValueError('Source anchor design requires source scores')
    if strategy == 'uniform':
        return np.linspace(0, 1, k).tolist()
    if strategy == 'quantile':
        return np.quantile(y, np.linspace(0, 1, k)).tolist()
    if strategy == 'density_adaptive':
        # Simple implementation assumption: quantiles of a Gaussian-smoothed source-score sample.
        # Do not claim equivalence to an unspecified published density-adaptive solver.
        return np.quantile(y, np.linspace(.025, .975, k)).tolist()
    if strategy == 'fixed_nonuniform':
        if fixed is None:
            raise ValueError('fixed_nonuniform requires model.anchors')
        return fixed
    raise ValueError(strategy)


def conditional_weights(y: torch.Tensor, anchors: torch.Tensor, sigma: float,
                        strategy: str = 'ordinal', generator=None):
    """Memberships stop gradients at human SOURCE labels.

    shuffled: shuffles the y-to-essay association *within the batch*, preserving
    the observed source-score marginal and membership-entropy distribution.
    nonordinal_soft: draws uniform random rank placements independently of y.
    No global anchor renaming is used (which would be an equivalent model).
    """
    if strategy == 'shuffled':
        perm = torch.randperm(len(y), device=y.device, generator=generator)
        y = y[perm]
    elif strategy == 'nonordinal_soft':
        y = torch.rand(y.shape, device=y.device, generator=generator)
    if strategy == 'hard':
        index = (y[:, None] - anchors[None, :]).abs().argmin(dim=-1)
        return F.one_hot(index, len(anchors)).float()
    if strategy not in {'ordinal', 'shuffled', 'nonordinal_soft'}:
        raise ValueError(strategy)
    logits = -((y[:, None] - anchors[None, :]) ** 2) / (2 * sigma**2)
    return F.softmax(logits, dim=-1)


def sample_pairs(y: torch.Tensor, domains: torch.Tensor, mode: str, max_pairs: int,
                 generator=None):
    b = len(y)
    a, c = torch.triu_indices(b, b, offset=1, device=y.device)
    if len(a) <= max_pairs or max_pairs <= 0:
        return a, c
    def draw(aa, bb, n):
        if n <= 0 or not len(aa):
            return aa[:0], bb[:0]
        indices = torch.randperm(len(aa), device=y.device, generator=generator)[:n]
        return aa[indices], bb[indices]
    if mode == 'balanced':
        cross = domains[a] != domains[c]
        ac, bc = a[cross], c[cross]
        ai, bi = a[~cross], c[~cross]
        nc = min(max_pairs // 2, len(ac)); ni = min(max_pairs-nc, len(ai))
        if nc + ni < max_pairs:
            nc = min(len(ac), max_pairs-ni)
            ni = min(len(ai), max_pairs-nc)
        xa, xb = draw(ac, bc, nc)
        ya, yb = draw(ai, bi, ni)
        return torch.cat([xa, ya]), torch.cat([xb, yb])
    if mode == 'all':
        return draw(a, c, max_pairs)
    raise ValueError(mode)


def ordinal_loss(latent: torch.Tensor, y: torch.Tensor, domains: torch.Tensor,
                 margin: float = .5, eta: float = 1., pair_mode: str = 'balanced',
                 max_pairs: int = 128, min_diff: float = 1e-6, ties_in_distance: bool = True,
                 generator=None):
    i, j = sample_pairs(y, domains, pair_mode, max_pairs, generator)
    zero = latent.sum() * 0
    if len(i) == 0:
        return zero, {'rank': zero.detach(), 'distance': zero.detach(), 'pairs': 0}
    yd = y[i] - y[j]
    ud = latent[i] - latent[j]
    ordered = yd.abs() > min_diff
    rank = F.relu(margin * yd[ordered].abs() - yd[ordered].sign() * ud[ordered]).mean() if ordered.any() else zero
    if not ties_in_distance:
        yd = yd[ordered]; ud = ud[ordered]
    dist = F.smooth_l1_loss(ud, yd) if len(yd) else zero
    return rank + eta * dist, {'rank': rank.detach(), 'distance': dist.detach(), 'pairs': len(i)}


def compute_losses(out: dict, target_scores: torch.Tensor, domain_ids: torch.Tensor,
                   model: ODAA_AES, cfg: dict, progress: float = 1.0, generator=None):
    tr = cfg['training']; mc = cfg['model']
    score = F.huber_loss(out['score'], target_scores, delta=float(tr.get('huber_delta', 1.0)))
    zero = score * 0
    gpaa = F.cross_entropy(out['gpaa_logits'], domain_ids) if 'gpaa_logits' in out else zero
    ocaa = zero
    if 'ocaa_logits' in out:
        q = conditional_weights(target_scores.detach(), model.anchors, model.sigma,
                                mc.get('conditioning', 'ordinal'), generator=generator)
        b, k, nc = out['ocaa_logits'].shape
        ce = F.cross_entropy(out['ocaa_logits'].reshape(b*k, nc),
                             domain_ids[:, None].expand(b, k).reshape(-1), reduction='none').reshape(b, k)
        if mc.get('anchor_mass_normalize', False):
            ocaa = (((q*ce).sum(dim=0)) / (q.sum(dim=0)+1e-8)).mean()
        else:
            ocaa = (q*ce).sum(dim=1).mean()
    if model.use_ospl:
        ospl, ord_detail = ordinal_loss(out['latent'], target_scores, domain_ids,
                   margin=float(tr['ordinal_margin']), eta=float(tr['ordinal_distance_weight']),
                   pair_mode=tr.get('pair_mode', 'balanced'), max_pairs=int(tr.get('max_pairs',128)),
                   min_diff=float(tr.get('min_delta_score', 1e-6)),
                   ties_in_distance=bool(tr.get('ties_in_distance', True)), generator=generator)
    else:
        ospl = zero
        ord_detail = {}
    sched = sigmoid_grl_schedule(progress)
    # Eq. 13, Eq. 14: coefficients alpha_r=alpha_0 * lambda(progress), beta_r=beta_0 * lambda(progress).
    # GRL flips encoder gradient once (scale 1); alpha_r/beta_r weight both branches.
    total = score + sched * tr['alpha'] * gpaa + sched * tr['beta'] * ocaa + tr['gamma'] * ospl
    details = {'loss': total.detach(), 'score': score.detach(), 'gpaa': gpaa.detach(),
               'ocaa': ocaa.detach(), 'ospl': ospl.detach(), 'grl_schedule': sched}
    details.update(ord_detail)
    return total, details
