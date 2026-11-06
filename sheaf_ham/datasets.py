"""Sec. 5.1.2-5.1.3 — 数据集构造。"""
import numpy as np
import torch
from .models import LatticeModel


def make_target_fragments(kind, N, periodic, params, L=4):
    """从精确对角化抽取每个 0-cell 的当地片段。"""
    lm = LatticeModel(kind=kind, N=N, L=L, params=params, periodic=periodic)
    H = lm.build()
    if kind == "Ising":
        d = 2
        frags = torch.zeros(N, d * d, dtype=torch.complex128)
        for i in range(N):
            idx0 = 0
            idx1 = 1 << (N - 1 - i)
            block = H[np.ix_([idx0, idx1], [idx0, idx1])]
            frags[i] = torch.tensor(block.reshape(-1), dtype=torch.complex128)
        return frags
    if kind in ("HatanoNelson", "Hofstadter"):
        N0 = H.shape[0]
        frags = torch.zeros(N0, 1, dtype=torch.complex128)
        frags[:, 0] = torch.tensor(np.diag(H), dtype=torch.complex128)
        return frags
    raise ValueError(kind)


def make_observation_features(N, feat_dim, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(N, feat_dim, generator=g)


def make_incompatible_target(frags, delta, seed=0):
    """Table 5.3：相邻 cell 冲突幅度 Δ。"""
    g = torch.Generator().manual_seed(seed)
    N = frags.shape[0]
    noise = torch.zeros_like(frags)
    for i in range(N):
        sign = 1.0 if i % 2 == 0 else -1.0
        noise[i] = sign * delta * torch.randn(frags.shape[1],
                                              generator=g, dtype=frags.dtype)
    return frags + noise


def add_fragment_noise(frags, sigma, seed=42):
    g = torch.Generator().manual_seed(seed)
    return frags + sigma * torch.randn(frags.shape, generator=g,
                                       dtype=frags.dtype)