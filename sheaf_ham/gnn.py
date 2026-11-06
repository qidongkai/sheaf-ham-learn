"""Sec. 4.1 — 胞腔等变 GNN。"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CellularEquivariantGNN(nn.Module):
    def __init__(self, feat_dim: int, hidden: int, site_dim: int,
                 n_layers: int = 2, equivariant: bool = True):
        super().__init__()
        self.site_dim = site_dim
        self.equivariant = equivariant
        self.in_proj = nn.Linear(feat_dim, hidden)
        self.layers = nn.ModuleList(
            [nn.Linear(2 * hidden, hidden) for _ in range(n_layers)])
        self.out = nn.Linear(hidden, site_dim * site_dim * 2)

    def forward(self, x, edges, periodic=False):
        """
        x: (B,N,F); edges: (2,E)
        返回 H: (B,N,d^2) complex
        """
        B, N, _ = x.shape
        h = self.in_proj(x)
        for layer in self.layers:
            src, dst = edges[0], edges[1]
            agg = torch.zeros_like(h)
            agg.index_add_(1, dst, h[:, src])
            deg = torch.zeros(N, device=x.device)
            deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
            agg = agg / deg.clamp_min(1.0).view(1, N, 1)
            h = F.gelu(layer(torch.cat([h, agg], dim=-1)))
        if self.equivariant and periodic:
            h = h - h.mean(dim=1, keepdim=True)  # 平移平均投影
        o = self.out(h)
        d2 = self.site_dim ** 2
        return torch.complex(o[..., :d2], o[..., d2:])