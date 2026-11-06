"""Sec. 3.1.2 — 层 G 与限制映射 rho。"""
from dataclasses import dataclass
from typing import List, Tuple, Dict
import torch


@dataclass
class SheafData:
    n_cells_0: int
    edges_0_1: List[Tuple[int, int]]
    overlap_pairs: List[Tuple[int, int]]
    P: Dict[Tuple[int, int], torch.Tensor]
    dim0: int

    def __post_init__(self):
        self._D = None

    @property
    def D(self) -> torch.Tensor:
        if self._D is None:
            self._D = build_coboundary_matrix(self)
        return self._D


def _kron_rho(P: torch.Tensor) -> torch.Tensor:
    """rho(vec(M)) = vec(P M P^†) = (conj(P) ⊗ P) vec(M)。"""
    return torch.kron(P.conj(), P)


def build_coboundary_matrix(sheaf: SheafData) -> torch.Tensor:
    """Sec. 3.2.2 Eq.(15)：D: C^0 -> C^1。"""
    n0 = sheaf.n_cells_0
    n1 = len(sheaf.overlap_pairs)
    d = sheaf.dim0
    D = torch.zeros(n1 * d, n0 * d, dtype=torch.complex128)
    for k, (a, b) in enumerate(sheaf.overlap_pairs):
        Pab = sheaf.P[(a, b)]
        rho = _kron_rho(Pab)
        D[k * d:(k + 1) * d, a * d:(a + 1) * d] += rho
        D[k * d:(k + 1) * d, b * d:(b + 1) * d] -= rho
    return D


def build_sheaf_1d(N: int, site_dim: int, periodic: bool = False) -> SheafData:
    """1D 平凡投影：P = I（论文 Sec. 3.1.2 末段）。"""
    overlaps = [(i, i + 1) for i in range(N - 1)]
    if periodic and N > 2:
        overlaps.append((N - 1, 0))
    d = site_dim * site_dim
    I = torch.eye(d, dtype=torch.complex128)
    P = {p: I.clone() for p in overlaps}
    P.update({(b, a): I.clone() for (a, b) in overlaps})
    edges = [(i, i + 1) for i in range(N - 1)]
    if periodic and N > 2:
        edges.append((N - 1, 0))
    return SheafData(n_cells_0=N, edges_0_1=edges,
                     overlap_pairs=overlaps, P=P, dim0=d)


def build_sheaf_2d(L: int, site_dim: int, periodic: bool = False) -> SheafData:
    """2D 方格：0-cell = site，overlap 对 = 最近邻 pair。"""
    def idx(x, y): return x * L + y
    N = L * L
    overlaps = []
    edges = []
    for x in range(L):
        for y in range(L):
            i = idx(x, y)
            if x < L - 1 or periodic:
                j = idx((x + 1) % L, y)
                overlaps.append((i, j))
                edges.append((i, j))
            if y < L - 1 or periodic:
                j = idx(x, (y + 1) % L)
                overlaps.append((i, j))
                edges.append((i, j))
    d = site_dim * site_dim
    I = torch.eye(d, dtype=torch.complex128)
    P = {}
    for (a, b) in overlaps:
        P[(a, b)] = I.clone()
        P[(b, a)] = I.clone()
    return SheafData(n_cells_0=N, edges_0_1=edges,
                     overlap_pairs=overlaps, P=P, dim0=d)