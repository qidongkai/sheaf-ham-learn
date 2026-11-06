"""Sec. 5.1.1, Table 5.1 — 物理模型。"""
import numpy as np
from dataclasses import dataclass
from typing import Dict


@dataclass
class LatticeModel:
    kind: str
    N: int = 8            # 1D
    L: int = 4            # 2D 边长
    params: Dict[str, float] = None
    periodic: bool = False

    def build(self) -> np.ndarray:
        if self.kind == "Ising":            return self._ising()
        if self.kind == "HatanoNelson":     return self._hatano_nelson()
        if self.kind == "Hofstadter":       return self._hofstadter()
        raise ValueError(self.kind)

    # ---- 非厄米 Ising（1D spin chain） ----
    def _ising(self):
        N = self.N
        sx = np.array([[0, 1], [1, 0]], dtype=complex)
        sz = np.array([[1, 0], [0, -1]], dtype=complex)
        I2 = np.eye(2, dtype=complex)

        def site_op(op, i):
            mats = [I2] * N
            mats[i] = op
            out = np.array([[1.0 + 0j]])
            for m in mats:
                out = np.kron(out, m)
            return out

        J = self.params.get("J", 1.0)
        h = self.params.get("h", 0.8)
        g = self.params.get("gamma", 0.0)
        H = np.zeros((2 ** N, 2 ** N), dtype=complex)
        for i in range(N - 1):
            H += J * (site_op(sx, i) @ site_op(sx, i + 1))
        if self.periodic:
            H += J * (site_op(sx, N - 1) @ site_op(sx, 0))
        for i in range(N):
            H += (h + 1j * g) * site_op(sz, i)
        return H

    # ---- Hatano-Nelson（1D 非互易 hopping） ----
    def _hatano_nelson(self):
        N = self.N
        tR = self.params.get("tR", 1.0)
        tL = self.params.get("tL", 1.0)
        H = np.zeros((N, N), dtype=complex)
        for i in range(N - 1):
            H[i, i + 1] = tR
            H[i + 1, i] = tL
        if self.periodic:
            H[N - 1, 0] = tR
            H[0, N - 1] = tL
        return H

    # ---- Hofstadter（2D 方格 + 磁通） ----
    def _hofstadter(self):
        L = self.L
        N = L * L
        t = self.params.get("t", 1.0)
        phi = self.params.get("phi", 0.25)
        H = np.zeros((N, N), dtype=complex)
        def idx(x, y): return x * L + y
        for x in range(L):
            for y in range(L):
                i = idx(x, y)
                # x 方向 hopping（Peierls phase）
                if x < L - 1 or self.periodic:
                    xn = (x + 1) % L
                    j = idx(xn, y)
                    H[i, j] += -t
                    H[j, i] += -t
                # y 方向 hopping，带相位
                if y < L - 1 or self.periodic:
                    yn = (y + 1) % L
                    j = idx(x, yn)
                    phase = np.exp(1j * 2 * np.pi * phi * x)
                    H[i, j] += -t * phase
                    H[j, i] += -t * np.conj(phase)
        return H