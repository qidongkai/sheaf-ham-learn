"""
从精确对角化的全局 H 抽取 local fragments（Sec. 4.1.2 / 4.3.3）。

片段类型：
  0-cell (site i) :  on-site block  H_i ∈ C^{d_i × d_i}
  1-cell (i,j)    :  hopping block  H_ij ∈ C^{d_i × d_j}
  2-cell (ijk)    :  phase/flux     φ_ijk ∈ R  （Hofstadter 用 Peierls phase）

对 1D 模型，论文 Table 5.1 的 Ising 是 spin-1/2（d=2），
Hatano-Nelson 是单粒子（d=1）。
"""
import numpy as np
import torch


def extract_fragments_1d(H: np.ndarray, kind: str) -> dict:
    """
    H : 全局哈密顿量矩阵
        - Ising:      (2^N, 2^N)
        - Hatano-Nelson: (N, N)
    返回 dict:
        {
          "onsite": Tensor (N, d^2),        # 每个 site 的 on-site 片段
          "hopping": Tensor (N-1, d^2),     # 每个 bond 的 hopping 片段
        }
    """
    N = H.shape[0]
    if kind == "Ising":
        N_sites = int(np.log2(N))
        d = 2
        onsite = np.zeros((N_sites, d, d), dtype=complex)
        # 取每个 site 的 2x2 边缘块：把其他 site 冻结在 |0>
        for i in range(N_sites):
            idx0 = 0
            idx1 = 1 << (N_sites - 1 - i)
            onsite[i] = H[np.ix_([idx0, idx1], [idx0, idx1])]
        # hopping 片段：从 H 中提取 i,i+1 的耦合矩阵（不精确分解，
        # 论文里 local fragment 是由 GNN 学习的，数据集只提供 supervision target）
        hopping = np.zeros((N_sites - 1, d, d), dtype=complex)
        for i in range(N_sites - 1):
            idx_a = [1 << (N_sites - 1 - i)]
            idx_b = [1 << (N_sites - 1 - i - 1)]
            hopping[i] = H[np.ix_(idx_a, idx_b)]
        return {
            "onsite":  torch.tensor(onsite.reshape(N_sites, -1)),
            "hopping": torch.tensor(hopping.reshape(N_sites - 1, -1)),
        }

    if kind == "HatanoNelson":
        d = 1
        onsite = np.diag(H).reshape(N, 1, 1)
        hopping = np.zeros((N - 1, 1, 1), dtype=complex)
        for i in range(N - 1):
            hopping[i, 0, 0] = H[i, i + 1]
        return {
            "onsite":  torch.tensor(onsite.reshape(N, -1)),
            "hopping": torch.tensor(hopping.reshape(N - 1, -1)),
        }
    raise ValueError(kind)


def extract_fragments_2d(H: np.ndarray, L: int, kind: str) -> dict:
    """
    2D 版本：
      - onsite  : (L^2, d^2)
      - hopping : (E, d^2)  E = 2*L*(L-1) 个 bond（开边界）/ 2*L^2（周期）
      - plaquette: (L^2, 1) 每个 plaquette 的 flux（从 hopping 相位还原）
    """
    N = L * L
    if kind == "Hofstadter":
        d = 1
        onsite = np.diag(H).reshape(N, 1, 1).astype(complex)
        def idx(x, y): return x * L + y
        hop_list, flux_list, plaquettes = [], [], []
        for x in range(L):
            for y in range(L):
                i = idx(x, y)
                # x 方向 hopping
                if x < L - 1:
                    j = idx(x + 1, y)
                    hop_list.append(((i, j), H[i, j]))
                    hop_list.append(((j, i), H[j, i]))
                # y 方向 hopping（带相位）
                if y < L - 1:
                    j = idx(x, y + 1)
                    hop_list.append(((i, j), H[i, j]))
                    hop_list.append(((j, i), H[j, i]))
        # 每个 plaquette 的 flux
        for x in range(L):
            for y in range(L):
                if x == L - 1 or y == L - 1:
                    continue
                i00, i10 = idx(x, y), idx(x + 1, y)
                i11, i01 = idx(x + 1, y + 1), idx(x, y + 1)
                ph = H[i00, i10] * H[i10, i11] * H[i11, i01] * H[i01, i00]
                flux = np.angle(ph)
                plaquettes.append(flux)
        hop_vals = np.array([v for _, v in hop_list]).reshape(-1, 1, 1)
        return {
            "onsite": torch.tensor(onsite.reshape(N, -1)),
            "hopping": torch.tensor(hop_vals.reshape(len(hop_list), -1)),
            "plaquette": torch.tensor(np.array(plaquettes).reshape(-1, 1)),
        }
    raise ValueError(kind)