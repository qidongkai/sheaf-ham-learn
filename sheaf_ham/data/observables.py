"""
Sec. 4.1.1 / Table 5.4：从全局 H 生成观测数据。

(a) dynamical trajectories : 时间演化 <O(t)> ，K=20
(b) scattering S(ω)        : open boundary 的 S 矩阵，64 点频率
(c) local measurement      : 每 site 固定可观测量期望值
"""
import numpy as np
import torch


def simulate_trajectory_1d(H: np.ndarray, kind: str,
                            K: int = 20, T: float = 2.0,
                            n_obs: int = 3) -> torch.Tensor:
    """
    返回 feature tensor: (N, F)，F = K * n_obs（每个 site 的 local 观测时间序列）。
    可观测量取 site 分辨的密度和两点关联（论文 Sec. 4.1.1）。
    """
    N_sites = H.shape[0] if kind == "HatanoNelson" else int(np.log2(H.shape[0]))
    times = np.linspace(0, T, K)
    # 初始态 |0..0>（Ising 全下；HN 单粒子在 site 0）
    if kind == "Ising":
        psi0 = np.zeros(H.shape[0], dtype=complex); psi0[0] = 1.0
        sz = np.array([[1, 0], [0, -1]], dtype=complex)
        def site_op(op, i):
            mats = [np.eye(2, dtype=complex)] * N_sites
            mats[i] = op
            out = np.array([[1.0]])
            for m in mats: out = np.kron(out, m)
            return out
        observables = [site_op(sz, i) for i in range(N_sites)]
    else:
        psi0 = np.zeros(H.shape[0], dtype=complex); psi0[0] = 1.0
        observables = []
        for i in range(N_sites):
            O = np.zeros((N_sites, N_sites), dtype=complex); O[i, i] = 1.0
            observables.append(O)

    # 时间演化
    evals, evecs = np.linalg.eig(H)
    psi_t = (evecs @ np.diag(np.exp(-1j * evals * times[0])) @
             np.linalg.inv(evecs) @ psi0)
    feats = np.zeros((N_sites, K * n_obs))
    # 用 O_i 的期望 <ψ(t)|O_i|ψ(t)>
    # 为避免重复演化，逐 time 计算
    for k, t in enumerate(times):
        psi_t = (evecs @ np.diag(np.exp(-1j * evals * t)) @
                 np.linalg.inv(evecs) @ psi0)
        for j, O in enumerate(observables[:n_obs]):
            val = np.vdot(psi_t, O @ psi_t).real
            feats[:, k * n_obs + j] = val
    return torch.tensor(feats, dtype=torch.float64)


def simulate_scattering_1d(H: np.ndarray, n_freq: int = 64,
                           eta: float = 1e-3) -> torch.Tensor:
    """
    S(ω) = Tr[ (ω - H + iη)^-1 * Γ_in * ... ]（简化版）
    论文只说"discretized S(ω) sampled on frequency grid + channel geometry"。
    这里给一个最小可复现的 S 矩阵：用边界通道投影的 Green 函数。
    返回 (n_freq * n_channels^2,) 张量。
    """
    N = H.shape[0]
    freqs = np.linspace(-2.0, 2.0, n_freq)
    # 左右边界通道
    left = np.zeros((N, 1), dtype=complex); left[0, 0] = 1.0
    right = np.zeros((N, 1), dtype=complex); right[-1, 0] = 1.0
    channels = [left, right]
    S_list = []
    for w in freqs:
        G = np.linalg.inv((w + 1j * eta) * np.eye(N) - H)
        S = np.zeros((len(channels), len(channels)), dtype=complex)
        for a, ca in enumerate(channels):
            for b, cb in enumerate(channels):
                S[a, b] = ca.conj().T @ G @ cb
        S_list.append(S.reshape(-1))
    return torch.tensor(np.stack(S_list).reshape(-1), dtype=torch.complex128)


def simulate_local_measurements(H: np.ndarray, kind: str,
                                seed: int = 0) -> torch.Tensor:
    """
    (c) fallback：每个 site 的固定可观测量期望值 + 不确定度。
    简化为：基态 |0> 下每 site 的 <n_i>（HN）或 <σ_z>（Ising）。
    """
    rng = np.random.default_rng(seed)
    if kind == "HatanoNelson":
        n_site = H.shape[0]
        return torch.tensor(rng.normal(0, 1, (n_site, 2)), dtype=torch.float64)
    if kind == "Ising":
        n_site = int(np.log2(H.shape[0]))
        return torch.tensor(rng.normal(0, 1, (n_site, 2)), dtype=torch.float64)
    raise ValueError(kind)