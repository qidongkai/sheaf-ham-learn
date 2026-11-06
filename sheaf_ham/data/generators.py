"""
Table 5.3 / Sec. 5.1.2：五类数据集完整构造器。

每类返回 dict:
    {
      "global_H":   np.ndarray,           # 参考全局哈密顿量
      "fragments":  {"onsite":..., ...},  # supervision target
      "observables": torch.Tensor,        # 观测特征（trajectory / scattering / local）
      "meta":       {...}                 # Δ, σ, boundary, seed, ...
    }
"""
import numpy as np
import torch
from pathlib import Path

from ..models import LatticeModel
from .fragments import extract_fragments_1d, extract_fragments_2d
from .observables import (simulate_trajectory_1d, simulate_scattering_1d,
                          simulate_local_measurements)


# ---------- 公共: 根据模型 kind 决定 site_dim ----------
def _site_dim(kind):
    return 2 if kind == "Ising" else 1


# ---------- (1) Compatible ----------
def make_compatible_dataset(kind: str, params: dict, N: int = 8,
                            L: int = 4, periodic: bool = False,
                            obs_mode: str = "trajectory",
                            seed: int = 0) -> dict:
    lm = LatticeModel(kind=kind, N=N, L=L, params=params, periodic=periodic)
    H = lm.build()

    if kind in ("Ising", "HatanoNelson"):
        frags = extract_fragments_1d(H, kind)
    else:
        frags = extract_fragments_2d(H, L, kind)

    if obs_mode == "trajectory":
        obs = simulate_trajectory_1d(H, kind)
    elif obs_mode == "scattering":
        obs = simulate_scattering_1d(H)
    elif obs_mode == "local":
        obs = simulate_local_measurements(H, kind, seed)
    else:
        raise ValueError(obs_mode)

    return {
        "global_H": H,
        "fragments": frags,
        "observables": obs,
        "meta": {"class": "compatible", "kind": kind, "params": params,
                 "N": N, "L": L, "periodic": periodic,
                 "obs_mode": obs_mode, "seed": seed},
    }


# ---------- (2) Incompatible: local contradiction ----------
def make_local_contradiction_dataset(kind: str, params: dict, delta: float,
                                     N: int = 8, L: int = 4,
                                     periodic: bool = False,
                                     obs_mode: str = "trajectory",
                                     seed: int = 0) -> dict:
    """
    Table 5.3："相邻 cell 在 overlap 上不一致，冲突幅度 Δ"。
    做法：先构造 compatible fragments，然后在相邻 cell 上交替叠加 ±Δ。
    单位：典型 hopping norm。
    """
    base = make_compatible_dataset(kind, params, N, L, periodic, obs_mode, seed)
    frags = base["fragments"]
    onsite = frags["onsite"].clone()
    n0 = onsite.shape[0]
    # 典型 hopping 尺度
    if "hopping" in frags:
        scale = frags["hopping"].abs().mean().item()
    else:
        scale = 1.0
    scale = max(scale, 1e-6)
    g = torch.Generator().manual_seed(seed)
    for i in range(n0):
        sign = 1.0 if i % 2 == 0 else -1.0
        noise = delta * scale * torch.randn(onsite.shape[1],
                                            generator=g, dtype=onsite.dtype)
        onsite[i] = onsite[i] + sign * noise
    base["fragments"]["onsite"] = onsite
    base["meta"]["class"] = "local_contradiction"
    base["meta"]["delta"] = delta
    return base


# ---------- (3) Incompatible: boundary conflict ----------
def make_boundary_conflict_dataset(kind: str, params: dict,
                                   N: int = 8, L: int = 4,
                                   obs_mode: str = "trajectory",
                                   seed: int = 0) -> dict:
    """
    Table 5.3：一部分 patch 用周期边界、一部分用开边界。
    做法：构造开边界与周期边界两个 compatible 数据集，
          按 patch 划分（前半用 OBC，后半用 PBC）拼接 fragments。
    """
    obc = make_compatible_dataset(kind, params, N, L,
                                  periodic=False, obs_mode=obs_mode, seed=seed)
    pbc = make_compatible_dataset(kind, params, N, L,
                                  periodic=True, obs_mode=obs_mode, seed=seed)
    frags = obc["fragments"]
    n0 = frags["onsite"].shape[0]
    half = n0 // 2
    frags["onsite"][half:] = pbc["fragments"]["onsite"][half:]
    obc["meta"]["class"] = "boundary_conflict"
    obc["meta"]["boundary"] = "OBC×PBC"
    return obc


# ---------- (4) Incompatible: symmetry breaking ----------
def make_symmetry_breaking_dataset(kind: str, params: dict,
                                   patch_frac: float,
                                   N: int = 8, L: int = 4,
                                   periodic: bool = False,
                                   obs_mode: str = "trajectory",
                                   seed: int = 0) -> dict:
    """
    Table 5.3：子集片段破缺较低对称性（10% 或 100% 的 cell）。
    做法：对指定 patch 的 onsite 片段施加一个不对称扰动。
    """
    base = make_compatible_dataset(kind, params, N, L, periodic, obs_mode, seed)
    frags = base["fragments"]
    onsite = frags["onsite"].clone()
    n0 = onsite.shape[0]
    n_patch = max(1, int(patch_frac * n0))
    g = torch.Generator().manual_seed(seed)
    for i in range(n_patch):
        onsite[i] = onsite[i] + 0.05 * torch.randn(
            onsite.shape[1], generator=g, dtype=onsite.dtype)
    base["fragments"]["onsite"] = onsite
    base["meta"]["class"] = "symmetry_breaking"
    base["meta"]["patch_frac"] = patch_frac
    return base


# ---------- (5) Noisy ----------
def make_noisy_dataset(kind: str, params: dict, sigma: float,
                       N: int = 8, L: int = 4, periodic: bool = False,
                       obs_mode: str = "trajectory",
                       seed: int = 0) -> dict:
    """
    Table 5.3：片段级复高斯噪声，σ 单位 = 典型片段范数。
    注意：这是 fragment-level noise，与 observation-level noise 分开。
    """
    base = make_compatible_dataset(kind, params, N, L, periodic, obs_mode, seed)
    frags = base["fragments"]
    g = torch.Generator().manual_seed(seed)
    for key in ("onsite", "hopping"):
        if key in frags:
            T = frags[key]
            typ = T.abs().mean().item() + 1e-12
            noise = sigma * typ * (torch.randn(T.shape, generator=g, dtype=T.dtype)
                                   + 1j * torch.randn(T.shape, generator=g, dtype=T.dtype)) / np.sqrt(2)
            frags[key] = T + noise
    base["meta"]["class"] = "noisy"
    base["meta"]["sigma"] = sigma
    return base


# ---------- 注册表（方便脚本批量生成） ----------
DATASET_REGISTRY = {
    "compatible":              make_compatible_dataset,
    "local_contradiction":     make_local_contradiction_dataset,
    "boundary_conflict":       make_boundary_conflict_dataset,
    "symmetry_breaking":       make_symmetry_breaking_dataset,
    "noisy":                   make_noisy_dataset,
}


# ---------- 批量生成 + 磁盘落盘 ----------
def generate_full_suite(outdir: str, kind: str, params: dict,
                        N: int = 8, L: int = 4, periodic: bool = False,
                        obs_mode: str = "trajectory",
                        deltas=(0.02, 0.05, 0.10, 0.20),
                        sigmas=(1e-4, 1e-3, 1e-2, 3e-2),
                        patch_fracs=(0.1, 1.0),
                        seeds=(0, 1, 2, 3, 4)):
    """对应论文：每个数据集 5 seeds（Sec. 5.1.2），批量生成全 suite。"""
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    idx = {}
    for seed in seeds:
        # compatible
        d = make_compatible_dataset(kind, params, N, L, periodic, obs_mode, seed)
        _save(d, out / f"compatible_{kind}_seed{seed}.pt"); idx.setdefault("compatible", []).append(str(out / f"compatible_{kind}_seed{seed}.pt"))
        # local contradiction
        for Δ in deltas:
            d = make_local_contradiction_dataset(kind, params, Δ, N, L, periodic, obs_mode, seed)
            p = out / f"local_contradiction_d{Δ:.2f}_{kind}_seed{seed}.pt"
            _save(d, p); idx.setdefault("local_contradiction", []).append(str(p))
        # boundary conflict
        d = make_boundary_conflict_dataset(kind, params, N, L, obs_mode, seed)
        p = out / f"boundary_conflict_{kind}_seed{seed}.pt"
        _save(d, p); idx.setdefault("boundary_conflict", []).append(str(p))
        # symmetry breaking
        for frac in patch_fracs:
            d = make_symmetry_breaking_dataset(kind, params, frac, N, L, periodic, obs_mode, seed)
            p = out / f"symmetry_breaking_p{frac}_{kind}_seed{seed}.pt"
            _save(d, p); idx.setdefault("symmetry_breaking", []).append(str(p))
        # noisy
        for σ in sigmas:
            d = make_noisy_dataset(kind, params, σ, N, L, periodic, obs_mode, seed)
            p = out / f"noisy_s{σ:.0e}_{kind}_seed{seed}.pt"
            _save(d, p); idx.setdefault("noisy", []).append(str(p))
    import json
    (out / "index.json").write_text(json.dumps(idx, indent=2))
    return idx


def _save(d: dict, path: Path):
    torch.save({
        "fragments": d["fragments"],
        "observables": d["observables"],
        "meta": d["meta"],
        # global_H 太大且可从 meta 重建，不落盘
    }, path)