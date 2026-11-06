"""Sec. 4.3.3 — 训练循环。"""
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass

from .sheaf import build_sheaf_1d, build_sheaf_2d
from .cech import build_cech_cochain
from .cohomology import (DifferentiableCohomologicalProxy,
                         strict_cohomology_class)
from .gnn import CellularEquivariantGNN
from .losses import (compute_eta_min_norm, reconstruction_loss,
                     physical_constraint_loss, chain_closure_residual)
from .datasets import make_observation_features


@dataclass
class TrainConfig:
    model_kind: str = "Ising"
    N: int = 8
    L: int = 4
    periodic: bool = False
    feat_dim: int = 8
    hidden: int = 32
    n_layers: int = 2
    site_dim: int = 2
    equivariant: bool = True
    lr: float = 1e-3
    epochs: int = 200
    batch_size: int = 16
    lam: float = 0.1
    mu: float = 0.05
    lam_chain: float = 1e-4
    eps_scale: float = 1e-3
    recon_mode: str = "frobenius"
    seed: int = 0


def _make_sheaf(cfg: TrainConfig):
    if cfg.model_kind in ("Ising", "HatanoNelson"):
        return build_sheaf_1d(cfg.N, cfg.site_dim, cfg.periodic)
    return build_sheaf_2d(cfg.L, cfg.site_dim, cfg.periodic)


def _make_edges(sheaf):
    edges = []
    for (i, j) in sheaf.edges_0_1:
        edges += [[i, j], [j, i]]
    return torch.tensor(edges, dtype=torch.long).T


def train_one(cfg: TrainConfig, H_target: torch.Tensor, verbose=True):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    sheaf = _make_sheaf(cfg)
    proxy = DifferentiableCohomologicalProxy(sheaf, eps_scale=cfg.eps_scale)
    model = CellularEquivariantGNN(cfg.feat_dim, cfg.hidden, cfg.site_dim,
                                   cfg.n_layers, cfg.equivariant)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    edges = _make_edges(sheaf)
    feats = make_observation_features(sheaf.n_cells_0, cfg.feat_dim, cfg.seed)
    H_target = H_target.unsqueeze(0)

    history = {k: [] for k in ("loss", "proxy", "recon", "phys", "omega")}
    for ep in range(cfg.epochs):
        optimizer.zero_grad()
        H_pred = model(feats.unsqueeze(0), edges, cfg.periodic)
        dist, _ = proxy(H_pred)
        omega_flat = build_cech_cochain(H_pred[0], sheaf).reshape(1, -1)
        eta = compute_eta_min_norm(omega_flat, sheaf)
        L_rec = reconstruction_loss(H_pred, H_target, eta, sheaf, cfg.recon_mode)
        L_phys = physical_constraint_loss(H_pred, eta, sheaf, cfg.model_kind)
        L_chain = chain_closure_residual(omega_flat, sheaf).mean()
        loss = L_rec + cfg.lam * dist.mean() + cfg.mu * L_phys \
               + cfg.lam_chain * L_chain
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        history["loss"].append(loss.item())
        history["proxy"].append(dist.mean().item())
        history["recon"].append(L_rec.item())
        history["phys"].append(L_phys.item())
        history["omega"].append(omega_flat.norm().item())
        if verbose and (ep % 50 == 0 or ep == cfg.epochs - 1):
            print(f"ep {ep:4d} loss={loss.item():.3e} "
                  f"proxy={dist.mean().item():.3e} "
                  f"rec={L_rec.item():.3e} "
                  f"||omega||={omega_flat.norm().item():.3e}")

    with torch.no_grad():
        H_pred = model(feats.unsqueeze(0), edges, cfg.periodic)
        dist, diag = proxy(H_pred)
        omega_flat = build_cech_cochain(H_pred[0], sheaf).reshape(1, -1)
        eta = compute_eta_min_norm(omega_flat, sheaf)
        H_corr = H_pred - eta.view(1, sheaf.n_cells_0, -1)
        rec_err = F.mse_loss(H_corr, H_target).item()
        strict, strict_res = strict_cohomology_class(H_pred[0], sheaf)

    return {"model": model, "sheaf": sheaf, "proxy_module": proxy,
            "history": history, "final_proxy": dist.mean().item(),
            "final_recon_mse": rec_err,
            "final_omega_norm": omega_flat.norm().item(),
            "strict_trivial": strict, "strict_residual": strict_res,
            "diag": diag}