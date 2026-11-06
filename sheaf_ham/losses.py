"""Sec. 4.3.1 — 复合损失、最小范数提升、物理约束。"""
import math
import torch
import torch.nn.functional as F
from .sheaf import SheafData


def compute_eta_min_norm(omega_flat: torch.Tensor, sheaf: SheafData):
    """eta = D^+ omega（最小范数）。"""
    B = omega_flat.shape[0]
    D = sheaf.D
    etas = []
    for b in range(B):
        eta_b, *_ = torch.linalg.lstsq(D, omega_flat[b].unsqueeze(-1))
        etas.append(eta_b.squeeze(-1))
    return torch.stack(etas, dim=0)


def reconstruction_loss(H_frags, H_target, eta, sheaf, mode="frobenius"):
    """Sec. 4.3.1：L_recon 在 {H_sigma - eta_sigma} 上评估。"""
    B, n0, d = H_frags.shape
    H_corr = H_frags - eta.view(B, n0, d)
    if mode == "frobenius":
        return F.mse_loss(H_corr, H_target)
    raise ValueError(mode)


def physical_constraint_loss(H_frags, eta, sheaf, model_kind):
    """Sec. 4.3.1 / Assumption 3：G_phys 在全局截面上检验。"""
    B, n0, d = H_frags.shape
    H_corr = H_frags - eta.view(B, n0, d)
    loss = torch.zeros((), dtype=torch.float64, device=H_frags.device)
    if model_kind == "Ising":
        side = int(math.isqrt(d))
        blocks = H_corr.reshape(B, n0, side, side)
        anti = (blocks + blocks.conj().transpose(-1, -2)) / 2
        anti_diag = torch.diagonal(anti, dim1=-2, dim2=-1)
        anti_off = anti - torch.diag_embed(anti_diag)
        loss = loss + anti_off.abs().pow(2).mean()
        loss = loss + F.relu(blocks.abs() - 10.0).pow(2).mean()
    elif model_kind == "HatanoNelson":
        loss = loss + F.relu(H_corr.real.abs() - 5.0).pow(2).mean()
    return loss


def chain_closure_residual(omega_flat, sheaf):
    """Sec. 4.2.1：||delta^1 omega||^2，1D 恒为 0（无 triple overlap）。"""
    return torch.zeros(omega_flat.shape[0], device=omega_flat.device)