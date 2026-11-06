"""Sec. 3.3 — 严格判据 + 可微代理。"""
import torch
import torch.nn as nn
from typing import Tuple, Dict
from .sheaf import SheafData
from .cech import build_cech_cochain


# ---------- 严格判据（Theorem 1） ----------
def strict_cohomology_class(H_frags: torch.Tensor, sheaf: SheafData,
                            rtol: float = 1e-8) -> Tuple[bool, float]:
    omega = build_cech_cochain(H_frags, sheaf).flatten()
    D = sheaf.D
    eta, *_ = torch.linalg.lstsq(D, omega.unsqueeze(-1))
    residual = omega - (D @ eta.squeeze(-1))
    res_norm = torch.linalg.norm(residual).item()
    is_trivial = res_norm <= rtol * (torch.linalg.norm(omega).item() + 1e-12)
    return is_trivial, res_norm


# ---------- 可微代理（Eq.27-29） ----------
class DifferentiableCohomologicalProxy(nn.Module):
    def __init__(self, sheaf: SheafData, eps_scale: float = 1e-3,
                 svd_cutoff: float = 1e-10):
        super().__init__()
        self.sheaf = sheaf
        self.eps_scale = eps_scale
        D = sheaf.D
        U, S, _ = torch.linalg.svd(D, full_matrices=False)
        keep = S > svd_cutoff * S.max()
        self.register_buffer("U_r", U[:, keep])
        self.register_buffer("S", S)
        self.register_buffer("keep", keep)
        self.cond = (S.max() / S[keep].min()).item() if keep.any() else float("inf")

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        Ur = self.U_r
        return (x @ Ur) @ Ur.conj().T if x.dim() == 2 else Ur @ (Ur.conj().T @ x)

    def forward(self, H_frags: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        B = H_frags.shape[0]
        omega = torch.stack([build_cech_cochain(H_frags[b], self.sheaf)
                             for b in range(B)], dim=0)
        omega_flat = omega.reshape(B, -1)
        P_omega = self._project(omega_flat)
        residual = omega_flat - P_omega
        omega_norm = torch.linalg.norm(omega_flat, dim=-1)
        res_norm = torch.linalg.norm(residual, dim=-1)
        eps = self.eps_scale * omega_norm.detach().mean().clamp_min(1e-12)
        dist = res_norm / (omega_norm + eps)
        return dist, {"omega_norm_mean": omega_norm.mean().item(),
                      "residual_mean": res_norm.mean().item(),
                      "cond_D": self.cond, "eps": eps.item()}