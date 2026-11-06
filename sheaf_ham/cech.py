"""Sec. 3.2.2 / 4.2.1 — Čech 1-上链 omega 与 D。"""
import torch
from .sheaf import SheafData, build_coboundary_matrix


def build_cech_cochain(H_frags: torch.Tensor, sheaf: SheafData) -> torch.Tensor:
    """
    H_frags: (n_cells_0, d)
    返回 omega: (n_overlap, d)
    omega_{ab} = P_ab (H_a - H_b) P_ab^†
    """
    omegas = []
    for (a, b) in sheaf.overlap_pairs:
        Pab = sheaf.P[(a, b)]
        Ha, Hb = H_frags[a], H_frags[b]
        rho = torch.kron(Pab.conj(), Pab)
        omegas.append(rho @ (Ha - Hb))
    if not omegas:
        return torch.zeros(0, sheaf.dim0,
                           dtype=H_frags.dtype, device=H_frags.device)
    return torch.stack(omegas, dim=0)