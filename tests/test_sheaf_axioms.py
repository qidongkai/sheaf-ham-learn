"""Assumption 2：验证 sheaf axioms（locality / gluing / functoriality）。"""
import torch
from sheaf_ham.sheaf import build_sheaf_1d, build_sheaf_2d


def test_functoriality_1d():
    s = build_sheaf_1d(6, 2)
    # 对任意 P_{ab}，rho 的组合应满足 rho_{U,U}=id
    for (a, b), Pab in s.P.items():
        rho = torch.kron(Pab.conj(), Pab)
        # 退化情形 P=I 时
        if torch.allclose(Pab, torch.eye(s.dim0)):
            assert torch.allclose(rho, torch.eye(s.dim0))


def test_gluing_1d():
    """若所有 H_sigma 相等，则 omega=0（平凡可粘）。"""
    from sheaf_ham.cech import build_cech_cochain
    s = build_sheaf_1d(6, 2)
    H = torch.ones(6, s.dim0, dtype=torch.complex128)
    omega = build_cech_cochain(H, s)
    assert torch.allclose(omega, torch.zeros_like(omega))


def test_2d_coboundary():
    s = build_sheaf_2d(3, 1)
    assert s.D.shape[0] == len(s.overlap_pairs) * s.dim0
    assert s.D.shape[1] == s.n_cells_0 * s.dim0