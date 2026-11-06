"""Theorem 1：在小系统上验证严格判据。"""
import torch
from sheaf_ham.sheaf import build_sheaf_1d
from sheaf_ham.cohomology import strict_cohomology_class
from sheaf_ham.datasets import make_target_fragments, make_incompatible_target


def test_compatible_is_trivial():
    s = build_sheaf_1d(6, 2)
    H = torch.randn(6, s.dim0, dtype=torch.complex128)
    # 人为把 H 构造得完全兼容：全部投影到同一子空间
    H[:] = H[0]
    trivial, res = strict_cohomology_class(H, s)
    assert trivial and res < 1e-10


def test_incompatible_is_nontrivial():
    s = build_sheaf_1d(6, 2)
    H = torch.zeros(6, s.dim0, dtype=torch.complex128)
    H[0] = 1.0
    H[1] = -1.0
    # omega_{01} = 2 ≠ 0，且不在 im(D)（D 核很大）
    trivial, res = strict_cohomology_class(H, s)
    assert not trivial