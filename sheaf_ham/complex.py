"""Sec. 3.1.1 — 胞腔复形构造。"""
from typing import List, Tuple


def build_complex_1d(N: int, periodic: bool = False):
    """
    0-cells: 0..N-1
    1-cells: (i,i+1)
    返回 (cells_0, cells_1, edges_0_1)
    """
    cells_0 = list(range(N))
    edges_0_1 = [(i, i + 1) for i in range(N - 1)]
    if periodic and N > 2:
        edges_0_1.append((N - 1, 0))
    cells_1 = edges_0_1[:]
    return cells_0, cells_1, edges_0_1


def build_complex_2d(L: int, periodic: bool = False):
    """
    0-cells: (x,y)
    1-cells: 水平 (x,y)->(x+1,y)，垂直 (x,y)->(x,y+1)
    2-cells: plaquette (x,y)，由 4 个 1-cell 围成
    返回 (cells_0, cells_1, cells_2, edges_0_1, plaquettes)
    """
    cells_0 = [(x, y) for x in range(L) for y in range(L)]
    cells_1 = []
    for x in range(L):
        for y in range(L):
            if x < L - 1 or periodic:
                cells_1.append(((x, y), ((x + 1) % L, y)))
            if y < L - 1 or periodic:
                cells_1.append(((x, y), (x, (y + 1) % L)))
    cells_2 = [(x, y) for x in range(L) for y in range(L)]  # plaquette 左下角坐标
    edges_0_1 = [((x, y), (xn, yn)) for (x, y), (xn, yn) in cells_1]
    plaquettes = []
    for x in range(L):
        for y in range(L):
            xn, yn = (x + 1) % L, (y + 1) % L
            plaquettes.append(((x, y), (xn, y), (xn, yn), (x, yn)))
    return cells_0, cells_1, cells_2, edges_0_1, plaquettes