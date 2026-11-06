# Sheaf-Cohomological Learning (Reference Implementation)

复现论文：
Qi, Huang, Ye. *Sheaf-Cohomological Learning: Gluing Local Hamiltonian
Fragments into Global Hamiltonians*.

## 安装
    pip install -r requirements.txt

## 快速开始
    python scripts/run_experiment.py --config configs/ising_1d.yaml
    python scripts/run_noise_scan.py --config configs/ising_1d.yaml
    python scripts/run_delta_scan.py --config configs/ising_1d.yaml
    python scripts/plot_results.py --outdir results/

## 目录与论文对应
| 模块 | 论文章节 |
|---|---|
sheaf_hamiltonian
| sheaf_ham/models.py | Sec. 5.1.1, Table 5.1 |
| sheaf_ham/complex.py | Sec. 3.1.1 |
| sheaf_ham/sheaf.py | Sec. 3.1.2 |
| sheaf_ham/cech.py | Sec. 3.2.2, Sec. 4.2.1 |
| sheaf_ham/cohomology.py | Sec. 3.3, Theorem 1, Eq.(27)-(29) |
| sheaf_ham/gnn.py | Sec. 4.1 |
| sheaf_ham/losses.py | Sec. 4.3.1, Eq.(33)-(34) |
| sheaf_ham/datasets.py | Sec. 5.1.2-5.1.3, Table 5.3-5.4 |
| sheaf_ham/train.py | Sec. 4.3.3 |
| scripts/run_*.py | Sec. 5.2-5.5 |
| tests/ | Assumption 1-2, Theorem 1 |

## 显式限制（论文 Sec. 6.3）
- 代理 dist(omega,B^1) 不等于严格 H^1 判据
- 小幅度非平凡 cocycle 会被漏检（Table 5.8）
- 高噪声时假阳性（Table 5.8）
- 物理筛选 G_phys 不是子层，只能在全局截面上检验

sheaf_ham/
├── data/
│   ├── __init__.py
│   ├── fragments.py      # 从全局 H 切出 {H_σ}（0/1/2-cell）
│   ├── observables.py    # 轨迹 / 散射 / 局部测量
│   └── generators.py     # Table 5.3 五类数据集构造器
└── datasets.py           # 旧的快捷入口保留，内部调用 data/
