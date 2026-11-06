"""批量生成论文 Table 5.3 全数据集。"""
import argparse, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sheaf_ham.data.generators import generate_full_suite

PRESETS = {
    "ising": dict(kind="Ising", params={"J": 1.0, "h": 0.8, "gamma": 0.3},
                  N=8, L=4, periodic=False, obs_mode="trajectory"),
    "ising_ep": dict(kind="Ising", params={"J": 1.0, "h": 0.8, "gamma": 0.32},
                     N=8, L=4, periodic=False, obs_mode="trajectory"),
    "hatano": dict(kind="HatanoNelson", params={"tR": 1.0, "tL": 0.3},
                   N=8, L=4, periodic=False, obs_mode="trajectory"),
    "hofstadter": dict(kind="Hofstadter", params={"t": 1.0, "phi": 0.25},
                       N=8, L=4, periodic=True, obs_mode="trajectory"),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS), required=True)
    ap.add_argument("--outdir", default="datasets")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    cfg = PRESETS[args.preset]
    idx = generate_full_suite(args.outdir, seeds=args.seeds, **cfg)
    print(f"Generated {sum(len(v) for v in idx.values())} files → {args.outdir}")
    for k, v in idx.items():
        print(f"  {k:24s}: {len(v)} files")