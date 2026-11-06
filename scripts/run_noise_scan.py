"""Table 5.7 / 5.8 / 5.9：噪声扫描 + conditioning。"""
import argparse, os, json, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from run_experiment import load_cfg
from sheaf_ham.train import train_one
from sheaf_ham.datasets import (make_target_fragments,
                                make_incompatible_target, add_fragment_noise)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    cfg, params = load_cfg(args.config)
    H = make_target_fragments(cfg.model_kind, cfg.N, cfg.periodic, params, cfg.L)
    H_bad = make_incompatible_target(H, delta=0.05, seed=1)

    out = {}
    for sigma in [1e-4, 1e-3, 1e-2, 3e-2]:
        cfg_i = cfg; cfg_i.epochs = min(cfg.epochs, 150)
        r_noise = train_one(cfg_i, add_fragment_noise(H, sigma), verbose=False)
        r_bad   = train_one(cfg_i, add_fragment_noise(H_bad, sigma), verbose=False)
        diff = (r_bad["final_proxy"] - r_noise["final_proxy"]) / \
               max(r_noise["final_proxy"], 1e-12)
        out[f"sigma={sigma:.0e}"] = {
            "compatible_noisy_proxy": r_noise["final_proxy"],
            "incompatible_noisy_proxy": r_bad["final_proxy"],
            "standardized_difference": diff,
            "cond_D": r_noise["diag"]["cond_D"],
            "strict_compat": r_noise["strict_trivial"],
            "strict_incompat": r_bad["strict_trivial"],
        }
        print(f"σ={sigma:.0e}  proxy_compat={r_noise['final_proxy']:.3e}  "
              f"proxy_incompat={r_bad['final_proxy']:.3e}  "
              f"std_diff={diff:.2f}  cond={r_noise['diag']['cond_D']:.2e}")

    path = os.path.join(args.outdir, "noise_scan.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"saved → {path}")


if __name__ == "__main__":
    main()