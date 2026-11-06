"""主实验：Table 5.5 / 5.11 / 5.14 风格。"""
import argparse
import os, json, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch, yaml
from sheaf_ham.train import TrainConfig, train_one
from sheaf_ham.datasets import make_target_fragments, make_incompatible_target


def load_cfg(path):
    with open(path) as f:
        d = yaml.safe_load(f)
    t = d["train"]; m = d["model"]; s = d.get("sheaf", {}); g = d.get("gnn", {})
    return TrainConfig(
        model_kind=m["kind"], N=m.get("N", 8), L=m.get("L", 4),
        periodic=m.get("periodic", False),
        feat_dim=g.get("feat_dim", 8), hidden=g.get("hidden", 32),
        n_layers=g.get("n_layers", 2),
        site_dim=s.get("site_dim", 1), equivariant=g.get("equivariant", True),
        lr=t["lr"], epochs=t["epochs"], batch_size=t["batch_size"],
        lam=t["lam"], mu=t["mu"], lam_chain=t["lam_chain"],
        eps_scale=t["eps_scale"], recon_mode=t["recon_mode"], seed=t["seed"],
    ), m["params"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    cfg, params = load_cfg(args.config)
    H = make_target_fragments(cfg.model_kind, cfg.N, cfg.periodic, params, cfg.L)
    H_bad = make_incompatible_target(H, delta=0.20, seed=1)

    print(f"\n=== {cfg.model_kind} | compatible ===")
    r_c = train_one(cfg, H)
    print(f"\n=== {cfg.model_kind} | incompatible (Δ=0.20) ===")
    r_i = train_one(cfg, H_bad)

    out = {
        "config": cfg.__dict__,
        "compatible": {k: v for k, v in r_c.items() if k not in ("model", "sheaf", "proxy_module")},
        "incompatible": {k: v for k, v in r_i.items() if k not in ("model", "sheaf", "proxy_module")},
        "ratio": r_i["final_proxy"] / max(r_c["final_proxy"], 1e-12),
        "history_compatible": r_c["history"],
        "history_incompatible": r_i["history"],
    }
    name = os.path.splitext(os.path.basename(args.config))[0]
    path = os.path.join(args.outdir, f"{name}_result.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[Table 5.11] proxy ratio = {out['ratio']:.1f}x")
    print(f"results saved → {path}")


if __name__ == "__main__":
    main()