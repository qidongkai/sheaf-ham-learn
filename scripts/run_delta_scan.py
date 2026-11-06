"""Table 5.11：proxy 随 Δ 增长。"""
import argparse, os, json, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from run_experiment import load_cfg
from sheaf_ham.train import train_one
from sheaf_ham.datasets import make_target_fragments, make_incompatible_target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    cfg, params = load_cfg(args.config)
    cfg.epochs = min(cfg.epochs, 150)
    H = make_target_fragments(cfg.model_kind, cfg.N, cfg.periodic, params, cfg.L)
    ref = train_one(cfg, H, verbose=False)
    out = {"reference_proxy": ref["final_proxy"], "deltas": {}}
    for delta in [0.02, 0.05, 0.10, 0.20]:
        H_bad = make_incompatible_target(H, delta=delta, seed=1)
        r = train_one(cfg, H_bad, verbose=False)
        ratio = r["final_proxy"] / max(ref["final_proxy"], 1e-12)
        out["deltas"][f"{delta:.2f}"] = {
            "proxy": r["final_proxy"],
            "ratio": ratio,
            "strict_trivial": r["strict_trivial"],
            "strict_residual": r["strict_residual"],
        }
        print(f"Δ={delta:.2f}  proxy={r['final_proxy']:.3e}  "
              f"ratio={ratio:.1f}x  strict_trivial={r['strict_trivial']}")
    path = os.path.join(args.outdir, "delta_scan.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"saved → {path}")


if __name__ == "__main__":
    main()