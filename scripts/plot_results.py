"""出图：Fig 2 / Fig 3 风格 + Table 5.11 曲线。"""
import argparse, os, json
import matplotlib.pyplot as plt


def plot_delta_scan(path, outdir):
    d = json.load(open(path))
    deltas = sorted(d["deltas"].items(), key=lambda x: float(x[0]))
    xs = [float(k) for k, _ in deltas]
    ys = [v["proxy"] for _, v in deltas]
    ref = d["reference_proxy"]
    plt.figure(figsize=(5, 3.5))
    plt.plot(xs, ys, "o-", label="incompatible proxy")
    plt.axhline(ref, color="gray", ls="--", label="compatible ref")
    plt.xlabel("Δ"); plt.ylabel("dist(ω, B¹)"); plt.yscale("log")
    plt.legend(); plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path_out = os.path.join(outdir, "delta_scan.png")
    plt.savefig(path_out, dpi=160); plt.close()
    print("saved →", path_out)


def plot_noise_scan(path, outdir):
    d = json.load(open(path))
    sigmas = [1e-4, 1e-3, 1e-2, 3e-2]
    compat = [d[f"sigma={s:.0e}"]["compatible_noisy_proxy"] for s in sigmas]
    incompat = [d[f"sigma={s:.0e}"]["incompatible_noisy_proxy"] for s in sigmas]
    plt.figure(figsize=(5, 3.5))
    plt.plot(sigmas, compat, "o-", label="compatible + noise")
    plt.plot(sigmas, incompat, "s-", label="incompatible + noise")
    plt.xlabel("σ"); plt.ylabel("proxy"); plt.xscale("log"); plt.yscale("log")
    plt.legend(); plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path_out = os.path.join(outdir, "noise_scan.png")
    plt.savefig(path_out, dpi=160); plt.close()
    print("saved →", path_out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    dp = os.path.join(args.outdir, "delta_scan.json")
    np_ = os.path.join(args.outdir, "noise_scan.json")
    if os.path.exists(dp): plot_delta_scan(dp, args.outdir)
    if os.path.exists(np_): plot_noise_scan(np_, args.outdir)