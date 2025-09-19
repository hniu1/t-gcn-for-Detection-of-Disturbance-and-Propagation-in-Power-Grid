#!/usr/bin/env python3
import os, argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import stats
from collections import deque

def spectral_embed_2d(W, eps=1e-9):
    d = W.sum(axis=1)
    d = np.clip(d, eps, None)
    Dm = np.diag(1.0/np.sqrt(d))
    S  = Dm @ W @ Dm
    L  = np.eye(W.shape[0]) - S
    vals, vecs = np.linalg.eigh(L)
    if vecs.shape[1] < 3:
        return np.zeros((W.shape[0],2))
    E = vecs[:,1:3]
    E = E / (np.linalg.norm(E, axis=0, keepdims=True) + eps)
    return E

def shortest_path_hops(B, src):
    """Unweighted BFS hops from src on binary adjacency B."""
    N = B.shape[0]; dist = np.full(N, np.inf); dist[src]=0
    q = deque([src])
    while q:
        u = q.popleft()
        for v in np.where(B[u]>0)[0]:
            if np.isinf(dist[v]):
                dist[v] = dist[u]+1
                q.append(v)
    return dist

def fit_line(x, y):
    """OLS y ~ a + b*x; returns (a,b,R2)."""
    x = np.asarray(x); y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2: return np.nan, np.nan, np.nan
    X = np.c_[np.ones(mask.sum()), x[mask]]
    beta, *_ = np.linalg.lstsq(X, y[mask], rcond=None)
    yhat = X @ beta
    ss_res = np.sum((y[mask]-yhat)**2)
    ss_tot = np.sum((y[mask]-y[mask].mean())**2)
    R2 = 1.0 - ss_res/ss_tot if ss_tot>0 else np.nan
    return float(beta[0]), float(beta[1]), float(R2)

def kendall_tau(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2: return np.nan
    tau, _ = stats.kendalltau(x[mask], y[mask])
    return float(tau)

def analyze_run(run_dir):
    W = np.load(os.path.join(run_dir, "W.npy"))
    df_arr = pd.read_csv(os.path.join(run_dir, "arrival_times.csv"))
    T_i = df_arr["arrival_sec"].to_numpy(dtype=float)

    # Valid nodes (have arrivals)
    valid = np.isfinite(T_i)
    if valid.sum() < 3:
        return {"label": os.path.basename(os.path.normpath(run_dir)),
                "N_valid": int(valid.sum()),
                "hop_speed_hps": np.nan, "hop_R2": np.nan, "hop_tau": np.nan,
                "spec_speed_invsec": np.nan, "spec_R2": np.nan, "spec_tau": np.nan}

    # Epicenter = earliest arrival
    epic = int(np.nanargmin(T_i))
    B = (W > 0).astype(int)
    hops = shortest_path_hops(B, epic)

    # Hop model: T ~ a + b*hops
    a_h, b_h, R2_h = fit_line(hops[valid], T_i[valid])
    speed_hops = (1.0/b_h) if (np.isfinite(b_h) and b_h>1e-9) else np.nan
    tau_h = kendall_tau(hops[valid], T_i[valid])

    # Spectral-plane model
    E = spectral_embed_2d(W)
    x, y = E[:,0], E[:,1]
    # Fit plane: T ~ c + vx*x + vy*y
    Xmat = np.c_[np.ones(valid.sum()), x[valid], y[valid]]
    beta, *_ = np.linalg.lstsq(Xmat, T_i[valid], rcond=None)
    c, vx, vy = beta
    T_hat = Xmat @ beta
    ss_res = np.sum((T_i[valid]-T_hat)**2)
    ss_tot = np.sum((T_i[valid]-T_i[valid].mean())**2)
    R2_s = 1.0 - ss_res/ss_tot if ss_tot>0 else np.nan
    # Speed in embed units/sec is 1/||v||
    vnorm = np.hypot(vx, vy)
    spec_speed = (1.0/vnorm) if vnorm>1e-9 else np.nan
    # Rank correlation along the fitted direction
    proj = x*vx + y*vy
    tau_s = kendall_tau(proj[valid], T_i[valid])

    return {
        "label": os.path.basename(os.path.normpath(run_dir)),
        "N_valid": int(valid.sum()),
        "hop_speed_hps": float(speed_hops),
        "hop_R2": float(R2_h),
        "hop_tau": float(tau_h),
        "spec_speed_invsec": float(spec_speed),
        "spec_R2": float(R2_s),
        "spec_tau": float(tau_s),
    }

def plot_bars(df, cols, ylabel, out_path):
    labels = df["label"].tolist()
    vals = [df[c].to_numpy(dtype=float) for c in cols]
    x = np.arange(len(labels))
    width = 0.35

    fig = plt.figure(figsize=(8,3))
    ax = fig.add_subplot(111)
    for i, v in enumerate(vals):
        ax.bar(x + (i-0.5)*width, v, width, label=cols[i])
        for xi, yi in zip(x + (i-0.5)*width, v):
            if np.isfinite(yi):
                ax.text(xi, yi, f"{yi:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle=":", alpha=0.7)
    ax.legend(fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")

def main():
    ap = argparse.ArgumentParser(description="F7: Propagation speed & fit quality")
    ap.add_argument("--run_dirs", nargs="+", required=True,
                    help="results/<date> folders containing W.npy and arrival_times.csv")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="Optional custom labels (same length as run_dirs)")
    ap.add_argument("--out_dir", default="figs_F7", help="Where to save figures & CSV")
    args = ap.parse_args()

    rows = []
    for d in args.run_dirs:
        m = analyze_run(d)
        rows.append(m)
    df = pd.DataFrame(rows)

    # Optional custom labels
    if args.labels and len(args.labels) == len(df):
        df["label"] = args.labels

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "F7_propagation_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"[saved] {csv_path}")

    # Bars: speeds
    plot_bars(df, cols=["hop_speed_hps", "spec_speed_invsec"],
              ylabel="Speed (hops/s or embed-units/s)",
              out_path=os.path.join(args.out_dir, "F7a_speeds.png"))

    # Bars: fit quality
    plot_bars(df, cols=["hop_R2", "spec_R2"],
              ylabel="Fit R²",
              out_path=os.path.join(args.out_dir, "F7b_fitR2.png"))

    # Bars: rank coherence (optional third plot)
    plot_bars(df, cols=["hop_tau", "spec_tau"],
              ylabel="Kendall τ",
              out_path=os.path.join(args.out_dir, "F7c_kendall_tau.png"))

if __name__ == "__main__":
    main()

'''
python code/t-gcn/plot_f7_propagation.py \
  --run_dirs results/20110427 results/20110823 results/20110908 \
  --labels "2011-04-27" "2011-08-23" "2011-09-08" \
  --out_dir reports/figs_F7
'''