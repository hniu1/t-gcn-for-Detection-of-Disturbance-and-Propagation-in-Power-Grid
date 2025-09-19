#!/usr/bin/env python3
import os, argparse, numpy as np, pandas as pd
import matplotlib.pyplot as plt

# ---------- small helpers (mirroring tgcn_report logic) ----------
def estimate_event_time_from_X(tvec, X):
    # RoCoF channel (feature 1), median of per-node argmax |RoCoF|
    roc = X[:, :, 1]
    idxs = np.argmax(np.abs(roc), axis=0)
    return float(np.median(tvec[idxs]))

def largest_component_series(z, W, tau):
    T, N = z.shape
    Wb = (W > 0).astype(np.int8)
    series = np.zeros(T, dtype=int)
    for t in range(T):
        active = z[t] > tau
        if not np.any(active):
            series[t] = 0
            continue
        visited = np.zeros(N, dtype=bool); best = 0
        for u in np.where(active)[0]:
            if visited[u]: continue
            stack = [u]; visited[u] = True; sz = 0
            while stack:
                v = stack.pop()
                if not active[v]: continue
                sz += 1
                for w in np.where(Wb[v] > 0)[0]:
                    if active[w] and not visited[w]:
                        visited[w] = True; stack.append(w)
            best = max(best, sz)
        series[t] = best
    return series

def first_persistent_exceed(series_bool, persist_steps):
    if persist_steps <= 1:
        idx = int(np.argmax(series_bool))
        return idx if series_bool[idx] else None
    kernel = np.ones(persist_steps, dtype=int)
    conv = np.convolve(series_bool.astype(int), kernel, mode="same")
    idxs = np.where(conv >= persist_steps)[0]
    return int(idxs[0]) if len(idxs) else None

def burst_starts(mask_bool, dt, min_gap_s=30.0):
    x = mask_bool.astype(int)
    starts = np.where(np.diff(np.r_[0, x]) == 1)[0]
    if min_gap_s is None:
        return starts
    min_gap = max(1, int(np.ceil(min_gap_s / max(dt, 1e-6))))
    keep, last = [], -10**9
    for s in starts:
        if s - last >= min_gap:
            keep.append(s); last = s
    return np.array(keep, dtype=int)

# ---------- sweep & plot ----------
def sweep_tau(run_dir, Ms, tau_vals, persist_s=2.0, event_window_s=30.0):
    W = np.load(os.path.join(run_dir, "W.npy"))
    tvec = np.load(os.path.join(run_dir, "tvec.npy"))
    X = np.load(os.path.join(run_dir, "X.npy"))
    z = np.load(os.path.join(run_dir, "z.npy"))

    dt = float(np.median(np.diff(tvec))) if len(tvec) > 1 else 0.1
    t_event = estimate_event_time_from_X(tvec, X)
    persist_steps = max(1, int(round(persist_s / dt)))
    in_event = (np.abs(tvec - t_event) <= event_window_s)
    pre_mask = (tvec < (t_event - event_window_s))

    TTDs, FAh = [], []
    for tau in tau_vals:
        largest = largest_component_series(z, W, tau)
        any_region = (largest >= Ms)

        # persistent mask
        kernel = np.ones(persist_steps, dtype=int)
        conv = np.convolve(any_region.astype(int), kernel, mode="same")
        persistent = conv >= persist_steps

        # TTD
        idx = first_persistent_exceed(any_region, persist_steps)
        t_detect = tvec[idx] if idx is not None else np.nan
        TTDs.append(max(0.0, t_detect - t_event) if np.isfinite(t_detect) else np.nan)

        # pre-event FA/hour
        fa_starts_pre = burst_starts(persistent & pre_mask, dt, min_gap_s=30.0)
        if pre_mask.any():
            pre_hours = (tvec[pre_mask][-1] - tvec[pre_mask][0]) / 3600.0
            FAh.append((len(fa_starts_pre) / pre_hours) if pre_hours > 0 else np.nan)
        else:
            FAh.append(np.nan)
    return np.array(TTDs, float), np.array(FAh, float)

def sweep_Mmin(run_dir, tau, M_vals, persist_s=2.0, event_window_s=30.0):
    W = np.load(os.path.join(run_dir, "W.npy"))
    tvec = np.load(os.path.join(run_dir, "tvec.npy"))
    X = np.load(os.path.join(run_dir, "X.npy"))
    z = np.load(os.path.join(run_dir, "z.npy"))

    dt = float(np.median(np.diff(tvec))) if len(tvec) > 1 else 0.1
    t_event = estimate_event_time_from_X(tvec, X)
    persist_steps = max(1, int(round(persist_s / dt)))
    in_event = (np.abs(tvec - t_event) <= event_window_s)
    pre_mask = (tvec < (t_event - event_window_s))

    TTDs, FAh = [], []
    largest = largest_component_series(z, W, tau)  # reuse for all M
    kernel = np.ones(persist_steps, dtype=int)

    for M in M_vals:
        any_region = (largest >= M)
        conv = np.convolve(any_region.astype(int), kernel, mode="same")
        persistent = conv >= persist_steps

        idx = first_persistent_exceed(any_region, persist_steps)
        t_detect = tvec[idx] if idx is not None else np.nan
        TTDs.append(max(0.0, t_detect - t_event) if np.isfinite(t_detect) else np.nan)

        fa_starts_pre = burst_starts(persistent & pre_mask, dt, min_gap_s=30.0)
        if pre_mask.any():
            pre_hours = (tvec[pre_mask][-1] - tvec[pre_mask][0]) / 3600.0
            FAh.append((len(fa_starts_pre) / pre_hours) if pre_hours > 0 else np.nan)
        else:
            FAh.append(np.nan)
    return np.array(TTDs, float), np.array(FAh, float)

def plot_f5(run_dir, out_path, tau_vals=None, M_vals=None,
            M_fix=5, tau_fix=2.5, persist_s=2.0, event_window_s=30.0, label=None):
    if tau_vals is None: tau_vals = np.arange(2.0, 3.2+1e-9, 0.2)
    if M_vals is None:   M_vals   = np.arange(4, 16+1e-9, 2)

    if label is None:
        label = os.path.basename(os.path.normpath(run_dir))

    TTD_tau, FAh_tau = sweep_tau(run_dir, M_fix, tau_vals, persist_s, event_window_s)
    TTD_M,   FAh_M   = sweep_Mmin(run_dir, tau_fix, M_vals, persist_s, event_window_s)

    fig = plt.figure(figsize=(9, 3.2))

    # --- Panel (a): vary tau ---
    ax1 = fig.add_subplot(1,2,1)
    l1, = ax1.plot(tau_vals, TTD_tau, marker='o')
    ax1.set_xlabel(r'$\tau$ (z-score threshold)')
    ax1.set_ylabel('TTD (s)', color=l1.get_color())
    ax1.tick_params(axis='y', labelcolor=l1.get_color())
    ax1.grid(True, axis='both', linestyle=':', alpha=0.7)
    ax1.set_title(f"Sensitivity to $\\tau$ (fix $M_{{\\min}}$={M_fix}) — {label}")

    ax1b = ax1.twinx()
    l2, = ax1b.plot(tau_vals, FAh_tau, marker='s', linestyle='--', color='green')
    ax1b.set_ylabel('Pre-event FA/hour', color=l2.get_color())
    ax1b.tick_params(axis='y', labelcolor=l2.get_color())

    # --- Panel (b): vary Mmin ---
    ax2 = fig.add_subplot(1,2,2)
    l3, = ax2.plot(M_vals, TTD_M, marker='o')
    ax2.set_xlabel(r'$M_{\min}$ (nodes)')
    ax2.set_ylabel('TTD (s)', color=l3.get_color())
    ax2.tick_params(axis='y', labelcolor=l3.get_color())
    ax2.grid(True, axis='both', linestyle=':', alpha=0.7)
    ax2.set_title(f"Sensitivity to $M_{{\\min}}$ (fix $\\tau$={tau_fix}) — {label}")

    ax2b = ax2.twinx()
    l4, = ax2b.plot(M_vals, FAh_M, marker='s', linestyle='--', color='green')
    ax2b.set_ylabel('Pre-event FA/hour', color=l4.get_color())
    ax2b.tick_params(axis='y', labelcolor=l4.get_color())

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="F5: Sensitivity to τ and M_min (lines) using a saved run_dir")
    ap.add_argument("--run_dir", required=True, help="results/<date> directory containing W.npy, tvec.npy, X.npy, z.npy")
    ap.add_argument("--out", default="figs_F5/F5_sensitivity.png", help="output image path")
    ap.add_argument("--tau_fix", type=float, default=2.5, help="τ used when sweeping M_min")
    ap.add_argument("--M_fix", type=int, default=5, help="M_min used when sweeping τ")
    ap.add_argument("--persist_s", type=float, default=2.0, help="persistence seconds")
    ap.add_argument("--event_window_s", type=float, default=30.0, help="± seconds around onset to define event window")
    ap.add_argument("--tau_vals", type=str, default=None,
                    help="comma-separated list (e.g., '2.0,2.2,2.4,...'). If omitted, uses 2.0:0.2:3.2")
    ap.add_argument("--M_vals", type=str, default=None,
                    help="comma-separated list (e.g., '4,6,8,10,12'). If omitted, uses 4:2:16")
    args = ap.parse_args()

    tau_vals = None if args.tau_vals is None else np.array([float(x) for x in args.tau_vals.split(",")], float)
    M_vals   = None if args.M_vals   is None else np.array([int(x) for x in args.M_vals.split(",")], int)

    plot_f5(args.run_dir, args.out, tau_vals, M_vals,
            M_fix=args.M_fix, tau_fix=args.tau_fix,
            persist_s=args.persist_s, event_window_s=args.event_window_s)

if __name__ == "__main__":
    main()

'''
python plot_f5_sensitivity.py \
  --run_dir results/20110908 \
  --out figs_F5/F5_2011-09-08.png \
  --tau_fix 2.5 --M_fix 5 \
  --persist_s 2.0 --event_window_s 30
'''