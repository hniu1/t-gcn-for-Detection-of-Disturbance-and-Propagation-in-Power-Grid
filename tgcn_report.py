#!/usr/bin/env python3
"""
Per day (event):

Node z-score heatmap (sites × time, ordered by arrival) → z_heatmap.png
Largest active component vs time (region alarm curve) → largest_component.png
Propagation embedding (2D spectral layout colored by arrival time) → propagation_embedding.png

Plus metrics files:

metrics.json / metrics.csv per day:
TTD, false alarms metrics, footprint size @ +5s, weak-label AUROC/AUPRC, estimated event time, first detection time, etc.
reports/metrics_all_runs.csv when you pass multiple run folders.
"""

import os, argparse, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional but nice for AUROC/AUPRC; falls back gracefully if missing
try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    HAVE_SK = True
except Exception:
    HAVE_SK = False

# ---------- helpers ----------
def apply_params_json_overrides(args):
    """
    If --params_json is provided, override CLI defaults with values in the JSON.
    Only keys we care about for reporting are overridden.
    """
    if not getattr(args, "params_json", None):
        return args
    if not os.path.exists(args.params_json):
        print(f"[warn] params_json not found: {args.params_json}")
        return args

    with open(args.params_json, "r") as f:
        P = json.load(f)

    # keys that affect reporting
    for k in ["tau", "persist_s", "M_min", "event_window_s",
              "footprint_max_delta", "footprint_step"]:
        if k in P:
            setattr(args, k, P[k])

    print("[params] using overrides from", args.params_json,
          f"(tau={args.tau}, persist_s={args.persist_s}, M_min={args.M_min}, "
          f"event_window_s={args.event_window_s}, "
          f"footprint_max_delta={args.footprint_max_delta}, footprint_step={args.footprint_step})")
    return args


def load_run(run_dir):
    req = ["W.npy", "A_hat.npy", "tvec.npy", "X.npy", "err_r.npy", "z.npy", "arrival_times.csv"]
    missing = [f for f in req if not os.path.exists(os.path.join(run_dir, f))]
    if missing:
        raise FileNotFoundError(f"Missing files in {run_dir}: {missing}")

    W = np.load(os.path.join(run_dir, "W.npy"))
    A_hat = np.load(os.path.join(run_dir, "A_hat.npy"))
    tvec = np.load(os.path.join(run_dir, "tvec.npy"))
    X = np.load(os.path.join(run_dir, "X.npy"))           # [T,N,F] with (r, roc_ps)
    err = np.load(os.path.join(run_dir, "err_r.npy"))     # [T,N]
    z = np.load(os.path.join(run_dir, "z.npy"))           # [T,N]
    df_arr = pd.read_csv(os.path.join(run_dir, "arrival_times.csv"))  # has 'site'
    sites = df_arr["site"].tolist()
    return W, A_hat, tvec, X, err, z, sites, df_arr

def estimate_event_time_from_X(tvec, X):
    # Use RoCoF (feature index 1): median of per-node argmax |RoCoF|
    roc = X[:, :, 1]  # [T,N]
    idxs = np.argmax(np.abs(roc), axis=0)            # [N]
    times = tvec[idxs]
    onset_percentile = 50.0
    return float(np.percentile(times, onset_percentile))

def connected_components_active(mask_nodes, W_thresh):
    """Return sizes of all connected components among active nodes."""
    N = W_thresh.shape[0]
    visited = np.zeros(N, dtype=bool)
    sizes = []
    for u in range(N):
        if not mask_nodes[u] or visited[u]:
            continue
        stack = [u]
        visited[u] = True
        sz = 0
        while stack:
            v = stack.pop()
            if not mask_nodes[v]:
                continue
            sz += 1
            neighbors = np.where(W_thresh[v] > 0)[0]
            for w in neighbors:
                if mask_nodes[w] and not visited[w]:
                    visited[w] = True
                    stack.append(w)
        sizes.append(sz)
    return sizes

def largest_component_series(z, W, tau):
    """Compute largest active component size over time given node z-scores."""
    T, N = z.shape
    Wb = (W > 0).astype(np.int8)
    series = np.zeros(T, dtype=int)
    for t in range(T):
        active = z[t] > tau
        if not np.any(active):
            series[t] = 0
        else:
            sizes = connected_components_active(active, Wb)
            series[t] = max(sizes) if sizes else 0
    return series

def first_persistent_exceed(series, persist_steps):
    """Index of first time series>=threshold persists for persist_steps; None if not found."""
    if persist_steps <= 1:
        idx = np.argmax(series)
        return int(idx) if series[idx] else None
    kernel = np.ones(persist_steps, dtype=int)
    conv = np.convolve(series.astype(int), kernel, mode="same")
    idxs = np.where(conv >= persist_steps)[0]
    return int(idxs[0]) if len(idxs) else None

def spectral_embed_2d(W, eps=1e-6):
    """Simple 2D spectral embedding from weighted adjacency (no external deps)."""
    d = W.sum(axis=1)
    d = np.clip(d, eps, None)
    Dm = np.diag(1.0 / np.sqrt(d))
    S = Dm @ W @ Dm
    L = np.eye(W.shape[0]) - S
    vals, vecs = np.linalg.eigh(L)
    if vecs.shape[1] < 3:
        return np.zeros((W.shape[0], 2))
    emb = vecs[:, 1:3]
    emb = emb / (np.linalg.norm(emb, axis=0, keepdims=True) + eps)
    return emb

# --- NEW: refractory burst counter ---
def burst_starts(mask_bool, dt, min_gap_s=30.0):
    """
    Count starts of True-runs with a refractory gap so nearby episodes merge.
    Returns indices of starts.
    """
    x = mask_bool.astype(int)
    starts = np.where(np.diff(np.r_[0, x]) == 1)[0]
    if min_gap_s is None:
        return starts
    min_gap = max(1, int(np.ceil(min_gap_s / max(dt, 1e-6))))
    keep, last = [], -10**9
    for s in starts:
        if s - last >= min_gap:
            keep.append(s)
            last = s
    return np.array(keep, dtype=int)

def footprint_curve(largest_comp, tvec, t0, deltas_sec, N_sites):
    """
    largest_comp: [T] largest active component size over time
    tvec:         [T] seconds
    t0:           reference time (usually detection time)
    deltas_sec:   list/array of Δ seconds
    N_sites:      int

    Returns arrays: sizes [len(deltas)], percents [len(deltas)]
    """
    sizes = []
    for d in deltas_sec:
        idx = np.searchsorted(tvec, t0 + d, side="left")
        idx = np.clip(idx, 0, len(tvec)-1)
        sizes.append(int(largest_comp[idx]))
    sizes = np.asarray(sizes, dtype=int)
    pcts  = sizes / float(N_sites) * 100.0
    return sizes, pcts

def plot_footprint_curve(t_detect, t_event, args, largest_comp, tvec, W, out_dir):
        # ----- footprint vs Δt after detection -----
    # Choose reference: detection if available, otherwise fall back to event time
    t0_for_footprint = t_detect if np.isfinite(t_detect) else t_event
    deltas = np.arange( args.footprint_step,
                        args.footprint_max_delta + 1e-9,
                        args.footprint_step, dtype=float )
    fp_sizes, fp_pcts = footprint_curve(largest_comp, tvec, t0_for_footprint,
                                        deltas, W.shape[0])

    # Save CSV
    df_fp = pd.DataFrame({
        "delta_sec": deltas,
        "footprint_nodes": fp_sizes,
        "footprint_pct": fp_pcts
    })
    df_fp.to_csv(os.path.join(out_dir, "footprint_vs_delta.csv"), index=False)

    # Plot line of % sites vs Δt
    plt.figure(figsize=(7, 3))
    plt.plot(deltas, fp_pcts)
    ref = "detect" if np.isfinite(t_detect) else "event"
    plt.title(f"Footprint vs Δt (after {ref})")
    plt.xlabel("Δt after detection (s)" if np.isfinite(t_detect) else "Δt after event (s)")
    plt.ylabel("Active region (% of sites)")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "footprint_vs_delta.png"), dpi=200, bbox_inches="tight")
    plt.close()


# ---------- metrics & plots for one run ----------
def analyze_one_run(run_dir, out_dir, tau=2.5, persist_s=2.0, M_min=5, event_window_s=30.0):
    os.makedirs(out_dir, exist_ok=True)
    W, A_hat, tvec, X, err, z, sites, df_arr = load_run(run_dir)

    # dt in seconds (sampling interval)
    dt = float(np.median(np.diff(tvec))) if len(tvec) > 1 else 0.1
    persist_steps = max(1, int(round(persist_s / dt)))

    # estimate event time (weak GT)
    t_event = estimate_event_time_from_X(tvec, X)

    # Largest active component series at each t
    largest_comp = largest_component_series(z, W, tau)  # [T]
    any_region = (largest_comp >= M_min)                 # Boolean time series

    # Time-to-detect (TTD)
    t_idx_first = first_persistent_exceed(any_region, persist_steps)
    if t_idx_first is not None:
        t_detect = tvec[t_idx_first]
        TTD = max(0.0, t_detect - t_event)
    else:
        t_detect, TTD = np.nan, np.nan

    # --- UPDATED: persistent mask computed regardless of detection ---
    kernel = np.ones(persist_steps, dtype=int)
    conv = np.convolve(any_region.astype(int), kernel, mode="same")
    persistent = conv >= persist_steps

    # --- UPDATED: windows for off-event / pre-event ---
    in_event = (np.abs(tvec - t_event) <= event_window_s)
    pre_mask = (tvec < (t_event - event_window_s))
    off_mask = ~in_event

    # --- ORIGINAL FA/hr over entire trace (kept for backward compatibility) ---
    fa_mask_all = persistent & (~in_event)
    fa_starts_all = burst_starts(fa_mask_all, dt, min_gap_s=30.0)
    hours_all = (tvec[-1] - tvec[0]) / 3600.0 if len(tvec) > 1 else 1.0
    fa_per_hour = (len(fa_starts_all) / hours_all) if hours_all > 0 else np.nan

    # --- NEW: safer FA metrics ---
    # pre-event only
    fa_mask_pre = persistent & pre_mask
    fa_starts_pre = burst_starts(fa_mask_pre, dt, min_gap_s=30.0)
    if pre_mask.any():
        pre_hours = (tvec[pre_mask][-1] - tvec[pre_mask][0]) / 3600.0
        pre_hours = pre_hours if pre_hours > 0 else np.nan
    else:
        pre_hours = np.nan
    FA_count_pre    = int(len(fa_starts_pre))
    FA_per_hour_pre = float(len(fa_starts_pre) / pre_hours) if np.isfinite(pre_hours) else np.nan

    # alarm time fractions
    alarm_time_off = float(np.sum(persistent & off_mask)) * dt
    off_time_total = float(np.sum(off_mask)) * dt
    Alarm_time_off_pct = 100.0 * alarm_time_off / off_time_total if off_time_total > 0 else np.nan

    alarm_time_on = float(np.sum(persistent & in_event)) * dt
    denom_alarm   = alarm_time_on + alarm_time_off
    Alarm_precision_pct = 100.0 * alarm_time_on / denom_alarm if denom_alarm > 0 else np.nan

    # Footprint size at +5s after onset (if detected)
    footprint_size = np.nan
    if t_idx_first is not None:
        t_plus5 = t_detect + 5.0
        idx5 = np.searchsorted(tvec, t_plus5, side="left")
        idx5 = np.clip(idx5, 0, len(tvec)-1)
        footprint_size = int(largest_comp[idx5])

    # Node-level ROC/PR (weak labels around t_event)
    auc, aupr = np.nan, np.nan
    if HAVE_SK:
        y_true_time = (np.abs(tvec - t_event) <= event_window_s).astype(int)      # [T]
        y_true = np.broadcast_to(y_true_time[:, None], z.shape)                    # [T, N]
        y_score = z  # higher z = more anomalous
        mask = np.isfinite(y_score)
        if mask.any():
            y_true_f  = y_true[mask].ravel()
            y_score_f = y_score[mask].ravel()
            if y_true_f.min() != y_true_f.max():
                try:
                    auc  = roc_auc_score(y_true_f, y_score_f)
                except Exception:
                    auc = np.nan
                try:
                    aupr = average_precision_score(y_true_f, y_score_f)
                except Exception:
                    aupr = np.nan

    # ----- plots -----
    arrival_sec = df_arr["arrival_sec"].to_numpy()
    order = np.argsort(np.where(np.isfinite(arrival_sec), arrival_sec, 1e12))
    z_ord = z[:, order].T  # [N,T]
    sites_ord = [sites[i] for i in order]

    plt.figure(figsize=(7, 4))
    extent = [tvec[0], tvec[-1], 0, len(sites_ord)]
    plt.imshow(z_ord, aspect='auto', interpolation='nearest', extent=extent, origin='lower')
    plt.axvline(t_event, linestyle='--')
    if np.isfinite(t_detect):
        plt.axvline(t_detect, linestyle=':')
    plt.colorbar(label='z-score')
    plt.yticks(np.linspace(0.5, len(sites_ord)-0.5, min(len(sites_ord), 10)),
               [sites_ord[int(i)][:-3] for i in np.linspace(0, len(sites_ord)-1, min(len(sites_ord), 10)).astype(int)])
    plt.xlabel('Time (s)')
    plt.ylabel('Sites (ordered by arrival)')
    plt.title('Node z-scores (ordered by arrival time)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "z_heatmap.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(5, 3.25))
    plt.plot(tvec, largest_comp, linewidth=1.2)
    plt.axhline(M_min, linestyle='--')
    plt.axvline(t_event, linestyle='--')
    if np.isfinite(t_detect):
        plt.axvline(t_detect, linestyle=':')
    plt.xlabel('Time (s)')
    plt.ylabel('Largest comp size')
    plt.title('Largest active component over time (z > tau)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "largest_component.png"), dpi=200)
    plt.close()

    emb = spectral_embed_2d(W)
    c = arrival_sec.copy()
    finite = np.isfinite(c)
    c_norm = c.copy()
    if finite.any():
        mx = np.nanmax(c)
        c_norm[~finite] = mx + 5.0
    plt.figure(figsize=(5, 3.25))
    sc = plt.scatter(emb[:,0], emb[:,1], c=c_norm, s=40)
    plt.colorbar(sc, label='Arrival time (s)')
    for i, s in enumerate(sites):
        plt.text(emb[i,0], emb[i,1], s[:-3], fontsize=6, ha='center', va='center')
    # plt.title('Propagation (spectral layout)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "propagation_embedding.png"), dpi=200)
    plt.close()

    plot_footprint_curve(t_detect, t_event, args, largest_comp, tvec, W, out_dir)

    # ----- save metrics -----
    N = int(W.shape[0])
    metrics = {
        "run_dir": run_dir,
        "T": int(len(tvec)),
        "N": N,
        "num_sites": N,  # NEW: for plotting normalized footprints
        "dt": float(dt),
        "tau": float(tau),
        "persist_s": float(persist_s),
        "M_min": int(M_min),
        "event_window_s": float(event_window_s),
        "t_event": float(t_event),
        "t_detect": float(t_detect) if np.isfinite(t_detect) else None,
        "TTD_sec": float(TTD) if np.isfinite(TTD) else None,

        # False-alarm metrics
        "false_alarms_per_hour": float(fa_per_hour),          # legacy (whole-trace)
        "FA_count_pre": int(FA_count_pre),                    # NEW
        "FA_per_hour_pre": float(FA_per_hour_pre) if np.isfinite(FA_per_hour_pre) else None,  # NEW
        "Alarm_time_off_pct": float(Alarm_time_off_pct) if np.isfinite(Alarm_time_off_pct) else None,  # NEW
        "Alarm_precision_pct": float(Alarm_precision_pct) if np.isfinite(Alarm_precision_pct) else None,  # NEW

        # Footprint
        "footprint_size_at_detect_plus_5s": int(footprint_size) if np.isfinite(footprint_size) else None,
        "footprint_frac_at_detect_plus_5s": (float(footprint_size)/N*100.0) if np.isfinite(footprint_size) and N>0 else None,

        # Node-level separability
        "AUROC_nodelevel": float(auc) if np.isfinite(auc) else None,
        "AUPRC_nodelevel": float(aupr) if np.isfinite(aupr) else None,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame([metrics]).to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    print(f"[report] wrote metrics & plots to {out_dir}")
    return metrics

# ---------- CLI ----------
def main():
    global args
    ap = argparse.ArgumentParser(description="T-GCN report: metrics and plots")
    ap.add_argument("--in_dirs", nargs="+", required=True,
                    help="One or more run directories that contain saved npys/csvs")
    ap.add_argument("--out_dir", type=str, default="reports", help="Where to write report folders")
    # operating-point knobs
    ap.add_argument("--tau", type=float, default=2.5, help="Node z-score threshold")
    ap.add_argument("--persist_s", type=float, default=1.0, help="Seconds required above threshold")
    ap.add_argument("--M_min", type=int, default=5, help="Min nodes in connected component for region alarm")
    ap.add_argument("--event_window_s", type=float, default=30.0,
                    help="Label window (+/- seconds) for ROC/PR")
    # footprint curve controls
    ap.add_argument("--footprint_max_delta", type=float, default=10.0,
                    help="Max Δt (sec) after detection for footprint curve")
    ap.add_argument("--footprint_step", type=float, default=0.1,
                    help="Step (sec) for Δt after detection")
    # NEW: optuna/best params file
    ap.add_argument("--params_json", type=str, default=None,
                    help="JSON with best params (e.g., optuna_results/best_params.json)")
    args = ap.parse_args()

    # override from JSON if provided
    args = apply_params_json_overrides(args)

    os.makedirs(args.out_dir, exist_ok=True)

    all_rows = []
    for run in args.in_dirs:
        name = os.path.basename(os.path.normpath(run))
        out = os.path.join(args.out_dir, name)
        m = analyze_one_run(run, out,
                            tau=args.tau, persist_s=args.persist_s,
                            M_min=args.M_min, event_window_s=args.event_window_s)
        all_rows.append(m)

    # write the aggregate table once at the top-level out_dir
    df = pd.DataFrame(all_rows)
    agg_path = os.path.join(args.out_dir, "metrics_all_runs.csv")
    df.to_csv(agg_path, index=False)
    print(f"[aggregate] wrote {agg_path}")


if __name__ == "__main__":
    main()
