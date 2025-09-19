#!/usr/bin/env python3
import os, argparse, numpy as np, pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

import torch

# --- import your pipeline pieces ---
from preprocess_fnet import load_day_folder, estimate_event_time
from gcn_pipeline import (
    build_signal_graph, stack_node_features, train_tgcn,
    score_full_day, fit_nodewise_z, arrival_times
)

# ---- helpers (copied from report, simplified) ----
def largest_component_series(z, W, tau):
    T, N = z.shape
    Wb = (W > 0).astype(np.int8)
    series = np.zeros(T, dtype=int)
    for t in range(T):
        active = z[t] > tau
        if not np.any(active):
            series[t] = 0
        else:
            # simple DFS for largest comp
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

# ---- single fold (one day) evaluation ----
def eval_one_day(day_dir, params, device="cpu"):
    
    
    # Load and prep
    sites, data, dt = load_day_folder(day_dir)
    t_event = estimate_event_time(data)

    # Graph
    W, A_hat = build_signal_graph(
        sites, data, t_event,
        pre_window=params["pre_window"],
        guard=params["guard"],
        k=params["k"],
        lag_lambda=params["lag_lambda"]
    )

    # Features
    tvec, X = stack_node_features(sites, data, feature_cols=("r","roc_ps"))

    # Train T-GCN on pre-event windows
    model = train_tgcn(
        X, tvec, A_hat, t_event,
        lookback=params["lookback"], horizon=1,
        lr=params["lr"], weight_decay=params["weight_decay"],
        batch_size=params["batch_size"], epochs=params["epochs"],
        guard=params["guard"], device=device
    )

    # Score whole day and build z-scores
    err_r, _, _ = score_full_day(model, X, tvec,
                                 lookback=params["lookback"], horizon=1,
                                 dt=dt, alpha=0.3)
    z, mu, sd = fit_nodewise_z(err_r, tvec, t_event, guard=params["guard"])

    # Metrics at chosen thresholds
    tau = params["tau"]; persist_s = params["persist_s"]; M_min = params["M_min"]
    persist_steps = max(1, int(round(persist_s / max(dt, 1e-6))))
    largest = largest_component_series(z, W, tau)
    any_region = (largest >= M_min)

    # estimated onset proxy already computed (t_event)
    t_idx_first = first_persistent_exceed(any_region, persist_steps)
    t_detect = tvec[t_idx_first] if t_idx_first is not None else np.nan
    TTD = max(0.0, float(t_detect - t_event)) if np.isfinite(t_detect) else np.nan

    # FA/hour on pre-event only with refractory gap
    in_event = (np.abs(tvec - t_event) <= params["event_window_s"])
    pre_mask = (tvec < (t_event - params["event_window_s"]))
    kernel = np.ones(persist_steps, dtype=int)
    conv = np.convolve(any_region.astype(int), kernel, mode="same")
    persistent = conv >= persist_steps
    fa_starts_pre = burst_starts(persistent & pre_mask, dt, min_gap_s=30.0)
    if pre_mask.any():
        pre_hours = (tvec[pre_mask][-1] - tvec[pre_mask][0]) / 3600.0
        FA_per_hour_pre = float(len(fa_starts_pre) / pre_hours) if pre_hours > 0 else np.nan
    else:
        FA_per_hour_pre = np.nan

    # Node-level weak-label AUROC/AUPRC if available
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score

        # Broadcast time-only labels [T] -> [T, N] to match z
        W = float(params.get("event_window_s", 30.0))
        y_true_time = (np.abs(tvec - t_event) <= W).astype(np.int8)      # [T]
        y_true = np.broadcast_to(y_true_time[:, None], z.shape)          # [T, N]
        y_score = np.asarray(z, dtype=float)                              # [T, N]

        # Mask finite scores, then flatten both arrays consistently
        mask = np.isfinite(y_score)                                      # [T, N]
        if mask.any():
            y_true_f  = y_true[mask].ravel()
            y_score_f = y_score[mask].ravel()

            # Only compute if both classes are present
            if y_true_f.size > 0 and (y_true_f.min() != y_true_f.max()):
                AUROC = float(roc_auc_score(y_true_f, y_score_f))
                AUPRC = float(average_precision_score(y_true_f, y_score_f))
            else:
                AUROC = np.nan
                AUPRC = np.nan
        else:
            AUROC = np.nan
            AUPRC = np.nan

    except Exception:
        AUROC = np.nan
        AUPRC = np.nan

    # Footprint @ +5s after detection
    footprint = np.nan
    if np.isfinite(t_detect):
        t_plus5 = t_detect + 5.0
        idx5 = int(np.clip(np.searchsorted(tvec, t_plus5, side="left"), 0, len(tvec)-1))
        footprint = int(largest[idx5])

    # Diagnostics for weak labels and baseline
    dt = float(np.median(np.diff(tvec))) if len(tvec) > 1 else 0.1
    pos_frac = float(np.mean(np.abs(tvec - t_event) <= params["event_window_s"]))
    baseline_mask = (tvec <= (t_event - params["guard"]))
    baseline_len  = int(np.sum(baseline_mask))
    frac_finite_z = float(np.isfinite(z).mean())

    print(f"[debug {os.path.basename(day_dir)}] "
        f"W={params['event_window_s']:.1f}s | pos_frac={pos_frac:.2f} | "
        f"baseline_len={baseline_len} ({baseline_len*dt:.1f}s) | "
        f"z_finite={frac_finite_z:.2f}")

    return {
        "TTD_sec": TTD,
        "FA_per_hour_pre": FA_per_hour_pre,
        "AUROC_nodelevel": AUROC,
        "AUPRC_nodelevel": AUPRC,
        "footprint_size_at_detect_plus_5s": footprint,
        "N": X.shape[1],
        "dt": dt
    }

# ---- Optuna objective ----
def make_objective(day_dirs, device="cpu", multi_objective=False):
    def objective(trial):
        # --- search space ---
        params = {
            # graph
            "k": trial.suggest_int("k", 4, 12, step=2),
            "lag_lambda": trial.suggest_float("lag_lambda", 1.0, 4.0),
            "pre_window": trial.suggest_float("pre_window", 60.0, 180.0),
            "guard": trial.suggest_float("guard", 8.0, 15.0),
            # model
            "lookback": trial.suggest_int("lookback", 40, 120, step=10),
            "epochs": trial.suggest_int("epochs", 8, 25),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 96]),
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            # detection thresholds
            "tau": trial.suggest_float("tau", 2.0, 3.0),
            "persist_s": trial.suggest_float("persist_s", 1.0, 3.5),
            "M_min": trial.suggest_int("M_min", 4, 10),
            "event_window_s": trial.suggest_float("event_window_s", 20.0, 45.0),
        }

        # evaluate across all provided days and average
        metrics = [eval_one_day(d, params, device=device) for d in day_dirs]
        m = pd.DataFrame(metrics).mean(numeric_only=True).to_dict()

        # --- scalar objective (minimize) ---
        # normalize pieces so magnitudes are comparable
        TTD = m.get("TTD_sec", np.nan)
        FAh = m.get("FA_per_hour_pre", np.nan)
        AUPRC = m.get("AUPRC_nodelevel", np.nan)
        AUROC = m.get("AUROC_nodelevel", np.nan)

        # penalties for NaNs (no detection etc.)
        if not np.isfinite(TTD):  TTD = 10.0
        if not np.isfinite(FAh):  FAh = 10.0
        if not np.isfinite(AUPRC) or not np.isfinite(AUROC): return 10.0

        # lower is better: combine TTD, FA/h and (1-AUPRC)
        score = (TTD / 5.0) + 0.4 * (FAh / 5.0) + 0.6 * (1.0 - AUPRC)
        trial.set_user_attr("mean_TTD", float(TTD))
        trial.set_user_attr("mean_FAh", float(FAh))
        trial.set_user_attr("mean_AUPRC", float(AUPRC))
        trial.set_user_attr("mean_AUROC", float(AUROC))

        return score
    return objective

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day_dirs", nargs="+", required=True, help="paths to day folders (e.g., .../20110908 .../20110823 ...)")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--storage", type=str, default=None, help="optuna storage, e.g. sqlite:///tgcn_optuna.db")
    ap.add_argument("--study_name", type=str, default="tgcn_search")
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = ap.parse_args()

    sampler = TPESampler(seed=args.seed)
    pruner  = MedianPruner(n_warmup_steps=5)  # optional; works best if you report per-epoch

    study = optuna.create_study(direction="minimize",
                                sampler=sampler,
                                pruner=pruner,
                                storage=args.storage,
                                study_name=args.study_name,
                                load_if_exists=True)

    objective = make_objective(args.day_dirs, device=args.device)
    study.optimize(objective, n_trials=args.trials, gc_after_trial=True)

    print("\nBest trial:")
    bt = study.best_trial
    for k, v in bt.params.items():
        print(f"  {k}: {v}")
    print("  attrs:", {k: bt.user_attrs[k] for k in bt.user_attrs})

    # Save best params
    os.makedirs("optuna_results", exist_ok=True)
    with open("optuna_results/best_params.json", "w") as f:
        import json; json.dump(bt.params, f, indent=2)
    print("Saved best params to optuna_results/best_params.json")

if __name__ == "__main__":
    main()

'''
python optuna_tgcn_search.py \
  --day_dirs data/20110427 data/20110823 data/20110908 \
  --trials 30 --study_name tgcn_search
'''
