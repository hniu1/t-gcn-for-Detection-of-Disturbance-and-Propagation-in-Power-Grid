#!/usr/bin/env python3
"""
Runs T-GCN training/inference for ONE day and saves:

1) W.npy, A_hat.npy                  — graph adjacency (raw + normalized)
2) tvec.npy, X.npy                   — aligned time vector and features [T,N,2]
3) pred_r.npy                        — predicted next-step residuals [T,N]
4) err_r.npy                         — absolute forecast error (node anomaly) [T,N]
5) z.npy                             — node anomaly z-scores [T,N]
6) arrival_times.csv                 — per-site arrival time (sec + datetime)
7) sites.txt                         — site order used for arrays
8) run_config.json                   — full configuration + brief data summary
"""

import os, json, argparse
import numpy as np
import pandas as pd
import torch

# If you renamed the files, update these imports accordingly:
from preprocess_fnet import load_day_folder, estimate_event_time
from gcn_pipeline import (
    build_signal_graph,
    stack_node_features,
    train_tgcn,
    score_full_day,
    fit_nodewise_z,
    arrival_times,
)

# ---------- utils ----------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def maybe_load_params_json(args):
    """Override argparse values with a JSON params file (e.g., Optuna best_params.json)."""
    if args.params_json and os.path.exists(args.params_json):
        with open(args.params_json, "r") as f:
            P = json.load(f)
        for k, v in P.items():
            if hasattr(args, k):
                setattr(args, k, v)
        print(f"[params] loaded overrides from {args.params_json}")
    return args

# ---------- main ----------
def main(args):
    # allow JSON override (optional)
    args = maybe_load_params_json(args)

    set_seed(args.seed)
    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    print(f"[device] {device}")

    # 1) Load one day of data
    sites, data, dt = load_day_folder(args.data_dir)
    print(f"[load] {len(sites)} sites loaded, dt ≈ {dt:.3f}s")

    # 2) Estimate event time & build signal-driven graph
    t_event = estimate_event_time(data)
    print(f"[event] estimated onset at t ≈ {t_event:.2f}s")

    W, A_hat = build_signal_graph(
        sites, data, t_event,
        pre_window=args.pre_window,
        guard=args.guard,
        k=args.k,
        lag_lambda=args.lag_lambda,
    )
    print(f"[graph] built adjacency W (shape {W.shape}), normalized A_hat ready")

    # 3) Stack node features into [T, N, F]
    tvec, X = stack_node_features(sites, data, feature_cols=("r", "roc_ps"))
    T, N, F = X.shape
    print(f"[tensor] X shape = {(T, N, F)} [T,N,F], features=('r','roc_ps')")

    # 4) Train tiny T-GCN forecaster on pre-event windows
    model = train_tgcn(
        X, tvec, A_hat, t_event,
        lookback=args.lookback,
        horizon=args.horizon,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        guard=args.guard,
        device=device,
    )

    # 5) Score the full day → errors, z-scores, predictions
    err_r, _z_dummy, pred_r = score_full_day(
        model, X, tvec,
        lookback=args.lookback,
        horizon=args.horizon,
        dt=dt,
        alpha=0.3,
    )
    z, mu, sd = fit_nodewise_z(err_r, tvec, t_event, guard=args.guard)
    print(f"[score] z-scores computed using baseline t <= {t_event - args.guard:.2f}s")

    # 6) Arrival times for propagation analysis
    T_i = arrival_times(z, tvec, tau=args.tau, persist_s=args.persist_s, dt=dt)

    # 7) Save outputs
    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "W.npy"), W)
    np.save(os.path.join(args.out_dir, "A_hat.npy"), A_hat)
    np.save(os.path.join(args.out_dir, "tvec.npy"), tvec)
    np.save(os.path.join(args.out_dir, "X.npy"), X)
    np.save(os.path.join(args.out_dir, "err_r.npy"), err_r)
    np.save(os.path.join(args.out_dir, "z.npy"), z)
    np.save(os.path.join(args.out_dir, "pred_r.npy"), pred_r)
    np.save(os.path.join(args.out_dir, "arrival_times_sec.npy"), T_i)

    # Sites order
    with open(os.path.join(args.out_dir, "sites.txt"), "w") as f:
        for s in sites:
            f.write(s + "\n")

    # Also write a friendly CSV of arrival times (absolute + relative)
    base_dt = data[sites[0]]["DateTime"].iloc[0]
    abs_times = [(base_dt + pd.to_timedelta(float(tt), unit="s")) if np.isfinite(tt) else pd.NaT for tt in T_i]
    df_arr = pd.DataFrame({"site": sites, "arrival_sec": T_i, "arrival_datetime": abs_times})
    df_arr.to_csv(os.path.join(args.out_dir, "arrival_times.csv"), index=False)

    # Save run config for reproducibility (includes event_window_s even if not used here)
    run_cfg = {
        "data_dir": args.data_dir, "out_dir": args.out_dir, "seed": args.seed, "device": device,
        "N_sites": int(N), "dt": float(dt), "t_event": float(t_event),
        # Graph
        "pre_window": float(args.pre_window), "guard": float(args.guard),
        "k": int(args.k), "lag_lambda": float(args.lag_lambda),
        # Model/training
        "lookback": int(args.lookback), "horizon": int(args.horizon),
        "epochs": int(args.epochs), "batch_size": int(args.batch_size),
        "lr": float(args.lr), "weight_decay": float(args.weight_decay),
        # Detection thresholds (and reporting convenience)
        "tau": float(args.tau), "persist_s": float(args.persist_s),
        "M_min": int(args.M_min), "event_window_s": float(args.event_window_s),
        # Optional params file used
        "params_json": args.params_json,
    }
    with open(os.path.join(args.out_dir, "run_config.json"), "w") as f:
        json.dump(run_cfg, f, indent=2)

    print(f"[save] outputs written to {args.out_dir}")

    # 8) Quick console summary
    detected = [(s, tt) for s, tt in zip(sites, T_i) if np.isfinite(tt)]
    if detected:
        first_site, first_t = min(detected, key=lambda x: x[1])
        print(f"[summary] first arrival: {first_site} at {first_t:.2f}s")
    else:
        print("[summary] no arrivals exceeded threshold — consider lowering --tau or --persist_s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T-GCN spatio-temporal anomaly detection (train + export)")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to a single-day folder of txt files")
    parser.add_argument("--out_dir",  type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--params_json", type=str, default=None,
                        help="Optional JSON of hyperparameters (e.g., optuna_results/best_params.json)")

    # ---------- BEST-PARAM DEFAULTS (from Optuna) ----------
    # Graph building
    parser.add_argument("--pre_window", type=float, default=159.44850109823153,
                        help="Seconds of pre-event data for similarity")
    parser.add_argument("--guard", type=float, default=10.497273286855124,
                        help="Exclude last seconds before event to avoid leakage")
    parser.add_argument("--k", type=int, default=6, help="k-NN per node in adjacency")
    parser.add_argument("--lag_lambda", type=float, default=1.8140470953216878,
                        help="Lag penalty time constant (s)")

    # Model/training
    parser.add_argument("--lookback", type=int, default=60, help="History window length (samples)")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon (steps)")
    parser.add_argument("--epochs", type=int, default=17)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2.8691395365453717e-3)
    parser.add_argument("--weight_decay", type=float, default=2.0736445177905034e-4)

    # Detection thresholds (used for arrivals; reporting scripts also use M_min/event_window_s)
    parser.add_argument("--tau", type=float, default=2.198715681534172, help="Z-score threshold for node anomaly")
    parser.add_argument("--persist_s", type=float, default=1.013805292809006, help="Seconds required above threshold")
    parser.add_argument("--M_min", type=int, default=9, help="Min nodes in component for region alarm (for reporting)")
    parser.add_argument("--event_window_s", type=float, default=37.671433596190425,
                        help="± seconds around onset for weak labels (for reporting)")

    args = parser.parse_args()
    main(args)


'''
# Using baked-in best params
python main.py --data_dir data/20110427 --out_dir results/20110427

# Or explicitly point to the JSON we saved from Optuna
python main.py --data_dir data/20110823 --out_dir results/20110823 \
               --params_json optuna_results/best_params.json
'''