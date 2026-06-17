"""
Task 3: Synthetic Fault Injection

Create ground truth by corrupting the recorded 'true' array and leaving 'pred' unchanged.
The model predicted normal behavior, so residual = |pred - true (corrupted)| spikes exactly where the faults were injected.

Fault categories:
    - stuck failure: one sensor frozen at a constant value over a period of time (flat, isolated)
    - spiky/noisy failure: one sensor with sudden single-sample spikes (moved, isolated, non-smooth)
    - coherent physical event: a seed sensor + geo-neighbors get the same smooth dip (moved, spatially coherent)

Output: 
    - A results directory with 'pred' array copied unchanged and 'true' corrupted
    - injected_truth.csv

NOTE: this is target-only. The fault is not fed into the model input, thus preventing message-passing of the fault information by the forecast model. That requires input-level injection and re-inference.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import shutil

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inject synthetic faults into the recorded 'true' array.")
    p.add_argument("--results_dir", type=str, help="Directory to save the results.")
    p.add_argument("--split_name", type=str, default="pred", choices=["test", "pred"])
    p.add_argument("--out_dir", type=str, default="results_injected")
    p.add_argument("--sensor_order_file", type=str, default="results/sensor_order.npy")
    p.add_argument("--adjacency_file", type=str, default="results/A_geo.npy")
    p.add_argument("--out_truth", type=str, default="downstream_tasks/task3_fault_injection/outputs/injected_truth.csv")
    p.add_argument("--feature_idx", type=int, default=0, help="0 = freq_dev")
    p.add_argument("--n_failures", type=int, default=10)
    p.add_argument("--n_spiky", type=int, default=10)
    p.add_argument("--n_events", type=int, default=10)
    p.add_argument("--span", type=int, default=20, help="duration of each injection in samples")
    p.add_argument("--amplitude_k", type=float, default=3.0, help="amplitude = k * q0.90 of baseline residual")
    p.add_argument("--event_group_cap", type=int, default=5, help="max sensors per event (seed + neighbors)")
    p.add_argument("--margin", type=int, default=15, help="keep injections away from split edges")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def load_split_arrays(results_dir: Path, split_name: str) -> tuple[np.ndarray, np.ndarray]:
    pred = results_dir / f"{split_name}_predictions_original.npy"
    true = results_dir / f"{split_name}_targets_original.npy"
    if pred.exists() and true.exists():
        return np.load(pred), np.load(true)
    return np.load(results_dir / f"{split_name}_predictions.npy"), np.load(results_dir / f"{split_name}_targets.npy")

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    fidx = args.feature_idx

    pred, true = load_split_arrays(results_dir, args.split_name) # [S, H, N, F]
    S, H, N, F = pred.shape # [samples (windows), horizon, nodes, features]

    # Baseline residual (horizon-mean, freq) -> rank sensors by cleanliness, set amplitude.
    pred_m = pred.mean(axis=1)[:, :, fidx] # [S, N]
    true_m = true.mean(axis=1)[:, :, fidx]
    resid = np.abs(pred_m - true_m) # [S, N]
    clean_score = resid.mean(axis=0) # [N] score = mean abs error across all samples
    amplitude = float(args.amplitude_k * np.quantile(resid, 0.9)) # size of injected fault signal

    sensor_ids = np.load(Path(args.sensor_order_file)).astype(int)
    A = np.load(Path(args.adjacency_file)).astype(float)
    A = np.maximum(A, A.T) # make symmetric
    np.fill_diagonal(A, 0.0) # remove self-loops
    neighbors = {i: np.flatnonzero(A[i] > 0).tolist() for i in range(N)}

    pool = list(np.argsort(clean_score)) # sort by cleanest sensors first
    used: set[int] = set()
    true_inj = true.copy()
    records = []
    inj_id = 0

    def rand_t0() -> int: # generate a random start time for the fault
        return int(rng.integers(args.margin, S - args.span - args.margin))

    # == stuck failures (isolated sensor, frozen to a constant value) ==
    fail_sensors = [s for s in pool if s not in used][:args.n_failures]
    for s in fail_sensors:
        used.add(s)
        t0 = rand_t0()
        t1 = t0 + args.span
        base = float(true[t0, 0, s, fidx])
        true_inj[t0:t1, :, s, fidx] = base + amplitude
        records.append({
            "injection_id": inj_id,
            "kind": "failure",
            "subtype": "stuck",
            "sensor_indices": str(s),
            "sensor_ids": str(int(sensor_ids[s])),
            "t0": t0, 
            "t1": t1,
            "span": args.span,
            "amplitude": amplitude
        })
        inj_id += 1

    # == spiky/noisy failure: (isolated, sudden spikes) ==
    spike_sensors = [s for s in pool if s not in used][:args.n_spiky]
    for s in spike_sensors:
        used.add(s)
        t0 = rand_t0()
        t1 = t0 + args.span
        idxs = np.arange(t0, t1, 2) # every other sample so spikes are non-adjacent
        signs = rng.choice([-1.0, 1.0], size=len(idxs)) # random directions for spikes
        true_inj[idxs, :, s, fidx] += (signs * amplitude * 1.5)[:, None] # scale spikes to be more pronounced
        records.append({
            "injection_id": inj_id,
            "kind": "failure",
            "subtype": "spiky",
            "sensor_indices": str(s),
            "sensor_ids": str(int(sensor_ids[s])),
            "t0": t0,
            "t1": t1,
            "span": args.span,
            "amplitude": amplitude
        })
        inj_id += 1

    # == coherent events (spatially coherent, same dip-and-recovery) ==
    k = np.arange(args.span)
    dip = amplitude * np.sin(np.pi * k / args.span) # dip and recovery pattern (0 -> amplitude -> 0)
    events_made = 0
    for seed in pool:
        if events_made >= args.n_events:
            break
        if seed in used or len(neighbors[seed]) < 2:
            continue
        group = [seed] + [j for j in neighbors[seed] if j not in used]
        group = group[:args.event_group_cap]
        if len(group) < 2:
            continue
        for g in group:
            used.add(g)
        t0 = rand_t0()
        t1 = t0 + args.span
        for g in group:
            true_inj[t0:t1, :, g, fidx] -= dip[:, None]
        records.append({
            "injection_id": inj_id,
            "kind": "event",
            "subtype": "coherent_dip",
            "sensor_indices": ";".join(str(g) for g in group),
            "sensor_ids": ";".join(str(int(sensor_ids[g])) for g in group),
            "t0": t0,
            "t1": t1,
            "span": args.span,
            "amplitude": amplitude
        })
        inj_id += 1
        events_made += 1

    # write injected results dir (copy original, overwrite with injected data)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in results_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, out_dir / f.name) # copy file with metadata
    np.save(out_dir / f"{args.split_name}_targets_original.npy", true_inj)

    truth_path = Path(args.out_truth)
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(truth_path, index=False)

    n_stuck = sum(r["subtype"] == "stuck" for r in records)
    n_spiky = sum(r["subtype"] == "spiky" for r in records)
    n_evt = sum(r["kind"] == "event" for r in records)
    print(f"Injected into {args.split_name} ({S} samples x {N} sensors):")
    print(f"  amplitude = {amplitude:.6f}  (= {args.amplitude_k} x q0.90 baseline residual)")
    print(f"  stuck failures : {n_stuck}")
    print(f"  spiky failures : {n_spiky}")
    print(f"  coherent events: {n_evt}  (group sizes {[len(str(r['sensor_indices']).split(';')) for r in records if r['kind']=='event']})")
    print(f"  corrupted dir  -> {out_dir}")
    print(f"  ground truth   -> {truth_path}")


if __name__ == "__main__":
    main()