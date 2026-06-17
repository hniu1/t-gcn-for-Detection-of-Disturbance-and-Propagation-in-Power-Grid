"""Task 2 — model-free failure-vs-event discriminator.

Labels each flagged anomaly as 'sensor_failure', 'event', or 'ambiguous', using features computed on 
the raw recorded signal 'true' rather than on the forecast residual or any cluster grouping. 
The candidate pool is every cell whose residual exceeds a quantile, so isolated single-sensor 
failures are included.

Three independent tests on the raw signal:
    - Movement (flatline): windowed std of the sensor's own signal. A stuck sensor
    holds a constant (std ~ 0); a real event moves the measurement.
    - Coherence: do the sensor's graph neighbours deviate from their own baseline at
    the same time (dev = |true - median(true)|)? An event pushes the whole
    neighbourhood off baseline together.
    - Shape: residual against a moving median, |true - movmed|. Large for impulsive
    single-sample spikes, ~0 for a smooth swing. The threshold is anchored to the
    signal scale median|dev|, since a smooth signal's median-filter residual is ~0
    and anchoring to it would be degenerate. Disable with --disable_shape.

Label: flat -> sensor_failure; spiky -> sensor_failure; otherwise neighbours co-deviate -> event; 
else ambiguous (a lone, non-flat, non-spiky deviation that needs an additional feature).
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Model-free failure-vs-event discriminator on raw signal")
    p.add_argument("--results_dir", type=str, default="results_smoke_injected")
    p.add_argument("--split_name", type=str, default="pred", choices=["test", "pred"])
    p.add_argument("--adjacency_file", type=str, default="results/A_geo.npy")
    p.add_argument("--feature_idx", type=int, default=0, help="0 = freq_dev")
    p.add_argument("--cand_quantile", type=float, default=0.90, help="activation threshold on residual")
    p.add_argument("--window_radius", type=int, default=3, help="half-window for the flatline (std) test")
    p.add_argument("--flat_factor", type=float, default=0.25, help="flat if windowed std < factor * median std")
    p.add_argument("--dev_factor", type=float, default=1.5, help="a neighbor 'co-deviates' if dev > factor * median dev")
    p.add_argument("--coherence_min", type=float, default=0.3, help="coherent if this fraction of neighbors co-deviate")
    p.add_argument("--shape_factor", type=float, default=2.0, help="spiky if median-filter residual > factor * median|dev| (signal scale)")
    p.add_argument("--median_window", type=int, default=5, help="moving-median window")
    p.add_argument("--disable_shape", action="store_true", help="turn OFF the shape-based (spiky) feature")
    p.add_argument("--out_csv", type=str, default="downstream_tasks/task2_failure_vs_event/outputs/candidate_labels.csv")
    return p.parse_args()


def load_split_arrays(results_dir: Path, split_name: str) -> tuple[np.ndarray, np.ndarray]:
    pred = results_dir / f"{split_name}_predictions_original.npy"
    true = results_dir / f"{split_name}_targets_original.npy"
    if pred.exists() and true.exists():
        return np.load(pred), np.load(true)
    return np.load(results_dir / f"{split_name}_predictions.npy"), np.load(results_dir / f"{split_name}_targets.npy")


def windowed_std(x: np.ndarray, w: int) -> np.ndarray:
    """Per-cell std of x[:, s] over [t-w, t+w]. x is [S, N] -> [S, N]. ~0 for a frozen sensor."""
    S = x.shape[0]
    out = np.zeros_like(x)
    for t in range(S):
        lo, hi = max(0, t - w), min(S, t + w + 1)
        out[t] = x[lo:hi].std(axis=0)
    return out


def moving_median(x: np.ndarray, w: int) -> np.ndarray:
    """Per-cell moving median over a window of size w (odd). x is [S, N] -> [S, N].
    |x - moving_median(x)| is large for impulsive spikes, ~0 for a smooth swing."""
    S = x.shape[0]
    half = w // 2
    out = np.empty_like(x)
    for t in range(S):
        lo, hi = max(0, t - half), min(S, t + half + 1)
        out[t] = np.median(x[lo:hi], axis=0)
    return out


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    fidx = args.feature_idx

    pred, true = load_split_arrays(results_dir, args.split_name) # [S, H, N, F]
    pred_m = pred.mean(axis=1)[:, :, fidx] # [S, N]
    true_m = true.mean(axis=1)[:, :, fidx]
    resid = np.abs(pred_m - true_m) # detection signal
    N = true_m.shape[1]

    A = np.load(Path(args.adjacency_file)).astype(float)
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0.0)
    neighbors = {i: np.flatnonzero(A[i] > 0).tolist() for i in range(N)}

    # Flatline test source: windowed std of the sensor's own raw signal.
    std_w = windowed_std(true_m, args.window_radius)               # [S, N]
    flat_thr = max(1e-12, args.flat_factor * float(np.median(std_w)))

    # Coherence source: deviation of each sensor's raw signal from its own baseline.
    baseline = np.median(true_m, axis=0) # [N]
    dev = np.abs(true_m - baseline[None, :]) # [S, N]
    dev_thr = max(1e-12, args.dev_factor * float(np.median(dev)))

    # Shape source: residual against a moving median -> large for impulsive spikes.
    # Anchor the threshold to the signal scale median|dev| (NOT median(spike_resid): a smooth
    # signal has ~0 median-filter residual, which would make the threshold degenerate and flag
    # smooth event dips as spiky).
    spike_resid = np.abs(true_m - moving_median(true_m, args.median_window))   # [S, N]
    spike_thr = max(1e-12, args.shape_factor * float(np.median(dev)))

    # Candidate pool = activations (residual over quantile), singletons included.
    res_thr = float(np.quantile(resid, args.cand_quantile))
    cand_t, cand_s = np.where(resid > res_thr)

    use_shape = not args.disable_shape
    rows = []
    for t, s in zip(cand_t.tolist(), cand_s.tolist()):
        is_flat = bool(std_w[t, s] < flat_thr)
        is_spiky = bool(use_shape and spike_resid[t, s] > spike_thr)
        nbrs = neighbors[s]
        n_codev = int(sum(1 for j in nbrs if dev[t, j] > dev_thr))
        coherence = (n_codev / len(nbrs)) if nbrs else 0.0
        coherent = bool(coherence >= args.coherence_min)
        if is_flat:
            label = "sensor_failure"  # flatline (stuck)
        elif is_spiky:
            label = "sensor_failure" # impulsive (spiky/noisy)
        elif coherent:
            label = "event" # neighbors co-deviate
        else:
            label = "ambiguous"
        rows.append({
            "time_idx": t, "sensor_index": s, "residual": float(resid[t, s]),
            "std_w": float(std_w[t, s]), "is_flat": is_flat,
            "spike_resid": float(spike_resid[t, s]), "is_spiky": is_spiky,
            "dev": float(dev[t, s]), "n_codev_neighbors": n_codev, "n_neighbors": len(nbrs),
            "coherence": float(coherence), "spatially_coherent": coherent,
            "label": label,
        })

    out = pd.DataFrame(rows)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    n = len(out)
    counts = out["label"].value_counts().to_dict() if n else {}
    print(f"Discriminated {n} activations ({args.split_name}); shape clue {'OFF' if args.disable_shape else 'ON'}; "
          f"residual q{args.cand_quantile}={res_thr:.5f}, flat_thr={flat_thr:.5f}, dev_thr={dev_thr:.5f}, spike_thr={spike_thr:.5f}")
    for lab in ("event", "sensor_failure", "ambiguous"):
        c = counts.get(lab, 0)
        print(f"  {lab:15s}: {c:5d}" + (f" ({c/n:.1%})" if n else ""))
    if n:
        print("\n  2x2 (is_flat x spatially_coherent):")
        print(pd.crosstab(out["is_flat"], out["spatially_coherent"]).to_string())
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
