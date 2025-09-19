#!/usr/bin/env python3
import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def infer_label(csv_path, df):
    # Prefer run_dir column if present
    if "run_dir" in df.columns and isinstance(df["run_dir"].iloc[0], str):
        return os.path.basename(os.path.normpath(df["run_dir"].iloc[0]))
    # Else parent folder name, else file stem
    parent = os.path.basename(os.path.dirname(csv_path))
    if parent:
        return parent
    return os.path.splitext(os.path.basename(csv_path))[0]

def load_rows(inputs, labels):
    rows = []
    for i, path in enumerate(inputs):
        df = pd.read_csv(path)
        if len(df) == 0:
            continue
        row = df.iloc[0].to_dict()
        row["_label"] = labels[i] if labels and i < len(labels) else infer_label(path, df)
        rows.append(row)
    out = pd.DataFrame(rows)

    # Compute extras if possible:
    if "footprint_size_at_detect_plus_5s" in out.columns and "num_sites" in out.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            out["footprint_frac_at_plus5"] = (
                out["footprint_size_at_detect_plus_5s"].astype(float) /
                out["num_sites"].astype(float)
            ) * 100.0  # percent
    return out

def plot_bar(df, key, ylabel, out_path, fmt):
    labels = df["_label"].tolist()
    vals = df.get(key, pd.Series([np.nan]*len(df))).astype(float).to_numpy()

    x = np.arange(len(labels))
    width = 0.65

    fig = plt.figure(figsize=(6, 3.0))
    ax = fig.add_subplot(111)
    bars = ax.bar(x, np.nan_to_num(vals, nan=0.0), width)

    # Annotate bars (handle NaN as "NA")
    for rect, v in zip(bars, vals):
        txt = ("NA" if not np.isfinite(v) else fmt.format(v))
        ax.text(rect.get_x() + rect.get_width()/2.0, rect.get_height(),
                txt, ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Event")
    ax.set_title(ylabel)
    ax.grid(axis='y', linestyle=':', linewidth=0.8, alpha=0.7)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")

def main():
    ap = argparse.ArgumentParser(description="Plot F2 cross-event bars from metrics_all_runs.csv files")
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="Paths to metrics_all_runs.csv (one per event)")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="Optional custom labels (same length as inputs)")
    ap.add_argument("--out_dir", default="figs_F2", help="Where to save figures")
    args = ap.parse_args()

    df = load_rows(args.inputs, args.labels)

    # ---- Dynamically choose FA metric (prefer safer ones if present) ----
    fa_key, fa_ylabel, fa_file, fa_fmt = None, None, None, None
    if "FA_per_hour_pre" in df.columns:
        fa_key, fa_ylabel, fa_file, fa_fmt = "FA_per_hour_pre", "False alarms / hour (pre-event)", "F2b_FA_per_hour_pre.png", "{:.2f}"
    elif "Alarm_time_off_pct" in df.columns:
        fa_key, fa_ylabel, fa_file, fa_fmt = "Alarm_time_off_pct", "Off-event alarm time (%)", "F2b_AlarmTimeOffPct.png", "{:.1f}%"
    elif "false_alarms_per_hour" in df.columns:
        fa_key, fa_ylabel, fa_file, fa_fmt = "false_alarms_per_hour", "False alarms / hour", "F2b_FA_per_hour.png", "{:.2f}"

    # ---- Build the metric list to plot ----
    plots = []

    # TTD
    if "TTD_sec" in df.columns:
        plots.append(("TTD_sec", "Time-to-detect (s)", "F2a_TTD.png", "{:.1f}"))

    # FA metric (if any)
    if fa_key is not None:
        plots.append((fa_key, fa_ylabel, fa_file, fa_fmt))

    # Footprint size
    if "footprint_size_at_detect_plus_5s" in df.columns:
        plots.append(("footprint_size_at_detect_plus_5s", "Footprint @ +5s (nodes)", "F2c_Footprint.png", "{:.0f}"))

    # Normalized footprint (optional but recommended)
    if "footprint_frac_at_plus5" in df.columns:
        plots.append(("footprint_frac_at_plus5", "Footprint @ +5s (% of sites)", "F2d_FootprintFrac.png", "{:.1f}%"))

    # Sanity print
    cols_to_print = ["_label"] + [p[0] for p in plots]
    print(df[[c for c in cols_to_print if c in df.columns]])

    # Make plots
    os.makedirs(args.out_dir, exist_ok=True)
    for key, ylabel, fname, fmt in plots:
        out_path = os.path.join(args.out_dir, fname)
        plot_bar(df, key, ylabel, out_path, fmt)

if __name__ == "__main__":
    main()

# Example usage:
# python plot_f2_metrics.py \
#   --inputs reports/20110427/metrics_all_runs.csv \
#            reports/20110823/metrics_all_runs.csv \
#            reports/20110908/metrics_all_runs.csv \
#   --labels "2011-04-27" "2011-08-23" "2011-09-08" \
#   --out_dir figs_F2
