#!/usr/bin/env python3
import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def infer_label(csv_path, df):
    if "run_dir" in df.columns and isinstance(df["run_dir"].iloc[0], str):
        return os.path.basename(os.path.normpath(df["run_dir"].iloc[0]))
    parent = os.path.basename(os.path.dirname(csv_path))
    return parent if parent else os.path.splitext(os.path.basename(csv_path))[0]

def load_rows(inputs, labels):
    rows = []
    for i, path in enumerate(inputs):
        df = pd.read_csv(path)
        if len(df) == 0: continue
        row = df.iloc[0].to_dict()
        row["_label"] = labels[i] if labels and i < len(labels) else infer_label(path, df)
        rows.append(row)
    return pd.DataFrame(rows)

def bar_with_labels(ax, x, vals, labels, ylabel, fmt):
    bars = ax.bar(x, np.nan_to_num(vals, nan=0.0), width=0.65)
    for rect, v in zip(bars, vals):
        txt = "NA" if not np.isfinite(v) else fmt.format(v)
        ax.text(rect.get_x()+rect.get_width()/2.0, rect.get_height(), txt,
                ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel(ylabel); ax.grid(axis='y', linestyle=':', linewidth=0.8, alpha=0.7)

def main():
    ap = argparse.ArgumentParser(description="F3: AUROC/AUPRC bars from metrics_all_runs.csv (one per event)")
    ap.add_argument("--inputs", nargs="+", required=True, help="paths to metrics_all_runs.csv")
    ap.add_argument("--labels", nargs="*", default=None, help="optional custom labels")
    ap.add_argument("--out", default="figs_F3/F3_AUROC_AUPRC.png", help="output image path")
    args = ap.parse_args()

    df = load_rows(args.inputs, args.labels)
    if df.empty:
        raise SystemExit("No rows loaded. Check --inputs.")

    events = df["_label"].tolist()
    auroc = df.get("AUROC_nodelevel", pd.Series([np.nan]*len(df))).astype(float).to_numpy()
    auprc = df.get("AUPRC_nodelevel", pd.Series([np.nan]*len(df))).astype(float).to_numpy()

    fig = plt.figure(figsize=(8, 3.2))
    ax1 = fig.add_subplot(1,2,1)
    bar_with_labels(ax1, np.arange(len(events)), auroc, events, "AUROC", "{:.3f}")
    ax1.set_title("Node-level AUROC (weak labels)")

    ax2 = fig.add_subplot(1,2,2)
    bar_with_labels(ax2, np.arange(len(events)), auprc, events, "AUPRC", "{:.3f}")
    ax2.set_title("Node-level AUPRC (weak labels)")

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {args.out}")

if __name__ == "__main__":
    main()

# python plot_f3_rocpr.py \
#   --inputs reports/20110427/metrics_all_runs.csv \
#            reports/20110823/metrics_all_runs.csv \
#            reports/20110908/metrics_all_runs.csv \
#   --labels "2011-04-27" "2011-08-23" "2011-09-08" \
#   --out figs_F3/F3_AUROC_AUPRC.png