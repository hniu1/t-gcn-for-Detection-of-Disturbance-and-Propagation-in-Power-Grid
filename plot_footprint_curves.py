#!/usr/bin/env python3
import os, argparse, pandas as pd, matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report_dirs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--out", default="figs_F2/F2c_footprint_curves.png")
    args = ap.parse_args()

    if not args.labels or len(args.labels) != len(args.report_dirs):
        args.labels = [os.path.basename(os.path.normpath(d)) for d in args.report_dirs]

    plt.figure(figsize=(7,3))
    for d, lab in zip(args.report_dirs, args.labels):
        fcsv = os.path.join(d, "footprint_vs_delta.csv")
        df = pd.read_csv(fcsv)
        plt.plot(df["delta_sec"], df["footprint_pct"], label=lab)

    plt.xlabel("Δt after detection (s)")
    plt.ylabel("Active region (% of sites)")
    plt.title("Footprint growth after detection")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"[saved] {args.out}")

if __name__ == "__main__":
    main()

'''
python code/t-gcn/plot_footprint_curves.py \
  --report_dirs reports/20110427 reports/20110823 reports/20110908 \
  --labels "2011-04-27" "2011-08-23" "2011-09-08" \
  --out reports/figs_F2/F2c_footprint_curves.png
'''