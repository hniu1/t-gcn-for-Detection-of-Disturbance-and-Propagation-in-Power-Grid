# Self-Supervised T-GCN for Disturbance Detection & Propagation in Power Grids

A lightweight, label-free pipeline that forecasts frequency residuals with a Temporal Graph Convolutional Network (GCN→GRU) and turns forecast errors into node/region anomalies and propagation summaries.

![python](https://img.shields.io/badge/python-3.9%2B-blue)
![pytorch](https://img.shields.io/badge/PyTorch-2.x-red)
![license](https://img.shields.io/badge/license-MIT-green)

---

## What this repo contains

- **Data preprocessing** for FNET/FDR plain-text logs → tidy per-site DataFrames with absolute/relative time, residuals, and RoCoF.
- **Signal-built graph** (no geocoding): similarities from pre-event correlations with a lag penalty; k-NN sparsification.
- **T-GCN forecaster** trained **only on pre-event** windows (self-supervised). Node anomaly = one-step forecast error z-score.
- **Region alarm** from connected components of simultaneous node anomalies; per-node **arrival times**; **propagation** speed & consistency.
- **Reporting & plotting** scripts to reproduce all paper figures (F2–F7) and metrics.
- **Hyperparameter search** (Optuna) with a robust objective.

---

## Repo layout
```text
T-GCN/
├── gcn_pipeline.py # graph building, feature stacking, T-GCN model & training
├── preprocess_fnet.py # FNET/FDR decoding, absolute time, residuals, RoCoF
├── main.py # one-day training & scoring → save npy/csv artifacts
├── tgcn_report.py # per-day report: plots + metrics, footprint curves
├── plot_f2_metrics.py # F2: TTD / FA / footprint bars (cross-event)
├── plot_f3_rocpr.py # F3: AUROC/AUPRC bars (weak labels)
├── plot_f5_sensitivity.py # F5: sensitivity to tau and M_min (per-day)
├── plot_f7_propagation.py # F7: propagation speed & fit quality
├── plot_footprint_curves.py # footprint growth after detection
├── optuna_tgcn_search.py # optional: hyperparameter tuning
└── README.md
```


---

## Environment

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -U pip wheel
# Choose the right PyTorch index URL per your CUDA/CPU setup:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas scipy matplotlib scikit-learn optuna
```

## Data

The code expects a one-day folder of FNET/FDR .txt files, named like YYYYMMDD-HHMMSS-UsFlPlantcity623.txt. Each row resembles:

```php
<time_index> <...> <Frequency \t Delay>
```

preprocess_fnet.py:

* decodes absolute DateTime and relative seconds t,
* extracts Frequency (Hz) and Delay,
* builds residual r = f − 60 (Hz) and RoCoF Δf (Hz/s),
* estimates the sampling interval dt.
* Place your day folder (e.g., data/20110908/) before running.

## Quick start: train & score one day

```bash
python main.py \
  --data_dir data/20110908 \
  --out_dir results/20110908 \
  --pre_window 120 --guard 10 \
  --k 6 --lag_lambda 2.0 \
  --lookback 60 --horizon 1 \
  --epochs 15 --batch_size 64 --lr 1e-3 --weight_decay 1e-4 \
  --tau 2.5 --persist_s 2.0
```

Artifacts written to --out_dir:

- W.npy, A_hat.npy – raw & normalized adjacency

- tvec.npy, X.npy – aligned time vector and node features [T, N, 2] (r, RoCoF)

- pred_r.npy – one-step residual predictions [T, N]

- err_r.npy – absolute forecast error [T, N]

- z.npy – node anomaly z-scores [T, N]

- arrival_times.csv – per-site arrival time (seconds + timestamp)

## Per-day report (F1 + metrics)

```bash
python tgcn_report.py \
  --in_dirs results/20110908 \
  --out_dir reports \
  --tau 2.5 --persist_s 2.0 --M_min 5 --event_window_s 30
```

Outputs (under reports/20110908/):

- z_heatmap.png — node z-scores (sites ordered by arrival)

- largest_component.png — largest active component over time (region alarm curve)

- propagation_embedding.png — spectral layout colored by arrival time

- footprint_vs_delta.* — footprint growth after detection (CSV + PNG)

- metrics.json / metrics.csv

## Reproduce paper figures

F2 — cross-event bars (TTD / FA / Footprint)
(use each event’s metrics_all_runs.csv)

```bash
python plot_f2_metrics.py \
  --inputs reports/20110427/metrics_all_runs.csv \
          reports/20110823/metrics_all_runs.csv \
          reports/20110908/metrics_all_runs.csv \
  --labels 2011-04-27 2011-08-23 2011-09-08 \
  --out_dir figs_F2
```

F3 — AUROC / AUPRC (weak labels)

```bash
python plot_f3_rocpr.py \
  --inputs reports/20110427/metrics_all_runs.csv \
          reports/20110823/metrics_all_runs.csv \
          reports/20110908/metrics_all_runs.csv \
  --labels 2011-04-27 2011-08-23 2011-09-08 \
  --out_dir figs_F3
```

F5 — Sensitivity to thresholds (per day)

```bash
python plot_f5_sensitivity.py \
  --run_dir results/20110908 \
  --out figs_F5/20110908_sensitivity.png \
  --tau_fix 2.5 --M_fix 5 --persist_s 2.0 --event_window_s 30

```

Footprint growth curves

```bash
python plot_f5_sensitivity.py \
  --run_dir results/20110908 \
  --out figs_F5/20110908_sensitivity.png \
  --tau_fix 2.5 --M_fix 5 --persist_s 2.0 --event_window_s 30

```

F7 — Propagation (speed & fit quality)

```bash
python plot_f7_propagation.py \
  --run_dirs results/20110427 results/20110823 results/20110908 \
  --out_dir figs_F7
```

Hyperparameter search (optional)
A simple Optuna search covers graph, model, and detection knobs; the objective blends TTD, pre-event FA/hr, and node-level separability with penalties for invalid metrics.

```bash
python optuna_tgcn_search.py \
  --data_dir data/20110908 \
  --n_trials 50 \
  --study_name tgcn_20110908 \
  --out_dir optuna_runs/20110908

```

Once you choose a best configuration, re-run main.py with those values to produce the final artifacts/figures.

### Notes & tips

- tau (z-threshold) controls node sensitivity; M_min controls region size. Try tau ∈ [2.3, 2.9], M_min ∈ [5, 12].

- The global onset proxy used for weak labels can be changed in tgcn_report.py (median or an early percentile).

- If FA/hr looks high, prefer pre-event FA/hr (FA_per_hour_pre in metrics) or increase persist_s.

## License

MIT — see LICENSE.

## Citation

If you use this code, please cite:

ACM reference format


Haoran Niu, Yang Chen, Moumita Samanta, and Olufemi A. Omitaomu. 2025. Self-Supervised T-GCN for Detection of Disturbance and Propagation in Power Grid. In Proceedings of the ACM SIGSPATIAL UrbanAI Workshop (UrbanAI’25). ACM, New York, NY, USA, 8 pages.


### BibTex

```text
@inproceedings{niu2025tgcn,
  author    = {Haoran Niu and Yang Chen and Moumita Samanta and Olufemi A. Omitaomu},
  title     = {Self-Supervised T-GCN for Detection of Disturbance and Propagation in Power Grid},
  booktitle = {Proceedings of the ACM SIGSPATIAL UrbanAI Workshop (UrbanAI'25)},
  year      = {2025},
  address   = {New York, NY, USA},
  publisher = {ACM},
  doi       = {10.1145/XXXXXXX.XXXXXXX}
}
```

