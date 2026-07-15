# Task 2 — Failure-vs-Event Discrimination

Labels each flagged anomaly as a **sensor failure** or a **physical event** using
features computed on the raw recorded signal. The discriminator is model-agnostic:
it reads any forecaster output directory. Method and rationale are in
[`../METHOD.md`](../METHOD.md).

## Script

| Script | Role |
|--------|------|
| `discriminate.py` | label the raw activation pool by local movement, neighbor coherence, and shape |

## Usage

```bash
python downstream_tasks/task2_failure_vs_event/discriminate.py --results_dir <results_dir>
```

`<results_dir>` is any forecaster output directory (e.g. a training run output).
Use `--disable_shape` to turn off the shape test (3.3 in `METHOD.md`).

## Inputs

- `<results_dir>/{split}_predictions_original.npy`, `{split}_targets_original.npy` — forecaster output (falls back to scaled arrays if originals are absent)
- `results/A_geo.npy` — sensor adjacency

## Output

- `outputs/candidate_labels.csv` — one row per activation: sensor, time index,
  label (`sensor_failure` / `event` / `ambiguous`), and the underlying feature
  values.

## Parameters

`--cand_quantile`, `--window_radius`, `--flat_factor`, `--dev_factor`,
`--coherence_min`, `--shape_factor`, `--median_window`. Defaults and meanings are
tabulated in `METHOD.md` (3.5). They are threshold-sensitive and should be
recalibrated on a full-scale model.
