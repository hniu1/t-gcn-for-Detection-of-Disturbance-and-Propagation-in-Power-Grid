# Task 3 — Synthetic Fault Injection (validation)

Creates ground truth by injecting known faults and events into clean data, so the
Task 2 discriminator can be scored with a confusion matrix instead of inspected by
eye. Method and rationale are in [`../METHOD.md`](../METHOD.md).

## Scripts

| Script | Role |
|--------|------|
| `inject_faults.py` | corrupt the recorded `true` array → injected results dir + `injected_truth.csv` |
| `score.py` | match discriminator labels against `injected_truth.csv` → detection recall + confusion matrix |

## Usage

```bash
# 1. Inject known faults/events into a clean forecaster output directory
python downstream_tasks/task3_fault_injection/inject_faults.py --results_dir <clean_results_dir>

# 2. Run the discriminator on the injected directory
python downstream_tasks/task2_failure_vs_event/discriminate.py --results_dir <injected_results_dir>

# 3. Score the labels against the injected truth
python downstream_tasks/task3_fault_injection/score.py
```

## Fault model

- **failure** — single sensor (stuck or spiky).
- **event** — graph-connected group (coherent deviation).
- Target-only: corrupts the recorded `true` array, not the model input. See
  `METHOD.md` (5) for the target-only vs input-level distinction.

## Inputs and outputs

- `inject_faults.py`: reads `<clean_results_dir>`; writes `<injected_results_dir>`
  and `outputs/injected_truth.csv` (`sensor`, `time`, `type`).
- `score.py`: reads `outputs/injected_truth.csv` and the discriminator's
  `candidate_labels.csv`; writes `outputs/score.json` (detection recall and the
  discrimination confusion matrix).
