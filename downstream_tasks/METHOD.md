# Failure-vs-Event Discrimination: Method

Method for classifying a flagged anomaly as a **sensor failure** (faulty
instrument) or a **physical event** (real grid disturbance), implemented as a
downstream stage on the multi-feature T-GCN forecaster output.

## 1) Problem statement

The forecaster flags a sensor when its prediction error is large. A large error
has three possible causes: a real grid event, a sensor fault, or a model that is
weak on that sensor. Error magnitude alone does not separate them. The recorded
signal is also not verified truth — a faulty sensor's reading is still the
recorded value — so no label exists to train or score against directly.

This stage adds two components:

- a **discriminator** that assigns each flagged anomaly a label
  (`sensor_failure` / `event` / `ambiguous`), and
- a **synthetic fault-injection harness** that creates labels so the
  discriminator can be scored.

## 2) Design principle: features on the raw signal

All discrimination features are computed on the **raw recorded signal**, not on
the forecast residual and not on any detector's cluster grouping. Three reasons:

1. **Independence.** Measuring neighbor response on the neighbors' own recorded
   traces is independent of both the forecaster and the adjacency-based
   clustering. Reusing a detector's grouping to test whether neighbors react only
   restates the grouping rule.
2. **Robustness to spatial coupling.** The forecaster's graph convolution
   propagates a sensor's input to its neighbors, which can inflate a neighbor's
   *error* without its recorded trace changing. Raw-signal features are unaffected
   by this coupling.
3. **Separation of roles.** The forecaster locates *where* to look; the raw-signal
   features decide *what kind* of anomaly it is.

The discriminator operates on the raw activation pool — sensors whose residual
exceeds a quantile — not on cluster-detector output, which requires at least two
linked sensors and cannot surface an isolated failure.

## 3) Discrimination features

Per flagged sensor-time, three features are computed on the raw signal
(`feature_idx = 0`, freq_dev; adjacency from `results/A_geo.npy`).

### 3.1 Local movement (temporal)

Windowed standard deviation of the recorded signal. The sensor is **flat** if its
windowed std is below `flat_factor × median(std)`. A large error with a flat
recording indicates the instrument: a real event would have moved the
measurement.

### 3.2 Neighbor coherence (spatial)

A neighbor **co-deviates** if its recorded deviation exceeds
`dev_factor × median(deviation)`. The sensor is **coherent** if at least
`coherence_min` of its graph neighbors co-deviate. A real event moves a region
together; an isolated failure leaves the neighbors quiet.

### 3.3 Shape (statistical)

Moving-median residual of the recorded signal. The sensor is **spiky** if that
residual exceeds `shape_factor × median|deviation|`. Grid dynamics are
band-limited and smooth, so an impulsive single-sample spike survives a moving
median and indicates a noisy instrument. The threshold is anchored to the signal
scale `median|deviation|` rather than to the spike residual, which is degenerate
on smooth signals.

### 3.4 Decision rule

| recording | neighbors | shape | label |
|-----------|-----------|-------|-------|
| flat  | isolated | —      | `sensor_failure` |
| moved | coherent | —      | `event` |
| moved | isolated | spiky  | `sensor_failure` |
| moved | isolated | smooth | `ambiguous` |

Movement and coherence (3.1, 3.2) resolve stuck failures and coherent events; the
shape test (3.3) resolves the isolated-but-moving case, separating a noisy sensor
from a genuine local event. The `ambiguous` label is retained as an explicit
"needs an additional feature" bucket rather than a forced classification.

### 3.5 Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `cand_quantile` | 0.90 | residual quantile defining the activation pool |
| `window_radius` | 3 | half-window for the movement (std) test |
| `flat_factor` | 0.25 | flat if windowed std < factor × median std |
| `dev_factor` | 1.5 | neighbor co-deviates if dev > factor × median dev |
| `coherence_min` | 0.3 | coherent if this fraction of neighbors co-deviate |
| `shape_factor` | 2.0 | spiky if median-filter residual > factor × median\|dev\| |
| `median_window` | 5 | moving-median window for the shape test |

## 4) Extended features (planned)

The three features above address stuck, dropout, and spiky faults. Additional
features are planned, each targeting a fault class the current set does not
resolve:

- **Cross-channel consistency.** A real event moves frequency, RoCoF, angle, and
  voltage together in physically consistent directions; a failing channel moves
  one in isolation. Requires angle and voltage in the saved outputs.
- **Physical plausibility.** RoCoF is bounded by system inertia; an out-of-range
  one-sample jump is a data artifact, not a grid event.
- **Propagation.** A real disturbance travels with finite speed and a consistent
  direction across electrical distance; a shared-infrastructure glitch appears at
  zero lag. Supports a directed lead-lag graph.
- **Self-report.** The PMU status flag and the coverage mask already mark
  dropout-type failures.

**Common-mode caveat.** Sensors sharing a clock, concentrator, or vendor can fail
together and produce event-like spatial coherence. Separating these requires an
infrastructure-adjacency graph cross-referenced with the geographic graph, plus a
zero-lag test.

## 5) Validation by synthetic fault injection

With no real labels available, faults are injected into clean data so that
detection and discrimination can be scored.

**Target-only injection.** The forecaster predicted normal behavior, so the
prediction array is left unchanged and the recorded `true` array is corrupted. The
residual then rises exactly where injected, and the `(sensor, time, type)` is
known. A **failure** corrupts a single sensor (stuck or spiky); an **event**
corrupts a graph-connected group (coherent deviation). The harness injects, runs
the discriminator on the raw activation pool, and scores detection recall and a
discrimination confusion matrix against the injected truth.

**Input-level injection (planned).** Feeding the fault into the model input and
re-running inference is required to measure the forecaster's spatial-coupling
effect; target-only injection does not test it.

## 6) Scope and status

The discriminator and harness are **model-agnostic**: they read any forecaster
output directory. Current validation runs on a reduced-scale (smoke) model and
confirms that the pipeline runs end to end and the features behave as designed.
The injected faults are separable by construction, so a clean confusion matrix
demonstrates the mechanism rather than a final accuracy. Reported failure/event
rates require a full-scale model, threshold calibration into precision/recall
curves, and the harder fault classes (drift, common-mode, cross-channel).
