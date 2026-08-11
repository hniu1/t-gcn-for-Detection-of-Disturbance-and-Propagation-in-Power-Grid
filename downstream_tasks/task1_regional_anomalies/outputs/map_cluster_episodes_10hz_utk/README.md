# Regional Anomaly Results for Review

This folder contains candidate regional anomalies identified from discrepancies between measured frequency-deviation signals and T-GCN predictions. We are sharing these results with our UTK collaborators for independent review and interpretation.

The candidates should not be treated as confirmed power-system events. The detailed 10 Hz plots show several different types of behavior. Some high-ranked episodes are dominated by unusually large fluctuations from one sensor, while other episodes show fluctuations across every sensor in the selected cluster. These patterns may reflect sensor or data-quality issues, physical grid events, or a mixture of both. Additional operational context and domain review are needed to distinguish these cases.

## Initial observations

Our preliminary visual review found that several of the highest-ranked episodes appear to be driven primarily by one strongly fluctuating sensor:

- **Episode 0:** Sensor **853** has much larger and more irregular fluctuations than sensors 732 and 1623.
- **Episode 1:** Sensor **757** appears to dominate the abnormal behavior relative to sensors 730 and 1484.
- **Episode 2:** Sensor **757** again shows strong fluctuations relative to sensors 1382 and 1144.

Similar single-sensor-dominant behavior occurs in some other episodes. These cases are important, but they may be more indicative of a sensor-reading or data-quality problem than a regional physical event.

Other episodes have a different structure. In **episodes 24, 29, and 32**, all three sensors in each selected cluster show fluctuations. Coordinated multi-sensor behavior may be more consistent with a shared event, although simultaneity, spatial consistency, and external event records still need to be checked before drawing that conclusion.

These examples are intended to guide review, not to assign final labels. We ask UTK reviewers to consider both the magnitude of the fluctuations and whether the timing and shape are coherent across sensors.

## Folder contents

### `map_error_freq_dev_timeplay_interactive.html`

This is an interactive, time-animated map of the model's frequency-deviation prediction error across the sensor network.

- Each marker represents a sensor at its geographic location.
- Marker color represents the absolute prediction error, `|predicted - actual|`, for `freq_dev`.
- The map value is based on the mean across each 10-value forecast horizon. It is therefore a 1 Hz overview used for locating and grouping candidate anomalies; the episode plots described below retain the individual 10 Hz values.
- The color scale runs from approximately 0 to 0.05. Light yellow indicates small error, orange indicates larger error, and red/dark red indicates the largest error. Values at or above the upper end of the scale use the darkest color.
- Black-outlined, larger markers denote sensors in the rightmost 20% of the reordered heatmap axis. This outline is a visualization aid and does not by itself mean that a sensor is anomalous.
- Light connecting lines show the geographic-adjacency graph used by the analysis. They do not indicate power-flow direction or prove that an event propagated between two sensors.
- Click a marker to see its sensor ID, timestamp, model-sample index, absolute error, original sensor index, and reordered index.
- Use the controls at the bottom-left to play, pause, loop, change playback speed, or move to a specific time.
- The map frames are subsampled (`stride=100`) to keep the interactive file manageable, so use the episode plots for detailed temporal inspection.

To view the map, download the complete folder and open the HTML file in a modern web browser. An internet connection may be required to load the Leaflet basemap and JavaScript libraries referenced by the file.

### `episode_plots/`

This directory contains 35 detailed PNG plots, one for each selected cluster episode. The plots show the original 10 Hz values rather than the 1 Hz horizon mean used during candidate selection.

Files follow this naming convention:

```text
episode_<episode_id>_t<peak_time_idx>.png
```

For example, `episode_0_t1020.png` corresponds to episode 0 and has its peak 1 Hz model-sample index at 1020. The same `episode_id` can be used to find the episode metadata in the CSV file.

Each figure contains three panels:

1. **Cluster graph:** Shows the sensors in the selected cluster and their geographic-adjacency connections. Node color represents the absolute frequency-deviation error at the peak 10 Hz instant within the episode's peak model sample. Warmer/darker nodes have larger errors relative to the other sensors in that figure.
2. **Actual and predicted signals:** Solid lines are the measured `freq_dev` values and dashed lines are the T-GCN predictions. Each sensor has a consistent color across the figure.
3. **Absolute error:** Shows `|predicted - actual|` for each sensor.

The yellow shaded region is the detected episode window. The vertical dotted line marks the highest-error 10 Hz instant inside the episode's original peak 1 Hz model sample. The title lists the episode ID, cluster size, original 1 Hz peak-mean error, time window, sensor IDs, and sensor names.

Some measured traces contain flat segments because values were missing in the source data and were filled during preprocessing. Short gaps were linearly interpolated, medium gaps were forward- or backward-filled, and remaining missing regions used a training-data median fallback. Forward filling and median fallback can produce visibly constant sections. These flat sections should not automatically be interpreted as periods of genuinely stable sensor behavior. They should instead be treated as possible missing-data indicators and checked against the original sensor-availability or data-quality records.

### `map_freq_hot_cluster_episodes_selected.csv`

This table contains the metadata for the 35 selected episodes and connects the interactive-map detections to the detailed PNG plots.

Important columns include:

- `episode_id`: Episode identifier used in each PNG filename.
- `sensor_set`: Canonical, sorted set of sensors defining the cluster.
- `cluster_size`: Number of sensors in the cluster.
- `start_time_idx`, `end_time_idx`: Beginning and end model-sample indices of the collapsed episode.
- `peak_time_idx`: Model-sample index with the largest episode-level error; also appears after `_t` in the PNG filename.
- `n_frames`: Number of detected map frames combined into the episode.
- `duration_steps`: Inclusive duration of the episode in model-sample steps.
- `peak_mean_abs_err`: Largest cluster-mean absolute error at the 1 Hz horizon-mean resolution used for detection and ranking.
- `peak_max_abs_err`: Largest individual-sensor absolute error at that resolution.
- `sensor_ids`: Sensor IDs in plotting order.
- `sensor_names`: Sensor names corresponding to `sensor_ids`.
- `sensor_indices`: Internal model-array indices corresponding to the sensors.
- `peak_timestamp`: Reconstructed timestamp of the 1 Hz peak sample.
- `rank_score`: Abnormality score used to prioritize the episodes. The exact calculation is explained below. It is a ranking score, not an event probability.

## Abnormality score and episode ranking

The rows and episode IDs in the selected CSV are ordered from the highest to the lowest `rank_score`. The score is calculated as:

```text
rank_score = peak_mean_abs_err × [1 + 0.15 × max(cluster_size - 2, 0)]
```

Here:

- `peak_mean_abs_err` is the largest mean absolute prediction error across the sensors in the cluster, evaluated at the 1 Hz horizon-mean resolution used for detection.
- `cluster_size` is the number of sensors participating in the cluster.
- The multiplicative term gives a modest bonus to clusters involving more sensors. For example, the multiplier is 1.15 for a three-sensor cluster, 1.30 for four sensors, and 1.45 for five sensors.

The score is driven mainly by error magnitude, while the size bonus prevents a spatially broader anomaly from being ranked too low solely because its mean error is slightly smaller. `peak_max_abs_err` is reported for reference but is not directly included in `rank_score`. This distinction matters because an episode can receive a high score when one sensor fluctuates strongly enough to raise the cluster mean; a high rank therefore does not necessarily imply coherent behavior across every sensor.

For illustration, the first three episodes all contain three sensors:

| Episode | Peak mean absolute error | Peak maximum absolute error | Rank score | Preliminary pattern |
|---:|---:|---:|---:|---|
| 0 | 0.094685 | 0.199377 | 0.108888 | Dominated by sensor 853 |
| 1 | 0.091692 | 0.222711 | 0.105446 | Dominated by sensor 757 |
| 2 | 0.088816 | 0.224983 | 0.102138 | Dominated by sensor 757 |

Before assigning the final episode IDs, the selection procedure also suppresses nearby duplicate candidates: a lower-ranked candidate is skipped when its peak is within 180 model-sample steps of an already selected episode and the two candidates share at least one sensor. The resulting 35 episodes remain in descending abnormality-score order, but they are a diversity-filtered subset rather than every detected candidate.

## Suggested review procedure

1. Open the interactive map to identify when and where groups of elevated errors appear.
2. Use `peak_timestamp` and `sensor_ids` in the CSV to locate the corresponding map activity.
3. Open the matching episode PNG and compare the measured and predicted 10 Hz traces before, during, and after the yellow episode window.
4. Determine whether the abnormality score is driven mainly by one sensor or supported by coherent behavior across the full cluster.
5. Compare the onset, direction, waveform, and recovery timing across sensors; shared fluctuation alone is not sufficient if the temporal patterns are unrelated.
6. Where available, compare candidate times against known disturbances, switching operations, outages, frequency events, and sensor-quality records.

## Interpretation Notes

Patterns that may be more consistent with a **sensor or data-quality issue** include:

- A large fluctuation, clipping, discontinuity, or erratic high-frequency behavior in one sensor while nearby or connected sensors remain stable.
- Abrupt offsets or repeated spikes that are not shared by other sensors.
- A measured trace that changes implausibly while the prediction and peer sensors remain smooth.
- A high cluster score that is visibly dominated by one sensor, as seen preliminarily for sensor 853 in episode 0 and sensor 757 in episodes 1 and 2.
- A perfectly flat measured segment, particularly when it begins or ends abruptly, because missing-value filling can create this pattern.

Patterns that may be more consistent with a **candidate physical event** include:

- Similar changes occurring at nearly the same time across multiple sensors.
- A coherent onset and recovery visible in geographically separated or graph-connected sensors.
- Prediction errors that rise together because the measured signals share a structured deviation from their expected behavior.
- Fluctuations across all members of the cluster, as seen preliminarily in episodes 24, 29, and 32, provided that the timing and waveform are also physically consistent.

These are review heuristics only. A large prediction error means that the observation differed from the model expectation; it does not, by itself, establish the cause. We would especially appreciate UTK feedback on which episodes appear to be credible grid events, which appear sensor-related, and which remain ambiguous.
