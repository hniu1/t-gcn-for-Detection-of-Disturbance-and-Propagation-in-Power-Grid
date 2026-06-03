Current-version analysis scripts for the cov80 split/scaled run.

Defaults:
- `results_dir`: `results_masked_cov80_full_scaled`
- `split_name`: `pred`
- plots use original-unit arrays when available

Typical commands from the project root:

```bash
python test_analysis_cov80/analyze_test_results.py
python test_analysis_cov80/plot_full_window_all_sensors.py
python test_analysis_cov80/plot_per_sensor_two_features.py
python test_analysis_cov80/build_error_interactive_map.py --feature_idx 0
python test_analysis_cov80/build_error_interactive_map.py --feature_idx 1 \
  --out_html test_analysis_cov80/maps_pred/map_error_volt_dev_interactive.html \
  --out_html_animated test_analysis_cov80/maps_pred/map_error_volt_dev_timeplay_interactive.html
```