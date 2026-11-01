# UBFC-Phys s1/T1 Multimodal Pilot — Exploratory Analysis

## Scope

- Input: `ubfc_pipeline/out/features_full/features.csv`
- Table shape: 16 rows × 47 columns
- Subject IDs: [1]; tasks: ['T1']
- Window time range: 0–180 seconds
- Columns with no missing values: 47/47

This is an exploratory report from one participant and one resting task. Windows overlap, so rows are correlated and are not independent samples. There are no window-level distress labels; no stress classifier was trained or evaluated here.

## Heart-rate comparison

- Paired windows: 16
- Mean absolute difference: 10.04 bpm
- Median absolute difference: 9.25 bpm
- Windows within 5 bpm: 2/16 (12.5%)
- Pearson correlation across overlapping windows: 0.561

Video-derived rPPG heart rate is experimental in this pilot. The comparison does not establish that it is interchangeable with BVP.

## Constant features

`sid`, `task`, `p_scr_n`, `p_scr_amp`, `q_face_frac`

Features are constant only in this pilot; do not globally remove them based on one participant. Zero EDA SCR count/amplitude may reflect detector settings, signal properties, or the resting condition and requires validation.

## Interpretation limits

- Facial blendshape/action-like values are observable video features, not direct measurements of distress.
- Brightness and sharpness are visual-quality proxies, not distress markers.
- EDA peak counts are not validated until units, filtering, thresholds and candidate responses are checked.
- Correlations and plots are descriptive only; overlap and the one-participant design prevent generalization claims.
- Do not use these windows for supervised stress training without defensible labels and subject-level evaluation.

## Output files

- `numeric_feature_summary.csv`, `missing_values.csv`, `feature_constant_check.csv`
- `heart_rate_comparison.csv` (if HR columns exist)
- `exploratory_correlations.csv` (if enough varying features exist)
- `01_heart_rate_by_window.png`, `02_heart_rate_scatter.png`
- `03_feature_trajectories.png`, `04_feature_heatmap.png`, `05_exploratory_correlations.png`

All outputs are written to this new directory. Existing data, source files and model results are not modified.
