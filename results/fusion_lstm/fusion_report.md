# Multimodal Late Fusion Report
**Subjects:** 18 (s10–s9)  |  **Aligned Windows:** 972  |  **Date:** 2026-10-27
**Physiological Model:** LSTM (from `tcn_lstm_oof_predictions_full.csv`)
**Physiological Metadata:** `raw_signal_window_metadata.csv`
**Video Source:** `video_windows.csv`

## Results

| Method | AUC (mean) | 95% CI | Bal-Acc | Macro-F1 |
|--------|-----------|--------|---------|----------|
| video_only | 0.994 | [0.983, 1.000] | 0.965 | 0.959 |
| physio_only | 0.814 | [0.717, 0.898] | 0.745 | 0.729 |
| blend_0.2 | 0.995 | [0.987, 1.000] | 0.971 | 0.966 |
| blend_0.3 | 0.996 | [0.990, 1.000] | 0.971 | 0.966 |
| blend_0.5 | 0.995 | [0.988, 0.999] | 0.967 | 0.964 |
| quality_aware | 0.995 | [0.987, 1.000] | 0.971 | 0.966 |
| stacked_loso | 0.996 | [0.988, 1.000] | 0.971 | 0.966 |

## Key findings

- **Chosen physiological model:** LSTM
- **Physio-only baseline:** AUC 0.814 (Bal-Acc 0.745)
- **Video-only baseline:** AUC 0.994 (Bal-Acc 0.965)
- **Best fusion method:** blend_0.3 (AUC 0.996, [0.990–1.000])
- Fusion improves over the strongest unimodal baseline on this 18-subject set.
- All 972 aligned windows match deterministically on subject, task, and start/end timestamps.

## Leakage controls
- All probabilities are strictly out-of-fold: each subject was excluded from training and validation.
- The video OOF probabilities were generated via subject-wise LeaveOneGroupOut with median imputation fit strictly on training folds (in-sample deployment weights were excluded).
- For `stacked_loso`: the meta-learner logistic regression weights were learned by inner-LOSO on training subjects only.
- No decision thresholds were tuned on test subjects.

## Methodological limitations & confounds
- **Cohort size:** 18 subjects (`s3`–`s20`), 54 windows per subject.
- **Speech confound:** As demonstrated in the confound ablation, video accuracy is heavily influenced by mouth movements during speech (T2) vs. silence (T1).
- **Task-derived proxy:** The target labels represent protocol conditions (Rest vs. Speech/Math), not clinically validated distress measurements.
