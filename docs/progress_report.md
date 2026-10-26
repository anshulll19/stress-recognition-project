# SEC Project — Progress Report

## Project overview

This project explores stress/distress-related state recognition through physiological signals, video-derived facial/behavioral features, and planned multimodal fusion.

**Interpretation caveat:** The current labels are task-derived. Model outputs should be described as *rest-task proxy* and *stress-task proxy*, not as clinical diagnoses or clinically validated distress measurements.

## 1. UBFC-Phys Stage 1 physiology pipeline

Reported Colab artifacts:

| File | Reported shape / contents |
|---|---|
| `physiology_features_all56.csv` | `(10416, 19)`; 5,208 BVP rows and 5,208 EDA rows |
| `physiology_features_fused.csv` | `(5208, 16)` |
| `physiology_features_labeled.csv` | `(5208, 17)` |
| `raw_signal_windows.npz` | Raw signal windows |
| `raw_signal_window_metadata.csv` | Raw-window metadata |

Reported raw arrays: BVP `(5880, 640, 1)`, EDA `(5880, 40, 1)`. There are 56 participants and 1,960 windows per task (tasks 1, 2, and 3). Reported labels: 1,960 class-0 windows and 3,920 class-1 windows. Dataset validation was reported as passed.

Reported labeled feature distribution: task 1 / label 0 = 1,736 rows; task 2 / label 1 = 1,736 rows; task 3 / label 1 = 1,736 rows.

### Initial feature-based results

| Model | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| BVP only | 0.74 | — | — |
| EDA only | 0.65 | — | — |
| Original BVP + EDA, participant-wise LOSO | 0.7696 | 0.7876 | 0.7591 |

### Rest-calibration experiment

| Metric | Original BVP + EDA | Rest-calibrated BVP + EDA |
|---|---:|---:|
| Accuracy | 0.7696 | 0.8877 |
| Balanced accuracy | 0.7876 | 0.9107 |
| Macro F1 | 0.7591 | 0.8811 |

Reported task-wise accuracy for the rest-calibrated experiment: task 2 = `0.8272`, task 3 = `0.8560`. Recompute these numbers from saved predictions before treating them as final publication results.

## 2. TCN versus LSTM

The recorded LOSO run was partial, covering 45 of 56 participants; participants 46–56 were missing at the time of the report.

| Model | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| TCN | 0.7780 | 0.6943 | 0.7106 |
| LSTM | 0.8110 | 0.7687 | 0.7785 |

Reported artifacts:
- `tcn_lstm_oof_predictions_smoke.csv`
- `tcn_lstm_results_smoke.csv`
- `tcn_lstm_oof_predictions_full.csv` (the recorded run was still partial)
- Columns: `model`, `window_index`, `participant`, `true`, `probability`, `prediction`

Each model had 4,725 rows (45 participants), 9,450 rows total; the last saved participant was 45.

**Next:** finish participants 46–56, verify all participants are represented, and recompute metrics from the complete out-of-fold predictions.

## 3. Video feature dataset and offline baseline

Reported artifacts:
- `video_frames_compact.csv`
- `video_windows.csv`
- `video_feature_names.json`
- `video_model_metadata.json`
- `video_distress_baseline.joblib`

Reported dataset: `video_windows.csv` shape `(972, 36)`, 18 participants, 54 windows per participant, labels 0 = 324 windows and 1 = 648 windows. All 972 windows were reported as reliable.

The compact frame features include motion, eye aspect ratio, mouth opening/width, brow-eye distance, and head-turn/nod/tilt/depth proxies.

### Offline video baseline

Reported Random Forest LOSO results:

| Metric | Reported result |
|---|---:|
| Accuracy | 0.9660 |
| Balanced accuracy | 0.9668 |
| Macro F1 | 0.9622 |

The top five features were reported to account for 73.1% of feature importance. Mouth-opening statistics and nod-related differences were among the dominant features.

**Research caution:** this high score may reflect task-related behavior or speech differences. Audit participant/session leakage and task-label leakage, and clearly document the evaluation protocol and limitations.

## 4. Live webcam demo

A preliminary Gradio app was added to the Colab notebook using:
- `video_distress_baseline.joblib`
- `video_feature_names.json`
- MediaPipe Tasks Face Landmarker at `/content/face_landmarker.task`
- OpenCV for landmark visualization
- Gradio streaming webcam input
- A rolling 10-second feature window and `predict_proba` output

The model expects these 23 features in this exact order:

1. `f_motion_mean`
2. `f_motion_std`
3. `f_ear_mean_mean`
4. `f_ear_mean_std`
5. `f_mouth_open_mean`
6. `f_mouth_open_std`
7. `f_mouth_width_mean`
8. `f_mouth_width_std`
9. `f_brow_eye_mean`
10. `f_brow_eye_std`
11. `f_turn_proxy_mean`
12. `f_turn_proxy_std`
13. `f_nod_proxy_mean`
14. `f_nod_proxy_std`
15. `f_tilt_proxy_mean`
16. `f_tilt_proxy_std`
17. `f_turn_depth_proxy_mean`
18. `f_turn_depth_proxy_std`
19. `f_turn_proxy_absdiff`
20. `f_nod_proxy_absdiff`
21. `f_tilt_proxy_absdiff`
22. `f_motion_max`
23. `f_motion_p90`

### Current status and known issues

- The Gradio page opens and shows webcam video.
- MediaPipe facial landmarks are visible.
- The model and 23-feature input were previously loaded successfully.
- A screenshot showed a buffer of about 4.8/10 seconds but only 2 valid frames.
- Stable live predictions have **not** yet been confirmed.

The live motion feature currently uses normalized landmark displacement, while the offline training pipeline used a precomputed `landmark_motion_mean`. These calculations are not yet proven equivalent.

### Live-demo next steps

1. Diagnose callback frequency and face-detection success rate.
2. Confirm sufficient valid frames accumulate for a full 10-second window.
3. Compare live extraction against the offline feature-generation code.
4. Test the same recorded clips through both pipelines and compare feature vectors.
5. Test lighting, camera distance, and movement conditions.
6. Avoid clinical claims and clearly state limitations.

## 5. Environment notes

Reported versions/workspaces:
- Google Colab for UBFC-Phys experiments
- MacBook Air M2, 16 GB RAM for local development
- Python 3.12.13 in the stress-recognition workspace
- TensorFlow 2.21.0 and Keras 3.15.1; CPU-only TensorFlow reported
- MediaPipe 0.10.32
- OpenCV 4.14.0
- Gradio 6.30.0

Versions may differ by runtime. Record the exact environment used for each final experiment.

## 6. Suggested repository organization

```text
stress-recognition-project/
├── README.md
├── docs/
│   └── progress_report.md
├── notebooks/
│   └── UBFC_Phys_Stage1_Pipeline.ipynb
├── src/
├── models/
├── results/
│   ├── physiology/
│   ├── video/
│   └── temporal_models/
└── requirements.txt
```

This is a suggested organization, not a claim that all listed files already exist. Do not push raw participant data or large datasets without checking dataset licensing, consent, and repository access permissions.

## 7. Prioritized next steps

- [ ] Diagnose the low valid-frame count in the live webcam app.
- [ ] Validate live/offline feature parity.
- [ ] Complete TCN and LSTM LOSO runs for all 56 participants.
- [ ] Recompute final metrics and classification reports.
- [ ] Audit participant-wise splits and task-label leakage.
- [ ] Record environment versions and reproducible commands.
- [ ] Document limitations and intended use.
- [ ] Keep source code, documentation, model artifacts, and selected results versioned; exclude raw/private data.

## Reproducibility note

Metrics above are summarized from project progress notes. Verify every number against saved prediction files and scripts before publication. Distinguish completed experiments from partial runs, and offline model results from the unvalidated live demo.
