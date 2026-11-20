# Final Recommendation and Limitations

Status as of 20 November 2026. Research prototype, not a clinical system.

## 1. Recommended configuration for the demo

- Physiology: V3 (BVP + EDA Random Forest) as default. V5 (EDA-only SVM) offered as an alternative, flagged as below.
- Video: upper-face Random Forest, shown separately.
- Fusion: **not validated**. Show physiology and video side by side. Do not display a fused score.
- All outputs are labelled "score" or "estimate". None is a calibrated probability.

## 2. Results that can be reported (held-out, participant-wise LOSO)

| Model | Participants | Balanced acc. (95% CI) | Specificity | Sensitivity |
|---|---|---|---|---|
| V3 BVP+EDA RF | 56 | 0.898 (0.858-0.935) | 0.854 | 0.942 |
| V5 EDA-only RBF-SVM | 56 | 0.905 (0.865-0.942) | 0.933 | 0.877 |
| Upper-face video RF (200 trees, depth 8, balanced) | 18 | 0.904 (0.864-0.941) | 0.877 | 0.932 |

- V5 minus V3 balanced accuracy: +0.007 (95% CI -0.041 to 0.058). Not distinguishable.
- V5 specificity advantage: CI 0.008 to 0.156 (modest, fragile; 11 rest windows per participant).
- Video per-participant specificity ranges from 0.33 to 1.0.
- CIs are participant-level bootstrap. Video and physiology use different windows (10 s vs 30 s) and cohorts; do not compare them directly.

## 3. Results that must NOT be reported as generalization

- Rest control (V3 0/728, V5 170/728 false-distress windows): both models were refit on all 56 participants. In-sample pipeline check only.
- V3 and V5 outputs on UBFC-Phys participants through the live API: in-sample.
- Deployed video model on s3-s20: in-sample (trained on all 972 windows).
- Early video figures (0.97 / 0.90 accuracy, 100% sensitivity on T2/T3 only, AUC 1.000).
- Partial TCN/LSTM runs (45/56 participants).
- Synthetic-data API smoke test (5/5) verifies lifecycle only, not predictions.

## 4. Provenance

- Video model: Random Forest (200 trees, max_depth 8, balanced class weights, seed 42) + StandardScaler on 19 features, trained on 972 windows from 18 participants. Reproduced exactly (identical predictions) by `ubfc_pipeline/train_video_upperface.py`.
- V3/V5: refit on all 56 participants after T1 normalization (manifests). LOSO metrics are separate.
- V5 LOSO protocol matches the deployed recipe (19 features, C=1, gamma=scale, balanced, T1 median/IQR on first 85 s, rest scored from 100 s). EDA feature parity test passed (168 recordings, 5,208 windows).

## 5. Limitations

1. **Label confound.** T1 = 0 and T2/T3 = 1 for all models. Results show task-vs-rest discrimination, not measured stress. Video features (head turn, nod, tilt, motion) plausibly track speaking and task behaviour.
2. **V5 live gap (unresolved).** Offline false-positive rate on rest is about 6.7%; the live pipeline gives roughly 17-23% in-sample. Feature extraction, normalization rule and hyperparameters were checked and match. Cause unknown. Defensive warnings did not flag several worst cases (e.g. s4, s18, s37, s48).
3. **Scores are uncalibrated.** V3 gives RF scores; V5 gives raw SVM decision scores; video gives RF scores. None may be blended or read as probabilities.
4. **Fusion not validated.** The stacker is LOSO over existing OOF predictions, not nested; V5 paired predictions are in-sample; `test_fusion_integration.py` has an unresolved constructor mismatch.
5. **Video specifics.** 18 participants; strong per-person variation (s9 specificity 0.33). T1 baseline subtraction was inconclusive (change +0.015, CI -0.025 to 0.059). A single upload has no reliable rest segment. Live adapter verified only on one participant (s1), one window per clip; CSV parity not verified.
6. **Quality checks** for physiology are recording-level, not window-level. Expected sampling rates BVP 64 Hz, EDA 4 Hz; no resampling. Sessions are in-memory.
7. **Domain shift.** UBFC-Phys is a controlled dataset. Webcam or other-device performance is unknown.
8. **Terminology.** Use "task-vs-rest discrimination" or "change from baseline", not "stress detection" or "distress diagnosis".

## 6. Readiness

- Labelled research demo: acceptable.
- Research paper: needs nested stacking (or its removal), the V5 gap explained, and the confound stated prominently.
- Live/clinical deployment: not ready.
