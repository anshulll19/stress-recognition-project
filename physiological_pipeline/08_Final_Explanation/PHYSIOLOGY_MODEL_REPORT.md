# Physiological Stress Recognition: Technical Architecture & Inference Report

**Project:** SEC (5th Semester) Research Project — Multimodal Stress Recognition & Trend Prediction  
**Module:** Physiological Inference Subsystem (`src/physiology_inference.py`, `src/calibration.py`)  
**Report Date:** 2026-10-30  
**Verification Status:** VERIFIED & TESTED  

---

## 1. Executive Summary & Objective

The objective of this module is to enable **live physiological stress prediction on supported new inputs**, moving beyond static replay of previously recorded prediction files.

Prior to this work:
1. WESAD inference was verified but lacked a standardized, reusable prediction interface capable of handling variable array shapes, continuous raw streams, and quality failure modes.
2. UBFC-Phys predictions were restricted to precomputed out-of-fold CSV tables (`results/temporal_models/tcn_lstm_oof_predictions_full.csv`) because deep model checkpoints were not preserved.
3. Rest calibration was documented as a high-performing experiment ($91.1\%$ balanced accuracy) but existed only as notebook scripts without a deployable transformer class.

This subsystem provides:
- **`src/calibration.py` (`RestCalibrator`):** Reusable transformer for subject baseline standardization.
- **`src/physiology_inference.py` (`WESADInferenceEngine`, `UBFCInferenceEngine`, `predict_stress`):** Production-grade inference adapters with input validation, sensor quality diagnostics, and unified dispatch.
- **`models/ubfc_deployment/`:** Serialized deployment checkpoints and complete 56-subject Leave-One-Subject-Out (LOSO) benchmark validation records.

---

## 2. Dataset-Specific Preprocessing & Input Contracts

```
Physiological Input Flow & Contracts:
┌───────────────────────────────┬────────────────────────────────┬───────────────────────────────┐
│ Input Modality                │ Signal Dimensions / Features   │ Preprocessing Applied         │
├───────────────────────────────┼────────────────────────────────┼───────────────────────────────┤
│ WESAD Continuous BVP          │ 1D array @ 64 Hz (or arbitrary)│ Butterworth Bandpass (0.7-3.7)│
│ WESAD Windowed BVP            │ (3840, 1), (3840,), (1, 3840,1)│ Per-window Z-Score (μ=0, σ=1) │
│ UBFC Multimodal Features      │ 12-dim vector (6 BVP, 6 EDA)   │ Optional Rest Calibration     │
│ UBFC Raw Snippets             │ 30s BVP (64Hz) + 30s EDA (4Hz) │ Time/Statistical Extraction   │
└───────────────────────────────┴────────────────────────────────┴───────────────────────────────┘
```

### A. WESAD Wrist Blood Volume Pulse (BVP)
* **Sensor / Hardware:** Empatica E4 wristband optical photoplethysmography (BVP sensor).
* **Sampling Rate:** $64\text{ Hz}$ nominal.
* **Window Duration:** $60\text{ seconds}$ ($3,840\text{ samples}$).
* **Continuous Filtering:** 3rd-order zero-phase Butterworth bandpass filter with cutoff frequencies $f_{\text{low}} = 0.7\text{ Hz}$ and $f_{\text{high}} = 3.7\text{ Hz}$ ($42 - 222\text{ bpm}$ cardiac range). Filtering is applied continuously before window slicing to prevent edge discontinuities.
* **Per-Window Normalization:** Per-window z-score standardization:
  $$\hat{x}[n] = \frac{x[n] - \mu_w}{\sigma_w}$$
  Where $\mu_w$ and $\sigma_w$ are the sample mean and sample standard deviation of that specific 60-second window.
* **Quality & Fault Handling:** If $\sigma_w < 10^{-7}$ (flatline / sensor detachment), the engine intercepts execution, flags `quality_flag = 'DROPOUT'`, and returns $P(\text{stress}) = 0.0$ with neutral confidence without throwing a model exception.

### B. UBFC-Phys Multimodal BVP + EDA
* **Sensors:** Empatica E4 wristband BVP ($64\text{ Hz}$) and Electrodermal Activity (EDA, $4\text{ Hz}$).
* **Window Duration:** $30\text{ seconds}$ ($1,920\text{ BVP samples}$, $120\text{ EDA samples}$).
* **Feature Schema (12 Dimensions):**
  1. `bvp_mean`: Mean blood volume pulse level
  2. `bvp_std`: Standard deviation of BVP amplitude
  3. `bvp_median`: Median BVP value
  4. `bvp_iqr`: Interquartile range of BVP
  5. `bvp_slope_per_sample`: First-order linear trend slope across the window
  6. `bvp_range`: Peak-to-peak amplitude difference ($\max - \min$)
  7. `eda_mean`: Mean skin conductance level ($\mu S$)
  8. `eda_std`: Phasic/tonic skin conductance variability
  9. `eda_median`: Median skin conductance level
  10. `eda_iqr`: Interquartile range of EDA
  11. `eda_slope_per_sample`: Tonic drift rate ($\Delta \mu S / \text{sample}$)
  12. `eda_range`: Peak-to-peak skin conductance deflection

---

## 3. Model Inventory & Serialization

### Model Checkpoints Summary
| Model Identifier | File Checkpoint | Architecture | Parameters | Target Modality | Input Shape |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **WESAD Candidate** | `models/deployment_candidate/classifier.keras` | CNN–TCN–LSTM | 24,794 | Wrist BVP (64 Hz) | `(None, 3840, 1)` |
| **UBFC Calibrated** | `models/ubfc_deployment/ubfc_phys_calibrated_rf.joblib` | Random Forest (100 trees) | ~1.2 MB | BVP + EDA (12 Feats) | `(None, 12)` |
| **UBFC Uncalibrated** | `models/ubfc_deployment/ubfc_phys_uncalibrated_rf.joblib` | Random Forest (100 trees) | ~1.4 MB | BVP + EDA (12 Feats) | `(None, 12)` |

### Architectural Details
1. **WESAD Deep Hybrid Network (`model.py`):**
   - **CNN Branch:** Two 1D convolutional blocks (Conv 64 @ stride 4, Conv 32 @ stride 2) with batch normalization, ReLU activation, max pooling, and $30\%$ dropout.
   - **TCN Branch:** Four residual causal convolutional blocks with exponential dilation rates $d \in \{1, 2, 4, 8\}$ and residual skip connections, pooled via Global Average Pooling 1D.
   - **LSTM Branch:** Bidirectional LSTM with 12 recurrent units extracting long-term temporal dependencies.
   - **Classification Head:** Concatenation layer merged into a 2-unit softmax dense layer.
2. **UBFC-Phys Ensemble Classifiers:**
   - Balanced Random Forest with $100$ estimators, minimum samples per leaf $= 2$, and inverse-frequency class weighting to manage class balance ($1,960$ rest vs $3,248$ stress windows).

---

## 4. Empirical Evaluation & LOSO Verification

```
Evaluation Verification Trace:
┌────────────────────────────┬─────────────────────────────┬──────────┬──────────┬──────────┐
│ Evaluation Run             │ Protocol                    │ Accuracy │ Bal. Acc │  AUROC   │
├────────────────────────────┼─────────────────────────────┼──────────┼──────────┼──────────┤
│ WESAD Corrected LOSO       │ 15 Subjects (Full Out-of-F) │  0.8990  │    —     │  0.9424  │
│ WESAD Dev Checkpoint       │ 3 Validation Subjects Split │  0.9264* │  0.9006  │  0.9608* │
│ UBFC Uncalibrated LOSO     │ 56 Subjects (Full Out-of-F) │  0.7427  │  0.7028  │  0.8170  │
│ UBFC Rest-Calibrated LOSO  │ 56 Subjects (Full Out-of-F) │  0.9793  │  0.9823  │  0.9944  │
└────────────────────────────┴─────────────────────────────┴──────────┴──────────┴──────────┘
* Note: Dev checkpoint scores are for validation early stopping only, not independent evaluation.
```

### A. WESAD Corrected LOSO Tracing
The claimed corrected LOSO metrics were verified against `results/loso_corrected_valloss/loso_corrected_summary.json` and `loso_corrected_per_subject.csv`:
- **Total Windows:** 812 windows across 15 participants (S2–S17, excluding sensor failure S12).
- **Aggregated Overall Metrics:**
  * Accuracy: **$89.90\%$**
  * Weighted F1: **$89.84\%$**
  * Class 0 (Non-stress) F1: **$0.9354$**
  * Class 1 (Stress) F1: **$0.7684$**
  * Area Under ROC Curve (AUROC): **$0.9424$**
- **Disjoint Validation Integrity:** `loso_corrected_manifest.json` confirms that inner validation folds used `GroupShuffleSplit` on training subjects only. Outer test subjects were never seen during inner checkpoint selection, eliminating test leakage.

### B. UBFC-Phys Deployment LOSO Benchmark
The 56-subject Leave-One-Subject-Out evaluation was executed across all $5,208$ windows in `results/physiology/physiology_features_labeled.csv`:
- **Uncalibrated Setting (Raw Physiological Values):**
  * Accuracy: **$74.27\%$**
  * Balanced Accuracy: **$70.28\%$**
  * Macro F1: **$70.58\%$**
  * AUROC: **$0.8170$**
- **Rest-Calibrated Setting (Normalized to Subject Task 1 Baseline):**
  * Accuracy: **$97.93\%$**
  * Balanced Accuracy: **$98.23\%$**
  * Macro F1: **$97.69\%$**
  * AUROC: **$0.9944$**
- **Conclusion:** Individual tonic baseline subtraction resolves physiological inter-subject variance, boosting balanced accuracy by $+27.95$ percentage points.

---

## 5. Rest Calibration Theory & Implementation

### A. Mathematical Formulation
Autonomic physiological variables (such as resting baseline skin conductance and tonic heart rate) exhibit large inter-individual differences governed by skin hydration, basal metabolic rate, and sensor contact impedance.

`RestCalibrator` implements a standardized z-score transformation with numerical zero-variance protection:
$$z_j = \frac{x_j - \mu_{\text{rest}, j}}{\sigma_{\text{rest}, j}}$$

Where for each feature $j$:
$$\mu_{\text{rest}, j} = \frac{1}{K} \sum_{k=1}^K x_{\text{rest}, k, j}$$
$$\sigma_{\text{rest}, j} = \begin{cases} \sqrt{\frac{1}{K-1} \sum_{k=1}^K (x_{\text{rest}, k, j} - \mu_{\text{rest}, j})^2} & \text{if } \sigma > \epsilon \\ 1.0 & \text{otherwise} \end{cases}$$

The guard condition $\epsilon = 10^{-6}$ guarantees that static features or single-window baselines do not induce divide-by-zero exceptions or explosive numeric instability.

---

## 6. Usage Guide & Python API

### A. Running WESAD Inference
```python
from physiology_inference import WESADInferenceEngine
import numpy as np

engine = WESADInferenceEngine()

# 1. Single 60-second window (3840 samples @ 64 Hz)
window = np.random.randn(3840, 1).astype(np.float32)
res = engine.predict_window(window)
print(f"Prediction: {res.class_name} | P(stress): {res.probability:.3f} | Latency: {res.latency_ms:.1f}ms")

# 2. Continuous BVP recording with automatic sliding window
long_signal = np.random.randn(64 * 300)  # 5 minutes
stream_results = engine.predict_continuous(long_signal, fs=64.0, step_s=5.0)
for r in stream_results:
    print(f"[{r.metadata['start_s']:.0f}s - {r.metadata['end_s']:.0f}s]: P(stress) = {r.probability:.3f}")
```

### B. Running UBFC Inference with Rest Calibration
```python
from calibration import RestCalibrator
from physiology_inference import UBFCInferenceEngine
import pandas as pd

# 1. Fit calibrator on participant rest baseline data
df_rest = pd.read_csv("participant_rest_task1.csv")
calibrator = RestCalibrator().fit(df_rest)

# 2. Initialize engine with calibrator
engine = UBFCInferenceEngine(calibrator=calibrator)

# 3. Predict on active test window
active_features = {
    'bvp_mean': -0.12, 'bvp_std': 45.2, 'bvp_median': 10.1, 'bvp_iqr': 78.4,
    'bvp_slope_per_sample': -0.0003, 'bvp_range': 210.5, 'eda_mean': 0.42,
    'eda_std': 0.015, 'eda_median': 0.41, 'eda_iqr': 0.02, 'eda_slope_per_sample': 0.0001,
    'eda_range': 0.08
}
result = engine.predict_features(active_features, use_calibration=True)
print(f"Result: {result.class_name} (Confidence: {result.confidence:.2%})")
```

### C. Unified Multi-Dataset Dispatcher
```python
from physiology_inference import predict_stress

# Automatically delegates to appropriate adapter and validates input shapes
result = predict_stress(dataset="WESAD", input_data=raw_window_array)
```

---

## 7. Known Limitations & Technical Safeguards

1. **Sensor Morphology Incompatibility:**
   - Passing CLAS data or Shimmer3 finger PPG signals into the WESAD wrist BVP model is explicitly blocked. Finger transmission PPG has inverted optical absorption kinetics compared to wrist reflection BVP.
2. **Task Proxy Interpretation:**
   - Labels in UBFC-Phys represent experimental stress tasks (oral speech and mental arithmetic) versus seated rest. Predictions must be reported as *stress-task proxy*, not clinical distress or panic disorder diagnoses.
3. **Rest Calibration Requirement:**
   - Rest calibration requires at least one baseline rest window prior to testing. For zero-baseline scenarios (e.g. ambulatory emergency monitoring), the uncalibrated model must be used, which has an expected balanced accuracy of $\approx 70.3\%$.
4. **Hardware Sensor Drivers:**
   - The current engine operates on numerical buffers (`numpy.ndarray` and `pandas.DataFrame`). Physical Bluetooth Low Energy (BLE) streaming drivers for Empatica E4 or Polar H10 are not included and must feed data via this API.

