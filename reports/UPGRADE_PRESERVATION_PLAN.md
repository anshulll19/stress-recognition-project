# SEC Stress Recognition System: Stage A Preservation Audit

**Document:** `reports/UPGRADE_PRESERVATION_PLAN.md`  
**Execution Stage:** Stage A (Formal Audit & Preservation Plan)  
**System:** Physiology-First, Video-Assisted Multimodal Stress Monitoring Dashboard  
**Date:** 2026-10-30  

---

## 1. Git Branch & Working Tree State

* **Active Branch:** `main` (commit `158e0b2`: *"Add raw UBFC-Phys signal windows"*)
* **Repository Root:** `/Users/nishoosingh/stress-recognition-project`
* **Uncommitted / Active Files:**
  * `app.py` — Upgraded 6-tab healthcare analytics Streamlit application
  * `demo/` — UI adapters, styling modules (`demo/ui/modern_theme.py`, `demo/ui/hud_styles.py`), visualizers, and metric loaders
  * `models/deployment_candidate/` — WESAD CNN-TCN-LSTM checkpoint (`classifier.keras`) and training manifest
  * `models/ubfc_deployment/` — UBFC-Phys 56-subject Random Forest checkpoints (calibrated & uncalibrated), `loso_per_subject.csv`, and manifest
  * `models/video_distress_upperface.joblib` — 19-feature speech-confound-free video model
  * `models/video_upperface_features.json` — 19 upper-face landmark feature schema
  * `reports/` — `PHASE1_PROJECT_STATUS.md`, `PHYSIOLOGY_MODEL_REPORT.md`, `CURRENT_PROTOTYPE_CAPABILITIES.md`, `UPGRADE_PRESERVATION_PLAN.md`
  * `src/calibration.py` — `RestCalibrator` baseline normalization engine
  * `src/physiology_inference.py` — Reusable physiology inference engine and CLAS blocker
  * `src/multimodal_fusion.py` — Dynamic multimodal fusion engine and upper-face video pipeline
  * `test_adapters.py` — 10-test adapter validation suite
  * `test_physiology_inference.py` — 5-test integration suite
  * `ubfc_pipeline/s8_late_fusion.py` — Late fusion experiment script
  * `ubfc_pipeline/team_runner.py.before_provenance.bak` & `src/loso_harness.py.bak` — Backups

---

## 2. Existing Streamlit Dashboard Structure & Functionality

The original prototype was structured into five functional areas:

1. **Tab 1: Physiology Inspection:**
   * UBFC-Phys BVP (64 Hz) and EDA (4 Hz) waveforms for 56 participants.
   * WESAD wrist BVP (64 Hz) waveforms for 15 subjects.
   * CLAS sensor physics audit panel blocking cross-dataset inference.
   * Model selection: LSTM, TCN, and Random Forest.
2. **Tab 2: Facial Behavioral Analysis & Confound Isolation:**
   * 23-feature full-face baseline vs. 19-feature upper-face speech-confound ablation.
   * Delta calculation displaying mouth movement bias shifts.
   * Face detection fraction quality monitoring.
   * Full 972-window confound ablation summary table.
3. **Tab 3: Multimodal Late Fusion:**
   * Synchronized 10s windows across 18 participants (972 windows).
   * Fixed late fusion blends (`blend_0.3`, `blend_0.2`, `blend_0.5`, `blend_0.3_uf`).
   * Dynamic quality-aware weighting based on video face detection.
   * Synchronized session trajectory timeline.
4. **Tab 4: Temporal Trend Detector:**
   * Past-only causal Theil-Sen slope estimation on distress probabilities.
   * Anti-flicker hysteresis and cold-start window buffers.
   * Evaluation on real UBFC-Phys out-of-fold trajectories and synthetic benchmark patterns.
5. **Tab 5: Research Evaluation & Benchmarks:**
   * Performance comparison table across all architectures.
   * 15-subject WESAD corrected LOSO per-subject breakdown.
   * 56-participant UBFC deployment model per-subject breakdown.
   * CLAS failure analysis summary.

---

## 3. Physiological Models & Checkpoints

| Checkpoint Path | Architecture | Input Shape | Protocol & Training Cohort | Status & Performance |
| :--- | :--- | :--- | :--- | :--- |
| `models/deployment_candidate/classifier.keras` | 1D-CNN + TCN + LSTM + Dense | `(None, 3840, 1)` (60s @ 64Hz) | WESAD wrist BVP (15 subjects). Early-stopping on dev subjects (S10, S4, S6). | Verified loading. Independent corrected LOSO: **Acc=89.9%, AUROC=0.942**. |
| `models/ubfc_deployment/ubfc_phys_calibrated_rf.joblib` | Random Forest (100 estimators, depth=8) | `(None, 12)` BVP+EDA features | UBFC-Phys (56 participants). Normalized against individual Task 1 rest baseline via `RestCalibrator`. | Verified loading with `_fill_dtype` patch. LOSO: **BalAcc=98.2%, AUROC=0.994**. |
| `models/ubfc_deployment/ubfc_phys_uncalibrated_rf.joblib`| Random Forest (100 estimators, depth=8) | `(None, 12)` raw features | UBFC-Phys (56 participants). Uncalibrated baseline. | Verified loading. LOSO: **BalAcc=70.3%, AUROC=0.817**. |
| `results/temporal_models/tcn_lstm_oof_predictions_full.csv`| Deep TCN & LSTM OOF records | 5,880 windows across 56 subjects | UBFC-Phys raw windows. Deep temporal benchmarks. | Verified CSV artifacts. LSTM: BalAcc=76.1%, AUROC=0.799. TCN: BalAcc=69.3%, AUROC=0.796. |

---

## 4. Video Classifiers, Feature Schemas & MediaPipe Assets

| Asset Path | Type / Model | Feature Count & Schema | Notes & Confound Analysis |
| :--- | :--- | :--- | :--- |
| `models/video_distress_upperface.joblib` | Random Forest Classifier | 19 Features: 16 brow/eye landmarks + 3 head pose angles | **Clean affect model.** Mouth features removed to eliminate speech motion confound. LOSO: **BalAcc=90.6%, AUROC=0.968**. |
| `models/video_distress_baseline.joblib` | Random Forest Pipeline (SimpleImputer + RF) | 23 Features: 19 upper-face + 4 mouth features | **Confounded model.** AUROC=0.994 driven by mouth opening during oral speech (Task T2). |
| `models/video_upperface_features.json` | Feature schema list | 19 string identifiers | Exact column alignment for upper-face video inference. |
| `models/video_feature_names.json` | Feature schema list | 23 string identifiers | Full-face baseline feature ordering. |
| `models/video_model_metadata.json` | JSON metadata | Training hyperparams & window sizes | 10s window size, 5s stride, 30 fps video. |
| `models/face_landmarker.task` | MediaPipe Binary Model (~4 MB) | 478 3D landmarks + 52 blendshapes | Standard vision pipeline asset for facial landmark extraction. |

---

## 5. Multimodal Fusion & Decision Logic

* **Physiology-First Priority:** Cardiovascular (BVP) and sympathetic (EDA) responses provide the primary evidence for acute distress ($w_{phy} = 0.70$). Facial video provides secondary behavioral support ($w_{vid} = 0.30$).
* **Sensor Quality Gating & Fallback:**
  * If a physiological sensor disconnects or flatlines ($\sigma = 0$ or quality flag = `DROPOUT`), the decision engine automatically diverts 100% weight to the video branch ($w_{phy}=0.0, w_{vid}=1.0$) with a clear fallback indicator.
  * If face detection drops below 30% (occlusion, subject turned away, poor lighting), the decision engine shifts 100% authority to physiology ($w_{phy}=1.0, w_{vid}=0.0$).
  * If both modalities fail simultaneously, the system outputs an explicit **`INSUFFICIENT_DATA`** state.
* **Preserved Late Fusion Implementations:**
  * Fixed weighted late fusion (`blend_0.3`, `blend_0.3_uf`, `blend_0.2`, `blend_0.5`).
  * Dynamic quality-aware weighting.
  * Stacked logistic regression meta-learner (Inner-LOSO).

---

## 6. Trend Detection Implementation

* **Causal Theil-Sen Estimator:** Operates strictly on historical model distress probabilities ($P(\text{Stress})$) within a sliding past-only buffer.
* **No Future Peeking:** Verified strictly causal; no future imputation or bidirectional filtering.
* **Anti-Flicker Hysteresis:** Prevents noisy label oscillation around the 0.5 decision boundary:
  * Slope threshold: $m > +0.005 / \text{sec} \implies \text{RISING}$
  * Slope threshold: $m < -0.005 / \text{sec} \implies \text{FALLING}$
  * Magnitude threshold: $|m| \le 0.005 / \text{sec} \implies \text{STABLE}$
* **Cold-Start Guard:** Requires a minimum window history ($N \ge 3$) before generating trend assertions; returns `INSUFFICIENT_DATA` during warm-up.

---

## 7. Research Benchmarks & Evaluation Protocols

1. **WESAD Corrected LOSO (15 Subjects):**
   * Evaluated strictly subject-out on 15 participants (S2 to S17).
   * Metric source: `results/loso_corrected_valloss/loso_corrected_summary.json`.
   * Result: **Accuracy: 89.9%**, **Macro F1: 0.887**, **AUROC: 0.942**.
   * *Distinction:* Dev checkpoint metrics (S10, S4, S6 validation: Acc=92.6%, AUROC=0.961) were used solely for early stopping and are never represented as independent test results.
2. **UBFC-Phys Deployment Model LOSO (56 Participants):**
   * Full Leave-One-Subject-Out cross-validation across all 56 subjects.
   * Metric source: `models/ubfc_deployment/loso_summary.json` & `loso_per_subject.csv`.
   * Rest-Calibrated RF: **Balanced Accuracy: 98.2%**, **AUROC: 0.994**, **Macro F1: 0.978**.
   * Uncalibrated RF: **Balanced Accuracy: 70.3%**, **AUROC: 0.817**, **Macro F1: 0.697**.
3. **UBFC Multimodal Aligned Cohort (18 Participants · 972 Windows):**
   * Synchronized BVP, EDA, and video frames.
   * Full-Face Blend 0.3: BalAcc=97.1%, AUROC=0.996 (confounded).
   * Upper-Face Blend 0.3 (Clean): **BalAcc=81.0%**, **AUROC=0.933**.
   * Upper-Face Video Alone: **BalAcc=90.6%**, **AUROC=0.968**.
4. **CLAS Transfer Failure Protocol:**
   * Zero-shot cross-dataset transfer between CLAS (Shimmer3 finger PPG) and WESAD (Empatica E4 wrist BVP).
   * WESAD $\to$ CLAS: AUROC = 0.367 (sub-random).
   * CLAS $\to$ WESAD: AUROC = 0.223 (inverted ranking).
   * Proves optical hardware and sensor placement incompatibility.

---

## 8. Python Environment & Dependency Versions

* **Python Runtime:** Python 3.12 (64-bit, Clang 17.0.0)
* **Operating System:** macOS (arm64, Apple Silicon)
* **Core Libraries & Frameworks:**
  * `streamlit`: 1.43.2 / 1.66.0
  * `scikit-learn`: 1.9.1 (Note: `_fill_dtype` compatibility patch applied for SimpleImputer persistence)
  * `tensorflow`: 2.18.0 / 2.21.0
  * `mediapipe`: 1.2.0
  * `opencv-python`: 5.0.0
  * `pandas`: 2.2.3 / 3.0.6
  * `numpy`: 1.26.4 / 2.5.3
  * `matplotlib`: 3.10.0 (Runtime requirement: `MPLCONFIGDIR=/tmp`)
  * `joblib`: 1.4.2

---

## 9. Protected Files — Do Not Overwrite or Delete

1. **Trained Checkpoints:**
   * `models/deployment_candidate/classifier.keras`
   * `models/deployment_candidate/manifest.json`
   * `models/ubfc_deployment/ubfc_phys_calibrated_rf.joblib`
   * `models/ubfc_deployment/ubfc_phys_uncalibrated_rf.joblib`
   * `models/ubfc_deployment/loso_per_subject.csv`
   * `models/ubfc_deployment/loso_summary.json`
   * `models/video_distress_baseline.joblib`
   * `models/video_distress_upperface.joblib`
   * `models/video_feature_names.json`
   * `models/video_upperface_features.json`
   * `models/face_landmarker.task`
2. **Evaluation Ground-Truth Results:**
   * `results/loso_corrected_valloss/` (all CSVs and JSONs)
   * `results/fusion_lstm/aligned_oof_predictions.csv`
   * `results/fusion_lstm/fusion_summary.csv`
   * `results/fusion_lstm/fusion_confound_analysis.csv`
   * `results/physiology/raw_signal_windows.npz`
   * `results/physiology/raw_signal_window_metadata.csv`
   * `results/temporal_models/tcn_lstm_oof_predictions_full.csv`
3. **Core Datasets:**
   * `data/wesad/wesad_binary.npz`
   * `data/ubfc_phys_subset/`
   * `data/clas/`

---

## 10. Documented Inconsistencies & Evolution

| Item | Phase 1 Audit (`PHASE1_PROJECT_STATUS.md`) | Current Prototype Documentation | Resolution & State |
| :--- | :--- | :--- | :--- |
| **UBFC Deployment Checkpoint** | Noted that no deployment RF checkpoint existed (only TCN/LSTM OOF predictions in results/). | Trained and exported `models/ubfc_deployment/ubfc_phys_calibrated_rf.joblib` (56 subjects). | Deployment model exists, verified loading, achieves 98.2% BalAcc under LOSO. |
| **Rest Calibration Module** | Existed only as an experimental script (`src/finalize_ubfc_t1_pilot.py`). | Implemented as reusable module `src/calibration.py` with zero-std protection and serialization. | Fully functional in `src/calibration.py` and passing 5/5 integration tests. |
| **Upper-Face Video Checkpoint** | Analyzed in CSVs (`fusion_confound_analysis.csv`), but no standalone `.joblib` model. | Exported `models/video_distress_upperface.joblib` with 19 upper-face features. | Trained, verified loading, integrated into `MultimodalHUDEngine`. |
| **WESAD Reported Results** | Earlier run in `wesad_binary_results.npz` reported Acc=0.970, AUC=0.967. Audit identified this was uncorrected. | System strictly quotes the independent corrected LOSO run: **Acc=89.9%, AUROC=0.942**. | Corrected LOSO is the authoritative benchmark; uncorrected run is rejected. |
| **CLAS Status** | Documented self-supervised MAE encoder weights with no classification head. | Permanently blocked in `src/physiology_inference.py` with explicit physics explanations. | Enforced at code level: calling CLAS inference throws a descriptive `ValueError`. |

---

## 11. Verification Status Matrix

| System Component | Category | Evidence & Smoke Test Result | Notes / Safe Guard |
| :--- | :--- | :--- | :--- |
| **WESAD Inference Pipeline** | **VERIFIED IMPLEMENTATION** | Loaded `models/deployment_candidate/classifier.keras` (`input_shape=(None, 3840, 1)`). Passed `test_02_model_loading_and_inference` and `test_09_full_replay_run`. | Evaluated via independent corrected LOSO (Acc=89.9%, AUC=0.942). |
| **UBFC Rest Calibration** | **VERIFIED IMPLEMENTATION** | Passed `test_03_rest_calibration` in `test_physiology_inference.py`. Validated zero-variance protection and serialization. | Normalizes against Task 1 rest baseline ($z = \frac{x - \mu}{\sigma}$). |
| **UBFC Deployment RF** | **VERIFIED IMPLEMENTATION** | Loaded `models/ubfc_deployment/ubfc_phys_calibrated_rf.joblib` (100 estimators). Full 56-subject LOSO verified (BalAcc=98.2%, AUC=0.994). | Standalone deployable checkpoint. |
| **Upper-Face Video Pipeline** | **VERIFIED IMPLEMENTATION** | Loaded `models/video_distress_upperface.joblib` (Pipeline: Imputer + RF, 19 features). | Speech-confound-free video baseline (AUROC=0.968). |
| **Face Landmarker & Vision** | **VERIFIED IMPLEMENTATION** | Executed smoke test on `data/ubfc_phys_subset/s1/frames/frame_001.jpg`: Haar Cascade detected 1 face (`[176, 199, 330, 330]`), MediaPipe FaceLandmarker extracted 478 3D landmarks. | Confirmed functional on local disk frames. |
| **Multimodal Decision Engine**| **VERIFIED IMPLEMENTATION** | Dynamic weighting verified: $w_{phy}=0.70, w_{vid}=0.30$ default; sensor dropout shifts to video ($w_{vid}=1.0$); occlusion shifts to physio ($w_{phy}=1.0$). | Passed temporal alignment test `test_07_fusion_alignment`. |
| **Causal Trend Estimator** | **VERIFIED IMPLEMENTATION** | Passed causality test `test_06_missing_data_and_insufficient_windows`. Theil-Sen slope with anti-flicker hysteresis. | Pure past-only calculation (no future leakage). |
| **CLAS Cross-Transfer** | **VERIFIED BLOCKED** | Passed `test_08_clas_integrity_enforcement`. Throws explicit descriptive rejection. | Sensor physics and optical transmission mismatch. |
| **Live Hardware Webcam** | **DOCUMENTED / DISABLED** | OpenCV video decoding and frame processing verified. Direct live camera ingest disabled. | Requires live device permission and background thread queue; marked `NOT VERIFIED / DISABLED` in UI. |
| **Health Profile ML Scoring** | **INCOMPLETE / UI SHELL ONLY** | Form inputs (PSS-10, BP, HR, Sleep) validated and scored via clinical questionnaire rubrics. | **Not connected to ML models.** No pseudo-scientific risk score generated without validated clinical cohort. |

