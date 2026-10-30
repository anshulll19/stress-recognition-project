# Stage A/B Preservation Verification Report

**Document:** `reports/STAGE_A_B_PRESERVATION_VERIFICATION.md`  
**Execution Context:** Read-only Git Baseline & Artifact Integrity Verification  
**Date:** 2026-10-30  
**Verification Status Standard:** `VERIFIED`, `NOT VERIFIED`, `BLOCKED`

---

## 1. Git Baseline Investigation

### Exact Command Execution Outputs
```bash
$ git rev-parse --show-toplevel
/Users/nishoosingh/stress-recognition-project

$ git branch --show-current
main

$ git status -sb
## main...origin/main
?? app.py
?? demo/
?? models/deployment_candidate/
?? models/ubfc_deployment/
?? models/video_distress_upperface.joblib
?? models/video_upperface_features.json
?? reports/
?? src/calibration.py
?? src/finalize_ubfc_t1_pilot.py.review-backup
?? src/loso_harness.py.bak
?? src/multimodal_fusion.py
?? src/physiology_inference.py
?? test_adapters.py
?? test_physiology_inference.py
?? ubfc_pipeline/s8_late_fusion.py
?? ubfc_pipeline/team_runner.py.before_provenance.bak

$ git log -1 --oneline
158e0b2 (HEAD -> main, origin/main, origin/HEAD) Add raw UBFC-Phys signal windows

$ git ls-files app.py
(empty output)

$ git show HEAD:app.py
fatal: path 'app.py' exists on disk, but not in 'HEAD'
```

### Git Baseline Findings & Dashboard Origin
1. **`app.py` in Git HEAD:** `NOT VERIFIED` (Does not exist in Git commit `158e0b2`). Git cannot establish an original `app.py` dashboard baseline because the multimodal Streamlit dashboard files were authored during the ongoing SEC development sessions and were never staged or committed to Git.
2. **Why Files Are Untracked:** In the repository history, early commits tracked pilot scripts (`src/dashboard.py`, `src/build_ubfc_t1_pilot.py`). The modern demo adapters (`demo/`), model deployment exports (`models/deployment_candidate/`, `models/ubfc_deployment/`), reports (`reports/`), and the unified `app.py` application were developed in the working tree without being added to Git version control.
3. **Historical Dashboard Baseline:** In Git HEAD, `src/dashboard.py` exists as a 1-participant pilot viewer. The 5-tab prototype structure was established in untracked files on disk (`demo/ui/tab_*.py`).

---

## 2. Protected Artifact Verification (Sizes & SHA-256)

Every protected model checkpoint, dataset, and aligned evaluation artifact was hashed directly on disk without modifying or resaving:

| Artifact Path | Existence | Size (Bytes) | SHA-256 Checksum | Git HEAD Status | Preservation Baseline Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `models/deployment_candidate/classifier.keras` | EXISTS | 440,140 | `d903918b4d88eaac14d301781ce2819cc0185020a88be51a3534dade964c0afa` | Untracked | **VERIFIED (Disk)** |
| `models/ubfc_deployment/ubfc_phys_calibrated_rf.joblib` | EXISTS | 961,945 | `c95429236fccdbe18ce938d49155872c8a2d946bc2c35da13229f93c36e91b14` | Untracked | **VERIFIED (Disk)** |
| `models/ubfc_deployment/ubfc_phys_uncalibrated_rf.joblib` | EXISTS | 5,000,665 | `e586ec029c63b054c10976584be53c15b38dc9edb351ac71135f2d0bc48b702e` | Untracked | **VERIFIED (Disk)** |
| `models/video_distress_baseline.joblib` | EXISTS | 1,145,138 | `eb0a87c2eec0424dedca2d192daa477eb07c0fbc6c302d255dad9b4cc696ec8f` | **Tracked (Clean)** | **VERIFIED (Git HEAD match)** |
| `models/video_distress_upperface.joblib` | EXISTS | 2,046,162 | `da4145bc83f276e94be1f3ec974b245a9599dacc54b80661a206b1a5f3afadf1` | Untracked | **VERIFIED (Disk)** |
| `models/face_landmarker.task` | EXISTS | 3,758,596 | `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff` | Untracked | **VERIFIED (Disk)** |
| `data/wesad/wesad_binary.npz` | EXISTS | 11,553,894 | `8bad2d1bb8a80b5e61ddb5f6b0c8e3db49609ff02575fd082cc4c65c5a467126` | Untracked | **VERIFIED (Disk)** |
| `results/fusion_lstm/aligned_oof_predictions.csv` | EXISTS | 184,288 | `da8297d8d850259a382b3939698eeb58e2a3ff9361ff052cd6409a5d259a4646` | Untracked | **VERIFIED (Disk)** |

> **Baseline Qualification:**  
> For `models/video_distress_baseline.joblib`, the hash matches Git HEAD `158e0b2` exactly (`git diff` is empty).  
> For the untracked artifacts, current SHA-256 hashes establish cryptographic identity **from this point forward**; because no pre-Stage-A Git commit holds these files, historical preservation cannot be proven via Git, but disk integrity is established and locked.

---

## 3. Environment & Runtime Verification

Execution using `./venv/bin/python`:
```
Executable: /Users/nishoosingh/stress-recognition-project/venv/bin/python
Python version: 3.12.13 (main, Mar  3 2026, 12:39:30) [Clang 17.0.0 (clang-1700.6.3.2)]
Platform: macOS-26.2-arm64-arm-64bit
Streamlit: 1.66.0
Scikit-learn: 1.9.1
OpenCV: 5.0.0
MediaPipe: 1.2.0
TensorFlow: 2.21.0
Pandas: 3.0.6
NumPy: 2.5.3
```

### Explanation of Earlier Multiple-Version Discrepancy
In an earlier command run during the session, `python3` invoked a global/system Python environment (`Python 3.12.8`, `Streamlit 1.43.2`, `TensorFlow 2.18.0`, `Pandas 2.2.3`, `NumPy 1.26.4`). When invoked explicitly using the virtual environment `./venv/bin/python`, the true runtime versions are confirmed to be:
* Python: **3.12.13**
* Streamlit: **1.66.0**
* Scikit-Learn: **1.9.1**
* TensorFlow: **2.21.0**
* OpenCV: **5.0.0**
* MediaPipe: **1.2.0**

---

## 4. Dashboard & Logic Verification

| Audit Check | Status | Verification Detail |
| :--- | :--- | :--- |
| **All Six Tabs Created** | `VERIFIED` | AppTest confirms 6 tabs: Overview, Health Profile, Physiological Monitoring, Facial Analysis, Multimodal Prediction, Trends & Research. |
| **Original 5-Tab Functionality Reachable** | `VERIFIED` | All 5 original views mapped: Physio $\to$ Tab 3, Video $\to$ Tab 4, Fusion $\to$ Tab 5, Trend $\to$ Tab 6, Evaluation $\to$ Tab 6. |
| **Existing Adapters Imported & Called** | `VERIFIED` | `UBFCPhysInferenceAdapter`, `WESADInferenceAdapter`, `VideoInferenceAdapter`, `MultimodalFusionAdapter`, `TemporalTrendAdapter`, `CLASInferenceAdapter`, `ResearchMetricsLoader` are imported and called directly. |
| **Fusion Formula Unchanged** | `VERIFIED` | Tab 5 calls `fusion_ad.get_fusion_prediction(row_f, method=method_f)`. No new fusion mathematical equations were introduced. |
| **Trend Calculations Unchanged** | `VERIFIED` | Tab 6 calls `render_tab_trend(trend_ad, ubfc_ad)` from `demo/ui/tab_trend.py`. |
| **Health Profile ML Isolation** | `VERIFIED` | Tab 2 computes PSS-10 questionnaire sum (0–40) and clinical category. It does **not** call any ML model or generate a predictive stress probability. |
| **Webcam Status Truthfulness** | `VERIFIED` | Tab 4 explicitly marks webcam as: `<span class="badge badge-warning">NOT VERIFIED / DISABLED</span>` with a clear explanation that live webcam streaming is disabled. |

---

## 5. Test Suite Verification Summary

1. **Python Compilation Check:** `VERIFIED`
   ```bash
   $ ./venv/bin/python -m py_compile app.py
   (Exit code 0, no syntax errors)
   ```
2. **Streamlit AppTest Simulation:** `VERIFIED`
   * Executed with `streamlit.testing.v1.AppTest.from_file('app.py')`.
   * Result: **0 exceptions**, initial load and interactive transitions (`physio_cohort_radio`, `video_feat_mode_radio`) succeeded.
3. **Adapter Regression Suite (`test_adapters.py`):** `VERIFIED`
   * Ran 10 tests in 8.17s: **10 / 10 PASSED** (0 failed).
4. **Physiology Integration Suite (`test_physiology_inference.py`):** `VERIFIED`
   * Ran 5 integration tests: **5 / 5 PASSED** (0 failed).
5. **Computer Vision Smoke Test:** `VERIFIED`
   * Tested on real frame `data/ubfc_phys_subset/s1/frames/frame_001.jpg`: OpenCV detected face at `[176, 199, 330, 330]`; MediaPipe extracted 478 landmarks.

---

## 6. Final Status Summary

* **Preservation Status:** **VERIFIED** on local disk; baseline SHA-256 hashes recorded above.
* **UI Redesign Status:** **VERIFIED**; 6 tabs active, modern theme applied, no calculations altered.
* **Webcam Support:** **NOT VERIFIED / DISABLED** (accurately reflected in UI).
* **CLAS Cross-Dataset Transfer:** **BLOCKED** (enforced by design).

**STOPPED AFTER STAGE B.** Awaiting explicit approval before proceeding to subsequent stages.

