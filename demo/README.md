# Multimodal Stress Recognition & Distress Trend Prediction Demo

Research demonstration application developed for the SEC (5th semester) project on physiological stress recognition, video behavioral analysis, decision-level multimodal fusion, and causal temporal trend prediction.

The application replays and validates actual trained models, verified out-of-fold predictions, and multimodal fusion pipelines without fabricated predictions, synthetic probabilities, or simulated hardware sensors.

---

## 1. Quick Launch

Ensure you are in the project root directory and activate the existing virtual environment:

```bash
# Activate the existing virtual environment
source venv/bin/activate

# Launch the Streamlit application
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

To run the automated adapter test suite:

```bash
python test_adapters.py
```

---

## 2. System Architecture

The application is modularized to cleanly isolate the demonstration and validation logic from repository research code and raw data:

```text
stress-recognition-project/
├── app.py                     # Main Streamlit dashboard entrypoint
├── test_adapters.py           # Automated smoke & contract test suite
├── demo/
│   ├── README.md              # Setup, architecture, and SEC evaluation guide
│   ├── adapters/              # Dataset-specific inference and validation adapters
│   │   ├── base.py            # Base adapter interface & dataclasses
│   │   ├── ubfc_adapter.py    # UBFC-Phys multimodal replay & OOF prediction adapter
│   │   ├── wesad_adapter.py   # WESAD CNN-TCN-LSTM deployment model adapter
│   │   ├── video_adapter.py   # 23-feature Random Forest video baseline adapter
│   │   ├── fusion_adapter.py  # Synchronized decision-level late fusion adapter
│   │   ├── trend_adapter.py   # Causal Theil-Sen slope & safe hysteresis adapter
│   │   └── clas_adapter.py    # CLAS audit and limitations enforcement adapter
│   ├── utils/
│   │   ├── metrics.py         # Empirical research evaluation metric loader
│   │   └── visualization.py   # Research dark-theme plotting helpers (matplotlib/native)
│   └── ui/
│       ├── styles.py          # Dark theme CSS, metrics cards & badges
│       ├── tab_physio.py      # Tab 1: Physiological Stress Prediction
│       ├── tab_video.py       # Tab 2: Video-Based Facial Distress Analysis
│       ├── tab_fusion.py      # Tab 3: Multimodal Late Fusion
│       ├── tab_trend.py       # Tab 4: Temporal Trend Prediction
│       └── tab_eval.py        # Tab 5: Evaluation & Research Results Audit
```

---

## 3. Model & Dataset Compatibility Table

| Dataset | Modality & Sampling | Windows Available | Compatible Model / Checkpoint | Evaluation Protocol | Reported Performance | Demo Inference Status |
|---|---|---|---|---|---|---|
| **UBFC-Phys** | BVP (64 Hz) + EDA (4 Hz) | 5,880 (56 subjects, 10s windows) | LSTM & TCN Temporal Architectures | Full LOSO (56 subjects) | **LSTM:** Acc 80.4%, Bal-Acc 76.1%, AUC 0.799<br>**TCN:** Acc 77.2%, Bal-Acc 69.3%, AUC 0.796<br>**Rest-Calibrated RF:** Bal-Acc 91.1%, Macro-F1 0.881 | **READY** (Synchronous signal replay with verified OOF predictions) |
| **WESAD** | Single-channel BVP (64 Hz) | 812 (15 subjects, 60s windows) | CNN–TCN–LSTM (`models/deployment_candidate/classifier.keras`) | Corrected LOSO (15 subjects) vs Val Split (3 subjects) | **Corrected LOSO:** Acc 89.9%, Weighted F1 0.898, AUC 0.942<br>**Val Split:** Best Val Loss 0.3092 (epoch 27) | **READY** (Live Keras inference on 3840-sample windows) |
| **UBFC-Phys Video** | 23 Geometrical Facial Features | 972 (18 subjects, 10s windows) | Random Forest (`models/video_distress_baseline.joblib`) | Subject-wise LOSO (18 subjects) | Acc 96.6%, Bal-Acc 96.7%, Macro-F1 0.962 | **READY** (Offline 23-feature window replay & MediaPipe diagnostic) |
| **UBFC Multimodal Fusion** | Aligned LSTM Physio + RF Video | 972 (18 subjects, 10s windows) | Decision Late Fusion (`blend_0.3`, stacked LOSO) | Subject-wise LOSO (18 subjects) | **Blend 0.3:** Bal-Acc 97.1%, Macro-F1 0.966, **AUC 0.996** ([0.990, 1.000]) | **READY** (Synchronized decision fusion on matched intervals) |
| **CLAS** | Shimmer3 PPG (64 Hz) | 2,271 windows (53 subjects) | Pretrained MAE Encoder (`pretrained_clas_mae_encoder.weights.h5`) | Self-supervised pretraining (no labels) | Cross-dataset E2/E3 failed (AUC 0.223 – 0.367) | **DISABLED** (Audit only; no trained classifier head exists) |

---

## 4. University SEC Evaluation Demo Walkthrough

Use this 5-minute walkthrough during your project demonstration:

### Step 1: Physiological Stress Prediction (Tab 1)
1. Navigate to **🫀 Physiological Stress**.
2. Select **UBFC-Phys (Multimodal BVP + EDA)**:
   - Select Participant `#1` and model `LSTM`.
   - Scrub through Window Index: Observe how Task 1 (Rest, 0–180s) correctly yields non-stress predictions, transitioning to Task 2 (Speech) and Task 3 (Mental Arithmetic) with stress-related classifications.
   - Point out the synchronized BVP (64 Hz) pulse peaks and EDA (4 Hz) tonic skin conductance shifts.
3. Switch to **WESAD (Wrist BVP)**:
   - Select validation participant `S10` or test participant `S2`.
   - Observe real-time neural network inference using the 74,288-parameter CNN-TCN-LSTM deployment checkpoint (`classifier.keras`).
   - Note the distinction between checkpoint selection validation and independent LOSO test results.
4. Select **CLAS (Audit & Limitations)**:
   - Show that the system transparently identifies the absence of a trained classifier head and explains the failure of cross-dataset transfer (E2/E3), proving research integrity.

### Step 2: Video-Based Facial Analysis (Tab 2)
1. Navigate to **📹 Video-Based Facial Analysis**.
2. Under **Recorded Video Windows Replay**, select Participant `s3`.
3. Scrub through windows: Show how the 23 geometrical features (motion, eye aspect ratio, mouth width/open, head-turn/nod/tilt) are fed into the Random Forest baseline.
4. Highlight the **Confound Audit Notice**: Explain that the high accuracy (96.7%) is strongly influenced by speech kinematics during Task 2 rather than pure internal psychological distress.
5. Switch to **Live Webcam & MediaPipe Pipeline Diagnostics**: Show the loaded MediaPipe FaceLandmarker model and explain the 10-second buffer requirements.

### Step 3: Multimodal Late Fusion (Tab 3)
1. Navigate to **🔀 Multimodal Fusion**.
2. Select Participant `s3` and fusion method `blend_0.3` (30% Physio + 70% Video).
3. Point out the verified synchronization: Participant ID, task phase, and time interval match deterministically.
4. Compare the modality scores: Show how late fusion (AUC 0.996) improves classification stability and guards against unimodal signal degradation.

### Step 4: Temporal Trend Analysis (Tab 4)
1. Navigate to **📈 Temporal Trend Analysis**.
2. Select Participant `#1` and model `LSTM`.
3. Move the window slider along the task trajectory:
   - Windows 0–11 display `INSUFFICIENT DATA (COLD START)`.
   - Once K=12 windows accumulate, observe the Theil-Sen slope calculation (in logit per 60 seconds) and the displayed trend label (`STABLE`, `INCREASING`, `DECREASING`).
4. Highlight the **Safe Anti-Flicker Hysteresis**: Show that if data quality drops, the detector immediately reverts to `INSUFFICIENT DATA` to prevent presenting stale trends as current reliable results.

### Step 5: Research Benchmarks & Evaluation Audit (Tab 5)
1. Navigate to **📊 Evaluation & Research Results**.
2. Walk through:
   - Full 56-participant UBFC-Phys LOSO results and confusion matrices (5,880 windows).
   - WESAD Corrected LOSO summary (15 participants, 812 windows, 89.9% accuracy, 0.942 AUC).
   - Multimodal late fusion comparison table across 7 fusion strategies.
   - Confound ablation analysis and cross-dataset transfer failure diagnosis.

---

## 5. Test Suite Report

All 10 validation suites in `test_adapters.py` pass without errors:

| # | Test Suite | Scope | Result |
|---|---|---|---|
| 1 | `test_01_dataset_inventory` | WESAD (812 windows), UBFC (5,880 windows), Video (972 windows) schema checks | **PASS** |
| 2 | `test_02_model_loading_and_inference` | Keras checkpoint, scikit-learn pipeline, and OOF replay loading | **PASS** |
| 3 | `test_03_input_shape_and_feature_order` | Dimension enforcement (3840 BVP, 640 BVP, 40 EDA, 23 Video features) | **PASS** |
| 4 | `test_04_finite_outputs` | NaN/Inf rejection, probability bounds in [0.0, 1.0], sum-to-one checks | **PASS** |
| 5 | `test_05_class_mappings` | Semantic label consistency across all adapters | **PASS** |
| 6 | `test_06_missing_data_and_insufficient_windows` | Cold-start handling (< K-1 windows) and immediate gap reversion | **PASS** |
| 7 | `test_07_fusion_alignment` | Timestamp, subject, and task coherence checks | **PASS** |
| 8 | `test_08_clas_integrity_enforcement` | Rejection of fabricated predictions on self-supervised encoder | **PASS** |
| 9 | `test_09_full_replay_run` | End-to-end multi-window sequential replay verification | **PASS** |
| 10 | `test_10_preserved_artifacts` | Integrity check ensuring original checkpoints and LOSO files remain intact | **PASS** |

---

## 6. Known Limitations & Next Steps

1. **Task-Derived Labels vs. Clinical Distress:**
   - Labels in UBFC-Phys (Rest vs. Speech/Math) and WESAD (Baseline vs. TSST) represent experimental task conditions. They serve as proxies and should not be equated with clinical psychological diagnoses.
2. **Video Speech Confound:**
   - Facial feature importance is dominated by mouth opening and head movement during public speaking (Task 2). Incorporating rPPG pulse extraction or acoustic speech analysis is recommended to disentangle speaking from distress.
3. **Sensor Generalization & Polarity Inversion:**
   - Cross-dataset transfer between WESAD (wrist BVP) and CLAS (finger PPG) failed due to sensor hardware differences, sampling interfaces, and inverted PPG optical polarity. Unsupervised domain adaptation (e.g., adversarial domain alignment) is required before cross-dataset deployment.
4. **Temporal Trend Prototype Sensitivity:**
   - The trend detector is mechanically sound and causal, but fails synthetic acceptance criteria for subtle gradual ramps. Optimizing the slope threshold $\delta$ and adaptive window length $K$ based on participant baseline noise is recommended.

