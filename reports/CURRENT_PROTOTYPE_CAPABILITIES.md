# Current Prototype Capabilities & System Explanation

**Project:** SEC Multimodal Stress Recognition System (Phase 1 Prototype)  
**Target Architecture:** Physiology-First, Video-Assisted Stress Recognition  
**Document Purpose:** Detailed explanation of what the current system does, what inputs it accepts, whether it supports manual medical details, and how it makes predictions.

---

## 1. Quick Answer: Does it support entering medical details?

> **Short Answer: No.**  
> The system does **not** currently have a clinical intake form where a person manually enters medical details (such as age, gender, resting blood pressure numbers, hypertension history, cholesterol, or questionnaire scores like DASS-21 / PSS).

### Why Medical History Forms Are Not Included
1. **Sensor-Driven Affective Computing:** This is an engineering research prototype designed to detect **acute, real-time physiological and behavioral arousal** from continuous hardware sensors (photoplethysmography / BVP, electrodermal activity / EDA, and facial action units from camera video), rather than diagnosing chronic anxiety from demographic or medical records.
2. **Dataset Nature:** The underlying scientific benchmarks (**UBFC-Phys**, **WESAD**, **CLAS**) contain time-series physical recordings during controlled stress tasks (public speaking, mental arithmetic, Trier Social Stress Test), not clinical health record charts.
3. **Clinical Diagnosis vs. Transient Stress:** Static health metrics do not vary second-by-second, whereas this prototype evaluates continuous 10-second to 60-second windows to detect momentary acute distress.

---

## 2. What Does the Current Prototype Actually Do?

The prototype is a **research-grade multimodal stress recognition suite and interactive HUD dashboard**. It processes dynamic physical sensor signals and facial video to classify stress states into:
* **Class 0:** Non-Stress / Rest Baseline
* **Class 1:** Acute Stress (Speech task, Mental arithmetic, TSST stressor)

### The 5 Core Capabilities of the System

```
                         ┌────────────────────────────────────────────────────────┐
                         │              INPUT DATA SOURCES                        │
                         │                                                        │
                         │  [1] Wrist / Finger BVP (Blood Volume Pulse @ 64 Hz)   │
                         │  [2] EDA / GSR (Electrodermal Activity @ 4-64 Hz)     │
                         │  [3] Facial Landmark Trajectories (Video 30 fps)       │
                         └─────────────────────────┬──────────────────────────────┘
                                                   │
                         ┌─────────────────────────┴──────────────────────────────┐
                         │             INFERENCE ENGINES (src/)                   │
                         │                                                        │
                         │  * WESAD Engine (CNN-TCN-LSTM 60s BVP Windows)         │
                         │  * UBFC Calibrated Engine (10s BVP+EDA + Rest Baseline)│
                         │  * Upper-Face Video Engine (19 Clean Landmark Features)│
                         │  * Theil-Sen Trend Engine (Past-Only Temporal Drift)   │
                         └─────────────────────────┬──────────────────────────────┘
                                                   │
                         ┌─────────────────────────┴──────────────────────────────┐
                         │          PHYSIOLOGY-FIRST DYNAMIC FUSION HUD           │
                         │                                                        │
                         │  Normal Operation:      70% Physio  +  30% Upper-Face  │
                         │  Sensor Dropout (σ=0):   0% Physio  + 100% Video Fallback│
                         │  Face Occlusion (<30%): 100% Physio +   0% Video       │
                         └────────────────────────────────────────────────────────┘
```

---

## 3. Detailed Walkthrough of Current Prototype Features

### A. Physiological Inference Module (`src/physiology_inference.py`)
* **Live / Windowed Signal Inference:** Accepts raw time-series arrays or precomputed physiological statistical features.
* **WESAD Deployment Model:** 
  - Loads trained deep learning weights (`models/deployment_candidate/classifier.keras`).
  - Processes 60-second raw wrist BVP windows (3,840 samples @ 64 Hz).
  - Validates zero-variance dropouts and returns softmax probabilities.
* **UBFC-Phys Deployment Model:**
  - Loads deployment Random Forest model (`models/ubfc_deployment/ubfc_phys_calibrated_rf.joblib`).
  - Processes 10-second windows with 12 time-domain features across BVP and EDA.
  - Implements **Rest Calibration (`src/calibration.py`)**: normalizes signals against the individual's silent rest baseline ($z = \frac{x - \mu_{rest}}{\sigma_{rest}}$), boosting balanced accuracy from 70.3% to **98.2%**.
* **Strict Integrity Guard:** Permanently blocks CLAS inference with descriptive warnings, preventing scientifically invalid predictions caused by sensor physics mismatches (optical finger transmission vs. wrist reflection).

### B. Clean Upper-Face Video Analysis (`src/multimodal_fusion.py`)
* **Speech Confound Isolation:** In the UBFC-Phys protocol, stress tasks involve public oral speaking while rest is silent. Full-face models cheat by detecting mouth movement ($AUC = 0.994$).
* **19-Feature Upper-Face Model:** Isolates brow furrows, eye blinks, and gaze coordinates without mouth landmarks ($AUC = 0.968$).
* Shows side-by-side delta shifts in the UI to demonstrate how much speech artificially inflated the prediction.

### C. Multimodal Late Fusion HUD (`app.py` Tab 3)
* **Physiology-First Rule:** Physiological signals receive primary authority ($70\%$). Facial video acts as secondary supporting evidence ($30\%$).
* **Dynamic Quality Gating:**
  - If a pulse sensor falls off or experiences motion flatline ($\sigma = 0$), the HUD immediately activates the **amber pulsating dropout warning** and falls back 100% to video.
  - If the user turns away from the camera or lighting drops below 30% face detection, weight shifts 100% to physiology.

### D. Causal Temporal Trend Detector (`demo/ui/tab_trend.py`)
* Tracks distress trajectories across consecutive windows using causal Theil-Sen robust slope estimation.
* Uses anti-flicker hysteresis thresholds to detect whether stress is **RISING**, **FALLING**, or **STABLE** without looking into future windows.

### E. Research Verification Tab (`app.py` Tab 5)
* Displays independent Leave-One-Subject-Out (LOSO) benchmarks across all 56 UBFC participants and 15 WESAD subjects.
* Separates development early-stopping checkpoints from true cross-subject generalizability.

---

## 4. How Can a New User Currently Test the System?

Currently, there are **two ways** a person or evaluator can use this prototype:

### Method 1: Interactive HUD Replay & Inspection (Streamlit UI)
Run the application locally:
```bash
source ./venv/bin/activate
export MPLCONFIGDIR=/tmp
streamlit run app.py
```
From the browser dashboard, an evaluator can:
1. Select any of the **56 UBFC-Phys participants** or **15 WESAD subjects**.
2. Slide through time-synced signal windows (Task 1: Rest, Task 2: Speech, Task 3: Math).
3. Inspect raw photoplethysmography (BVP) and skin conductance (EDA) waveforms.
4. Toggle between confounded full-face and clean upper-face video models.
5. Inspect how the dynamic multimodal fusion engine calculates stress probabilities.

### Method 2: Programmatic Python Inference on New Signal Windows
A developer can feed a new 60s BVP signal or 10s BVP/EDA signal window directly through Python code:

```python
import numpy as np
from src.physiology_inference import predict_stress

# Example: 60 seconds of raw wrist BVP signal sampled at 64 Hz (3840 float samples)
new_bvp_window = np.random.randn(3840)

# Run model inference
result = predict_stress(
    dataset="wesad",
    input_data=new_bvp_window,
    subject_id="user_test_01"
)

print(f"Prediction: {result.class_label}")
print(f"Stress Probability: {result.probabilities[1]:.1%}")
print(f"Sensor Quality Status: {'OK' if result.signal_quality_ok else 'POOR'}")
```

---

## 5. What Would It Take to Add "Enter Medical Details"?

If the team wants to expand the project to allow users to input clinical/medical details, here is the technical roadmap required:

1. **New UI Intake Panel (Tab 6 or Sidebar Form):**
   - Inputs: Age, Gender, Resting Heart Rate (BPM), Blood Pressure (Systolic/Diastolic), Sleep Quality Score (0–10), Perceived Stress Scale (PSS-10).
2. **Prior-Probability / Bayesian Weighting:**
   - Static health variables cannot replace real-time sensor measurements, but they can set the **prior probability** $P(\text{Stress})$ before observing sensor readings:
     $$P(\text{Stress} \mid \text{Sensors}, \text{Medical}) \propto P(\text{Sensors} \mid \text{Stress}) \times P(\text{Stress} \mid \text{Medical})$$
3. **Required Dataset for Training:**
   - To make this scientifically credible rather than fake, the system would need a dataset that pairs baseline medical health records with acute stress sensor recordings (e.g., MIMIC-III/IV for ICU distress or specialized clinical stress cohorts).

