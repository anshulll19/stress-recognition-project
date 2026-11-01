# UBFC-Phys Multimodal Late-Fusion Audit Report
**Date:** 2026-10-27  
**Project:** `stress-recognition-project`  
**Evaluated Artifacts:** `results/fusion_lstm/`  
**Auditor:** Antigravity (Senior ML Experimental Auditor & Research Collaborator)

---

## 1. Executive Summary

This audit evaluates the decision-level late-fusion experiment on the UBFC-Phys dataset combining physiological temporal model predictions (LSTM) with facial video landmark features across 18 participants (`s3`–`s20`).

### Core Audit Verdict:
1. **Mathematical Pipeline Integrity:** The data alignment (972 windows, 18 subjects, exact timestamp matching) and subject-wise Leave-One-Subject-Out (LOSO) partitioning for both video and stacked fusion are mathematically leak-free. No testing subject data contaminated model training or preprocessing.
2. **Ceiling Effect & Nil Fusion Gain:** The apparent superiority of fusion (AUC 0.996, Bal-Acc 0.971) over video-only (AUC 0.994, Bal-Acc 0.965) is **statistically non-significant** ($p = 0.125$ for stacked fusion; $p = 0.313$ for Blend 0.3). For 14 out of 18 participants (77.8%), video alone already achieves an AUC of 1.0000. The nominal +0.0019 AUC gain in stacked fusion is driven entirely by a single subject (`s9`, $\Delta = +0.0216$).
3. **Severe Experimental Confound (Speech Artifact):** Feature ablation and task breakdown demonstrate that video performance is overwhelmingly driven by mouth motion during verbal tasks (T2 Speech and T3 Arithmetic) versus silence during T1 (Rest). A classifier trained on only 4 mouth-opening features achieves **0.988 AUC** (compared to 0.701 AUC for brow/eye affect features). The video model is primarily acting as a speech-activity detector rather than an affective distress sensor.
4. **Paper Readiness:** The headline metrics (**AUC 0.996**) are **NOT safe to report as a breakthrough in distress recognition** without extensive qualification and task-confound controls. Doing so would risk immediate peer-review rejection or retraction once the speech confound is recognized.

---

## 2. Verified Dataset Sizes and Alignment

### 2.1 Population and Cohort
- **Participants:** Exactly 18 participants (`s3`, `s4`, `s5`, `s6`, `s7`, `s8`, `s9`, `s10`, `s11`, `s12`, `s13`, `s14`, `s15`, `s16`, `s17`, `s18`, `s19`, `s20`).
- **Omissions:** `s1` and `s2` are absent from the physiological/video feature sets in this pipeline, consistent with UBFC-Phys preprocessing records.
- **Window Generation:** Windows of length $10.0\text{ s}$ with a stride of $10.0\text{ s}$ (non-overlapping within each task).
- **Per-Subject Windows:** Exactly 54 windows per participant (18 in Task 1, 18 in Task 2, 18 in Task 3). Total cohort windows = $18 \times 54 = 972$.

### 2.2 Join Key Validation
- **Join Keys:** Strict composite tuple `(subject_id, task, window_start_sec, window_end_sec)`.
- **Duplicate Keys:** 0 duplicate tuples detected in `video_windows.csv` ($N=972$). 0 duplicate tuples detected in filtered physiological predictions ($N=972$).
- **Merge Yield:** Exactly 972/972 windows aligned ($100.0\%$).
- **Label Concordance:** Ground-truth binary labels ($y_{\text{video}}$ vs. $y_{\text{physio}}$) match on all 972 rows ($100.0\%$ concordance; 0 mismatches).
- **Task Mapping:**
  - Task 1 (Rest): Class 0 (324 windows; 18 per subject).
  - Task 2 (Speech): Class 1 (324 windows; 18 per subject).
  - Task 3 (Arithmetic): Class 1 (324 windows; 18 per subject).

---

## 3. Leakage and Evaluation Audit

### 3.1 Video Out-of-Fold (OOF) Evaluation
- **Source Implementation:** `ubfc_pipeline/s8_late_fusion.py:load_video()` and `no_mouth_video_loso()`.
- **Fold Partitioning:** Evaluated strictly via `LeaveOneGroupOut` where group is `subject_id`. For each test subject $k$, training is executed exclusively on the remaining 17 subjects.
- **Preprocessing Isolation:** Missing value imputation (`SimpleImputer(strategy='median')`) is initialized, fit on `X[tr]`, and transformed on `X[te]` inside each fold loop. Zero feature statistics leak across participants.
- **Independence from Deployment Baseline:** The script explicitly bypasses `models/video_distress_baseline.joblib` (which was fit on in-sample data) and trains fresh Random Forest models ($N=300$, `min_samples_leaf=3`) per fold.
- **Risk Assessment:** Low risk of mathematical leakage. However, video features were extracted from continuous videos where face detector tracking may smooth landmarks across contiguous frames within a subject's session. Since cross-validation is subject-wise, this does not cross validation boundaries.

### 3.2 Physiological Out-of-Fold (OOF) Predictions
- **Source Artifact:** `results/temporal_models/tcn_lstm_oof_predictions_full.csv`.
- **Model Isolation:** The file contains 11,760 rows combining both TCN and LSTM models across multiple subjects. `s8_late_fusion.py` isolates `--physio-model LSTM` (5,880 rows total; 972 rows for `s3`–`s20`).
- **Metadata Alignment:** Predictions are indexed by `window_index` and verified against `results/physiology/raw_signal_window_metadata.csv`. Concordance was verified for participant IDs, task indices, and binary labels.
- **Fold Independence:** Physiological models were trained under subject-wise LOSO with `RobustScaler` fit strictly on training folds. Early stopping criteria were computed on validation splits drawn exclusively from training subjects.

### 3.3 Stacked Fusion Meta-Learner Isolation
- **Source Implementation:** `ubfc_pipeline/s8_late_fusion.py:loso_stacked()`.
- **Structure:** 2-input Logistic Regression meta-model ($P(\text{Stress} \mid p_{\text{physio}}, p_{\text{video}})$).
- **Leakage Check:** The meta-model is fitted via nested `LeaveOneGroupOut` across the 18 participants:
  $$\text{clf.fit}(X_{\text{tr}}, y_{\text{tr}}) \quad \text{where } X = [p_{\text{physio}}, p_{\text{video}}]$$
  For each subject $k$, the meta-model weights are learned strictly from the other 17 subjects. No test subject data informs the meta-classifier weights.

---

## 4. Fusion Method Audit

### 4.1 Comparative Method Performance
Evaluated across all 18 subjects (54 windows each, 972 total):

| Method | Mean AUC | 95% Bootstrap CI | Student's $t$ 95% CI | Mean Bal-Acc | Mean Macro-F1 |
|---|---:|:---:|:---:|---:|---:|
| Video only | 0.9938 | [0.982, 0.999] | [0.983, 1.000] | 0.9645 | 0.9587 |
| Physio only (LSTM) | 0.8136 | [0.707, 0.898] | [0.712, 0.915] | 0.7454 | 0.7295 |
| Fixed Blend ($\alpha=0.2$) | 0.9955 | [0.986, 1.000] | [0.987, 1.000] | 0.9715 | 0.9655 |
| Fixed Blend ($\alpha=0.3$) | 0.9963 | [0.990, 1.000] | [0.990, 1.000] | 0.9715 | 0.9655 |
| Fixed Blend ($\alpha=0.5$) | 0.9945 | [0.988, 1.000] | [0.989, 1.000] | 0.9668 | 0.9643 |
| Quality-aware Blend | 0.9955 | [0.986, 1.000] | [0.987, 1.000] | 0.9715 | 0.9655 |
| Stacked LOSO | 0.9957 | [0.986, 1.000] | [0.988, 1.000] | 0.9715 | 0.9655 |

*Note: In fixed blends, $\alpha$ represents the physiological prediction weight: $p = \alpha \cdot p_{\text{physio}} + (1-\alpha) \cdot p_{\text{video}}$.*

### 4.2 Quality-Aware Fusion Analysis
The quality-aware fusion algorithm adjusts $\alpha$ based on face detection fraction:
$$\alpha = 0.2 + (1.0 - \text{detection\_fraction}) \times 0.2$$
Across the 972 windows, face detection fraction had a mean of 0.9994 (minimum: 0.9412). Because face tracking in UBFC-Phys lab conditions was nearly flawless, $(1.0 - \text{det\_frac}) \approx 0$, collapsing the quality blend identically to `blend_0.2` on almost every window.

---

## 5. Confound Analysis and Interpretation

### 5.1 Video Feature Ablation
To determine what the video classifier is actually learning, the feature sets were ablated under identical subject-wise LOSO conditions:

| Feature Set | Features ($N$) | Balanced Acc | AUC | Macro-F1 |
|---|---:|---:|---:|---:|
| All features | 23 | 0.965 | 0.995 | 0.958 |
| Without mouth/jaw/lip | 19 | 0.907 | 0.968 | 0.904 |
| **Mouth/jaw/lip only** | **4** | **0.961** | **0.988** | **0.955** |
| Motion features only | 4 | 0.886 | 0.956 | 0.877 |
| **Brow/EAR only (Affect proxy)** | **4** | **0.669** | **0.701** | **0.667** |

### 5.2 The Speech Articulation Mechanism
In the UBFC-Phys protocol:
1. **Task 1 (Rest):** Subjects sit silently and passively.
2. **Task 2 (Speech):** Subjects deliver an unscripted speech test.
3. **Task 3 (Arithmetic):** Subjects verbally report arithmetic calculations out loud under time pressure.

Feature means by session reveal a massive divergence in articulatory geometry:
- `f_mouth_open_mean`: T1 = **0.0123** vs. T2 = **0.1084** (8.8× increase) and T3 = **0.1138** (9.2× increase).
- `f_mouth_width_std`: T1 = **0.0053** vs. T2 = **0.0224** (4.2× increase) and T3 = **0.0211** (4.0× increase).
- In stark contrast, brow furrowing (`f_brow_eye_mean`) changed by less than 6% across conditions (T1: 0.1518, T2: 0.1437, T3: 0.1428).

**Conclusion:** The video model's stellar performance (0.994 AUC) is not evidence that computer vision has solved autonomic stress recognition. It reflects the trivial fact that people open and move their mouths when speaking or counting aloud, but keep them still when resting. The classifier is an **acoustic/articulatory motion detector**, not a distress sensor.

### 5.3 Task Breakdown: Speech vs. Arithmetic
Evaluating predictions on specific task pairs:
- **T1 (Rest) vs. T2 (Speech):**
  - Video AUC: **0.9946**, Bal-Acc: **0.9676**
  - Physio AUC: **0.8097**, Bal-Acc: **0.7639**
  - Stacked AUC: **0.9961**, Bal-Acc: **0.9738**
- **T1 (Rest) vs. T3 (Arithmetic):**
  - Video AUC: **0.9946**, Bal-Acc: **0.9614**
  - Physio AUC: **0.7566**, Bal-Acc: **0.7269**
  - Stacked AUC: **0.9950**, Bal-Acc: **0.9691**

Because both T2 and T3 involved vocalization in this recording setup, video accuracy remains saturated across both tasks.

---

## 6. Per-Participant Robustness and Ceiling Effect

### 6.1 Participant Delta Distribution
Analyzing individual subject improvements of fusion over the unimodal baselines:

| Comparison | Mean $\Delta$ | Improved ($N$) | Degraded ($N$) | Tied ($N$) | Paired Test ($p$-value) |
|---|---:|:---:|:---:|:---:|:---:|
| **Stacked LOSO vs. Video** | **+0.0019** | **4** | **0** | **14** | **$p = 0.125$ (NS)** |
| Blend 0.2 vs. Video | +0.0016 | 5 | 0 | 13 | $p = 0.063$ (NS) |
| Blend 0.3 vs. Video | +0.0025 | 4 | 1 | 13 | $p = 0.313$ (NS) |
| Blend 0.5 vs. Video | +0.0007 | 3 | 4 | 11 | $p = 0.813$ (NS) |
| Stacked LOSO vs. Physio | +0.1821 | 18 | 0 | 0 | $p < 0.001$ (Sig) |

### 6.2 The Ceiling Effect
- For **14 out of 18 subjects** (`s3, s4, s7, s8, s10, s11, s12, s13, s15, s16, s17, s18, s19, s20`), `video_only` achieves an AUC of **exactly 1.0000**.
- In Stacked LOSO, only 4 subjects exhibited non-zero deltas:
  - `s5`: Video = 0.9969 $\rightarrow$ Stacked = 1.0000 ($\Delta = +0.0031$)
  - `s6`: Video = 0.9969 $\rightarrow$ Stacked = 1.0000 ($\Delta = +0.0031$)
  - `s9`: Video = 0.9090 $\rightarrow$ Stacked = 0.9306 ($\Delta = +0.0216$)
  - `s14`: Video = 0.9907 $\rightarrow$ Stacked = 0.9969 ($\Delta = +0.0062$)
- The nominal improvement across the entire 18-subject dataset is carried almost entirely by subject `s9`. With only 4 non-zero differences, statistical significance cannot be established ($n < 5$, exact two-tailed Wilcoxon $p = 0.1250$).

---

## 7. Statistical Testing and Confidence Interval Methodology

### 7.1 Unit of Statistical Independence
- In physiological and video affective computing, adjacent 10-second windows from the same subject are autocorrelated. Computing $p$-values across 972 pooled windows violates the independence assumption (pseudoreplication).
- The correct unit of observation for paired hypothesis testing is the **individual participant** ($N=18$).

### 7.2 Deceptive $p$-Value Reporting in Pipeline
- In `s8_late_fusion.py`, the terminal output and report generated:
  ```text
  stacked_loso: mean Δ=+0.182  n=18  Wilcoxon p=0.000 ← significant
  ```
- **Auditor Flag:** This test evaluated Stacked Fusion against **Physiology Only** ($0.996$ vs. $0.814$). Claiming fusion is "significant" based on this comparison is misleading because it does not test against the strongest baseline (`video_only`, AUC 0.994).
- Testing Stacked Fusion against Video Only yields **$p = 0.125$**, which is **statistically non-significant**.
- Printing `p=0.000` violates scientific reporting standards. Exact precision or bounds ($p < 0.001$) must be reported.

### 7.3 Confidence Intervals for $N=18$
- Percentile bootstrap with 5,000 resamples assumes the empirical sample distribution approximates the population distribution. For $N=18$, heavy-tailed distortions or ceiling truncations can produce overconfident intervals.
- Parametric Student's $t$-intervals ($t_{17} = 2.110$) should be reported alongside bootstrap intervals to provide conservative error bounds.

---

## 8. Confirmed Bugs vs. Potential Risks

### 8.1 Confirmed Bugs Identified & Corrected
1. **Model Filtering Bug in Loader:** `tcn_lstm_oof_predictions_full.csv` contained concatenated predictions for both TCN and LSTM (11,760 rows). The original loader did not filter by model, causing double-counting and mixed-architecture averaging. *(Corrected via `--physio-model LSTM` filter).*
2. **Global Window Index Heuristic Failure:** The original loader attempted to derive task numbers via `window_index // 35 + 1`. Because window indices in `tcn_lstm_oof_predictions_full.csv` are global across the entire UBFC cohort (0 to 5879), this produced impossible task numbers (7 to 60) and 0 aligned rows. *(Corrected by joining directly on `raw_signal_window_metadata.csv` on subject, task, and start/end timestamps).*
3. **Double-Negative Ablation Inversion:** In `_run_video_only()`, feature sets were inverted prior to passing to `no_mouth_video_loso()`, causing labels to misalign with evaluated features. *(Corrected by passing named feature sets directly).*
4. **Zero-P-Value Formatting:** The script output `p=0.000` due to `f"p={p:.3f}"`. *(Corrected in audit reporting).*

### 8.2 Methodological Risks
1. **Speech Confound in Binary Task Labels:** Binary protocol labels ($T_1=0$, $T_2=1, T_3=1$) confound physiological/affective stress with speech production and verbal arithmetic.
2. **Ceiling Distortion:** Video AUC (0.994) leaves virtually no headroom for multimodal synergy. In practical deployment where subjects are silent or speaking in both calm and stress states, video performance will deteriorate.

---

## 9. Paper Readiness & Reporting Safety

### Can the current results be safely reported in a paper?
**NO, not without fundamental reframing.**

If reported as:
> *"Our multimodal late-fusion pipeline achieves state-of-the-art distress recognition with an AUC of 0.996, significantly outperforming physiology alone,"*

The paper is vulnerable to immediate rejection for the following reasons:
1. **Omission of the Video Baseline Test:** Fusion does not statistically outperform video alone ($p = 0.125$). Reviewers will spot that video alone is 0.994 and reject claims of multimodal synergy.
2. **Speech Confound:** A knowledgeable reviewer will note that UBFC-Phys Rest (T1) is silent while T2/T3 involve speaking. The ablation showing mouth features alone reach 0.988 AUC proves the video model is a speech detector.
3. **Label Construct Validity:** The dataset measures task adherence under a laboratory protocol, not generalized psychological distress.

### Safe, Defensible Reporting Angle:
The results **are** publishable if framed as a **critical methodological audit and confound analysis**:
> *"We evaluate multimodal fusion under rigorous subject-wise LOSO and demonstrate that apparent near-perfect video distress classification (AUC 0.994) is an artifact of speech-related facial articulatory dynamics (mouth-only AUC 0.988 vs. affect-only AUC 0.701). When proper baseline comparisons are made, multimodal fusion provides no statistically significant benefit over video alone ($p=0.125$), exposing critical limitations in standard protocol-based affective computing benchmarks."*

---

## 10. Prioritized Next Experiments

1. **Non-Vocal Distress Evaluation:**
   Evaluate the models on an affective dataset where both stress and baseline conditions are speech-free (e.g., WESAD, where stress is induced by cold pressor or non-verbal Trier/mental arithmetic), or where speech occurs in both conditions.
2. **Speech-Residualized Video Features:**
   Train video models strictly on facial features orthogonal to speech (e.g., upper-face Action Units: AU1 Inner Brow Raiser, AU2 Outer Brow Raiser, AU4 Brow Lowerer, AU45 Blinking), discarding all mouth/jaw geometry.
3. **Physiological Signal Synergy:**
   Focus multimodal fusion on combining multiple physiological channels (ECG/PPG, EDA, Respiration, Blood Volume Pulse) where autonomic nervous system arousal is directly measured without articulatory confounds.
4. **Cross-Corpus Generalization:**
   Evaluate the UBFC-trained video and physiological models on an external dataset to test whether the 0.994 AUC collapses when protocol-specific camera angles and vocal tasks differ.
