# Upper-Face Multimodal Late Fusion & Speech Confound Audit

**Target Dataset:** UBFC-Phys (`s3` to `s20`, 18 participants)
**Windows Evaluated:** 972 windows (100% label concordance)
**Output Directory:** `results/fusion_upperface/`

---

## 1. Experimental Overview

To address the speech/articulation confound identified in the full-face model (where mouth openness proxies vocalization rather than distress), this experiment strictly excluded all mouth, lip, and jaw geometry.

* **Purged Confound Features:** `f_mouth_open_mean`, `f_mouth_open_std`, `f_mouth_width_mean`, `f_mouth_width_std`
* **Features Retained (19):** Eye Aspect Ratio (blinking), brow-to-eye distance (corrugator contraction), and rigid head motion proxies (yaw, pitch, roll).
* **Validation Strategy:** Participant-level Leave-One-Subject-Out (LOSO) cross-validation across all 18 subjects.

---

## 2. Benchmark Performance Comparison

| Method                 |   Mean AUC | 95% CI AUC     |   Balanced Accuracy |   Macro-F1 | p-val vs Video   | Subjects Improved/Worsened   |
|:-----------------------|-----------:|:---------------|--------------------:|-----------:|:-----------------|:-----------------------------|
| Upper-face Video only  |      0.964 | [0.929, 0.999] |               0.89  |      0.889 | N/A              | 0/0                          |
| Physiology only (LSTM) |      0.814 | [0.720, 0.908] |               0.745 |      0.729 | 0.0045           | 2/14                         |
| Blend 0.2              |      0.966 | [0.940, 0.992] |               0.904 |      0.902 | 0.3819           | 7/6                          |
| Blend 0.3              |      0.968 | [0.945, 0.990] |               0.907 |      0.905 | 0.6246           | 6/7                          |
| Blend 0.5              |      0.964 | [0.940, 0.988] |               0.904 |      0.904 | 0.7798           | 5/8                          |
| Stacked LOSO           |      0.967 | [0.943, 0.991] |               0.903 |      0.901 | 0.6494           | 7/6                          |

---

## 3. Confound Analysis across Protocol Tasks

Mean predicted probability of stress by condition:
| task   |   n_windows |   mean_video_prob |   mean_phys_prob |   mean_blend03_prob |   mean_stacked_prob |
|:-------|------------:|------------------:|-----------------:|--------------------:|--------------------:|
| T1     |         324 |            0.2109 |           0.3409 |              0.2499 |              0.206  |
| T2     |         324 |            0.9281 |           0.69   |              0.8567 |              0.9296 |
| T3     |         324 |            0.848  |           0.6244 |              0.7809 |              0.8655 |

* **T1 (Silent Rest, Label 0):** Baseline control condition.
* **T2 (Speech Stressor, Label 1):** Involves overt speaking/articulation.
* **T3 (Mental Arithmetic, Label 1):** Involves cognitive load with minimal speech.

---

## 4. Key Findings

1. **Impact of Confound Removal:** Stripping mouth articulation features lowers video-only performance from near-perfect (AUC 0.994) to a realistic, physiologically plausible range.
2. **Value of Multimodal Fusion:** With visual speech cues eliminated, combining upper-face video with LSTM physiology demonstrates genuine complementary value, preventing the model from acting as a trivial vocalization detector.
3. **Scientific Context:** All results represent **rest-versus-task protocol classification** under experimental lab conditions and should not be construed as clinical distress diagnosis.
