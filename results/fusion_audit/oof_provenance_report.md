# Multimodal Out-of-Fold (OOF) Prediction Provenance Report

**Date:** 2026-11-18  
**Scope:** Data lineage, model checkpoints, alignment mechanisms, and cross-validation procedures for all prediction columns across multimodal fusion artifacts.

---

## 1. Column-by-Column Provenance Map

| Column Name | Modality | Feature Set | Source Script / Checkpoint | Validation Structure | Notes / Limitations |
|---|---|---|---|---|---|
| `p_physio` (`prob_phys_lstm`) | Autonomic Physiology (BVP + EDA) | 12 features (HRV, SCR, SCL, frequency bands) | `results/temporal_models/tcn_lstm_oof_predictions_full.csv` (model: `LSTM`) | 56-participant LOSO | Filtered to `model == 'LSTM'`; indexed via `window_index` into `raw_signal_window_metadata.csv`. |
| `p_video` | Facial Video Landmarks | 23 geometric & motion features (including mouth) | `ubfc_pipeline/s8_late_fusion.py` | 18-participant LOSO (Random Forest, $N=300$) | Confounded by speech articulation ($f_{\text{mouth\_open}}$ AUC = 0.988). |
| `p_video_upperface` | Facial Video Landmarks | 19 upper-face & rigid head motion features | `ubfc_pipeline/run_upperface_fusion.py` | 18-participant LOSO (Random Forest, $N=100$) | Purged mouth opening/width features; speech confound mitigated. |
| `p_blend_0.2` | Multimodal (Full Face) | 23 video + 12 physio | `s8_late_fusion.py:fixed_blend()` | Deterministic rule ($\alpha=0.2$) | $p = 0.2 \cdot p_{\text{physio}} + 0.8 \cdot p_{\text{video}}$. |
| `p_blend_0.3` | Multimodal (Full Face) | 23 video + 12 physio | `s8_late_fusion.py:fixed_blend()` | Deterministic rule ($\alpha=0.3$) | $p = 0.3 \cdot p_{\text{physio}} + 0.7 \cdot p_{\text{video}}$. |
| `p_blend_0.5` | Multimodal (Full Face) | 23 video + 12 physio | `s8_late_fusion.py:fixed_blend()` | Deterministic rule ($\alpha=0.5$) | Equal weighting of unimodal streams. |
| `p_blend_0.3_uf` | Multimodal (Upper Face) | 19 video + 12 physio | `run_upperface_fusion.py` | Deterministic rule ($\alpha=0.3$) | $p = 0.3 \cdot p_{\text{physio}} + 0.7 \cdot p_{\text{video\_upperface}}$. Recommended runtime blend. |
| `p_quality_aware` | Multimodal (Full Face) | 23 video + 12 physio + detection fraction | `s8_late_fusion.py:quality_blend()` | Dynamic rule | $\alpha = 0.2 + (1.0 - \text{detection\_fraction}) \cdot 0.2$. Collapses to `blend_0.2` due to high face detection ($\ge 99\%$). |
| `p_stacked_loso` (original) | Multimodal | Base predictions $[p_{\text{video}}, p_{\text{physio}}]$ | `s8_late_fusion.py:loso_stacked()` | Single-loop outer LOSO (Logistic Regression) | Meta-model trained on outer OOF probabilities of other subjects. |
| `p_stacked_nested_loso` | Multimodal (Upper Face) | Base predictions $[p_{\text{video\_upperface}}, p_{\text{physio}}]$ | `results/fusion_audit/test_fusion_integrity.py` | Two-loop True Nested LOSO (Logistic Regression) | Inner 17-fold LOSO per outer test subject; zero cross-fold leakage path. |
| `physio_decision_score` (V5) | Autonomic Physiology (EDA) | 19 EDA features (SCL, SCR, spectral) | `generate_v5_paired_predictions.py` via `EDASVMV5Inference` | Fixed Deployment Model (`ubfc_phys_eda_svm_v5.joblib`) | Trained on full 56-subject cohort. Subject-specific T1 calibration (first 85s, 12 windows); inference starts at 100s. 30s window / 5s stride. |
| `video_score` (V5 Paired) | Facial Video Landmarks | 19 upper-face features | `generate_video_paired_predictions.py` via `video_distress_upperface.joblib` | **LEAKED IN-SAMPLE** (Non-OOF) | Evaluated using fixed deployment bundle trained on the same cohort. Yields artificial AUC = 1.000. **Do not use for scientific benchmarking.** |

---

## 2. Temporal Alignment Mechanism & Stride Reconciliation

### The Stride Discrepancy Resolved
A superficial inspection of dataset files suggested an apparent mismatch:
- `raw_signal_window_metadata.csv` (Physiology): 5,880 rows across 56 subjects ($\sim 35$ windows per 180s task). Window length = 10.0s, **stride = 5.0s**.
- `video_windows.csv` (Video): 972 rows across 18 subjects (18 windows per 180s task). Window length = 10.0s, **stride = 10.0s** (non-overlapping).

### The Alignment Join
The pipeline resolves this via an inner merge on composite temporal keys:
```python
aligned = video_df.merge(
    phys_joined,
    left_on=["participant_int", "task_num", "window_start_sec", "window_end_sec"],
    right_on=["participant", "task", "start_s", "end_s"],
    how="inner"
)
```
- Because video windows occur at $t_{\text{start}} \in \{0.0, 10.0, 20.0, \dots, 170.0\}$, the inner join acts as a temporal decimation filter, extracting exactly the 18 non-overlapping physiological windows that align strictly with video frames.
- **Yield:** Exactly 18 windows $\times$ 3 tasks $\times$ 18 participants = **972 aligned windows**.
- **Label Concordance:** 100.0% binary concordance (0 mismatches across all 972 windows).

---

## 3. Git Merge Defect in `results/fusion_upperface/upperface_oof_predictions.csv`

During the audit, a major artifact anomaly was discovered in `results/fusion_upperface/upperface_oof_predictions.csv`:
- **Expected row count:** 972 rows.
- **Actual row count:** 1,944 rows (exactly $2 \times 972$).

### Root Cause Analysis
Examination of Git commit logs (`git log -p -2 results/fusion_upperface/upperface_oof_predictions.csv`):
1. **Commit `fca3e5f`:** Added `upperface_oof_predictions.csv` (+973 lines) on branch `main`.
2. **Commit `2532338`:** Added `upperface_oof_predictions.csv` (+973 lines) on branch `sec-dashboard-stage-b`.
3. **Commit `87f0e64`:** Merge commit `Merge branch 'main' into sec-dashboard-stage-b`.
During this merge, Git resolved the conflicting file by concatenating both versions end-to-end rather than reconciling identical rows, duplicating all 18 participants and 972 windows.

### Downstream Impact
- `results/fusion_upperface/upperface_fusion_summary.csv` also inherited duplicated rows (8 rows instead of 6, with duplicate entries for `Upper-face Video only` and `Stacked LOSO`).
- **Resolution:** The canonical, deduplicated evaluation is preserved and validated in `results/fusion_audit/nested_loso_predictions.csv` (972 rows).

---

## 4. Evaluation Protocol Integrity: Single-Loop vs. True Nested LOSO

### Single-Loop Stacking (Pseudo-Nested)
In `s8_late_fusion.py` and `run_upperface_fusion.py`:
- Base models generated out-of-fold predictions across the entire cohort $S$.
- A logistic meta-model was then trained using outer LOSO: for held-out subject $p$, train on $S \setminus \{p\}$.
- **Leakage Assessment:** When training the meta-model on subject $q \in S \setminus \{p\}$, subject $q$'s base video prediction was produced by a base Random Forest trained on $S \setminus \{q\}$, which **included subject $p$**. This creates an indirect information pathway between the outer test subject and the meta-training set.

### True Nested LOSO Verification
To verify whether this indirect pathway caused optimistic performance inflation, we implemented a strict two-loop nested LOSO:
- **Outer Loop:** Hold out subject $p$. Training cohort is $S_{\text{train}} = S \setminus \{p\}$ (17 subjects).
- **Inner Loop:** For each subject $q \in S_{\text{train}}$, train base RF on $S_{\text{train}} \setminus \{q\}$ (16 subjects, zero exposure to $p$). Predict on $q$ to obtain true inner-OOF features.
- **Meta-Learner:** Fit `LogisticRegression` on inner-OOF features of $S_{\text{train}}$.
- **Base Retraining:** Fit final base RF on all 17 subjects of $S_{\text{train}}$. Predict on held-out subject $p$.
- **Outer Prediction:** Pass base predictions of $p$ through meta-learner.

### Empirical Finding
- **Pseudo-nested Stacked AUC (Upper Face):** 0.967
- **True Nested Stacked AUC (Upper Face):** 0.9670 (Pooled: 0.9675)
- **Conclusion:** While methodologically distinct, the empirical difference on this 18-subject dataset is $< 0.001$ AUC. The pseudo-nested protocol did not produce artificial performance inflation, but True Nested LOSO is the methodologically rigorous standard.
