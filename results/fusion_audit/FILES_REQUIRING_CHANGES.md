# Codebase Audit: Files Requiring Changes

**Audit Date:** 2026-11-18  
**Scope:** Remediation list for artifacts, pipeline scripts, and deployment configurations based on the comprehensive multimodal fusion audit.

---

## 1. Prioritized File Remediation Matrix

| File Path | Priority | Category | Issue Description | Recommended Action |
|---|:---:|---|---|---|
| `results/fusion_upperface/upperface_oof_predictions.csv` | **HIGH** | Data Defect | File contains 1,944 rows (972 duplicate rows) created during Git merge `87f0e64`. | Deduplicate on composite keys `['subject_id', 'session_id', 'window_start_sec', 'window_end_sec']` to restore 972 canonical rows. |
| `results/fusion_upperface/upperface_fusion_summary.csv` | **HIGH** | Benchmark Defect | Summary table contains 8 rows with duplicate entries for `Upper-face Video only` and `Stacked LOSO`. | Recompute from deduplicated predictions to output clean 6-row summary table. |
| `generate_video_paired_predictions.py` / `results/fusion/video_paired_predictions_s3_s20.csv` | **CRITICAL** | Data Leakage | Script scores all 18 participants using the deployment artifact `video_distress_upperface.joblib` (trained on those same subjects), causing train-on-test leakage (AUC = 1.000). | Replace deployment bundle scoring with held-out LOSO OOF predictions from `results/fusion_upperface/` (deduplicated to 972 rows) before temporal alignment. |
| `results/fusion/paired_fusion_dataset_s3_s20.csv` | **CRITICAL** | Benchmark Invalidation | Dataset contains leaked `video_score` values; invalid for research benchmarking or publication. | Mark as invalid for cross-validation benchmarking; regenerate using strictly out-of-fold predictions. |
| `ubfc_pipeline/run_upperface_fusion.py` | **MEDIUM** | Methodology | Single-loop stacking creates theoretical cross-fold dependency; does not guard against appending to existing CSV. | Update meta-learner to True Nested LOSO; ensure atomic file overwrite mode (`index=False`, overwrite existing). |
| `ubfc_pipeline/s8_late_fusion.py` | **MEDIUM** | Reporting | Script reports `Wilcoxon p=0.000` against `physio_only` rather than the stronger `video_only` baseline. | Update paired statistical testing to benchmark against `video_only`; replace `p=0.000` with standard scientific formatting (`p < 0.001`). |
| `src/multimodal_fusion.py` | **LOW** | Deployment | Verification that runtime inference strictly defaults to `video_feature_set='upper_face_19'`. | Confirmed: `UpperFaceVideoEngine` strictly defaults to `use_full_features=False` (19 features) and loads `video_upperface_features.json`. Maintain regression guard. |

---

## 2. Detailed Remediation Instructions

### 2.1 Deduplicating `results/fusion_upperface/upperface_oof_predictions.csv`
- **Root Cause:** Git merge conflict resolution in commit `87f0e64` concatenated two versions of the file.
- **Remediation Script (Do NOT execute until approved by user):**
  ```python
  import pandas as pd
  df = pd.read_csv("results/fusion_upperface/upperface_oof_predictions.csv")
  df_clean = df.drop_duplicates(subset=["subject_id", "session_id", "window_start_sec", "window_end_sec"])
  assert len(df_clean) == 972
  df_clean.to_csv("results/fusion_upperface/upperface_oof_predictions.csv", index=False)
  ```

### 2.2 Updating Statistical Comparisons in `ubfc_pipeline/s8_late_fusion.py`
- **Location:** Line 449 (`pairwise statistical tests`).
- **Current Behavior:**
  ```python
  ref = ps_df["physio_only"].dropna()
  ```
- **Recommended Fix:**
  ```python
  # Benchmark against the strongest unimodal baseline
  ref = ps_df["video_only"].dropna()
  for name in [n for n in ps_df.columns if n != "video_only"]:
      d = (ps_df[name] - ref).dropna()
      nz = d[np.abs(d) > 1e-9]
      p = wilcoxon(nz).pvalue if len(nz) >= 5 else np.nan
      p_formatted = "< 0.001" if p < 0.001 else f"= {p:.3f}"
      print(f"  {name:<20}: mean Δ={d.mean():+.3f}  n={len(d)}  Wilcoxon p {p_formatted}")
  ```

### 2.3 Preserving Canonical Audit Artifacts
- The clean, audited datasets and metrics are already preserved independently in `results/fusion_audit/`:
  - `nested_loso_predictions.csv` (972 canonical rows)
  - `fusion_metrics_reproduced.csv` (reproduced metrics)
  - `calibration_metrics.csv` (ECE and Brier scores)
  - `participant_wise_metrics.csv` (per-subject metrics)

### 2.4 Remediating the V5 Paired Fusion Dataset (`results/fusion/`)
- **Root Cause:** `generate_video_paired_predictions.py` scored all 18 participants using the deployment artifact `models/video_distress_upperface.joblib` instead of out-of-fold models. Because the deployment model was trained on the cohort, video predictions exhibit 100% classification accuracy ($AUC = 1.000$) due to direct train-on-test leakage.
- **Impact:** `paired_fusion_dataset_s3_s20.csv` cannot be used to benchmark multimodal fusion performance or compare against physiological models.
- **Recommended Remediation:**
  1. Do not use `paired_fusion_dataset_s3_s20.csv` in any research manuscript or benchmarking table.
  2. To create a valid paired V5-EDA + Upper-Face Video dataset, substitute the `video_score` column with the deduplicated out-of-fold `prob_video_upperface` predictions from `results/fusion_audit/nested_loso_predictions.csv`.
  3. Note that the V5 EDA SVM pipeline (`evaluate_v5_scr_ablation_loso.py`) already has genuine 56-subject LOSO predictions stored in `results/ubfc_deployment/v5_scr_ablation_loso/predictions.csv`, which are leakage-free.
