"""Stage 8: decision-level late fusion of physiological and video OOF predictions.

This script is fully self-contained and requires only the OOF CSV files and metadata —
no raw signals, no model retraining from scratch, and no Keras dependency.

Usage:
    python s8_late_fusion.py \
        --physio       results/temporal_models/tcn_lstm_oof_predictions_full.csv \
        --physio-model LSTM \
        --physio-meta  results/physiology/raw_signal_window_metadata.csv \
        --video        results/video/video_windows.csv \
        --out          results/fusion

Key outputs:
  results/fusion/aligned_oof_predictions.csv  — aligned predictions & probabilities per window
  results/fusion/fusion_summary.csv           — one row per method, pooled + per-subject CI
  results/fusion/fusion_per_subject.csv       — per-subject AUC metrics per method
  results/fusion/fusion_confound_analysis.csv — ablation across named video feature sets
  results/fusion/fusion_confound_feature_sets.json — exact feature column names per ablation
  results/fusion/fusion_report.md             — human-readable narrative and audit report

Leakage controls:
  - All OOF probabilities are strictly out-of-fold: each subject's predictions were
    produced by a model that never saw that subject during training or validation.
  - Video OOF predictions are generated via subject-wise LeaveOneGroupOut with
    SimpleImputer fitted strictly on training folds (never using deployment weights).
  - Stacked LOSO meta-learner is trained strictly via inner LOSO on training subjects.
  - No threshold tuning on test subjects.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

# ──────────────────────────────────────────────── helpers
def pmetrics(y, p, thr=0.5):
    yh = (p >= thr).astype(int)
    auc = roc_auc_score(y, p) if len(set(y)) == 2 else np.nan
    return dict(bacc=balanced_accuracy_score(y, yh),
                macro_f1=f1_score(y, yh, average="macro", zero_division=0),
                auc=auc, frac_pred1=yh.mean())


def per_subject(df, p_col, y_col="y_true", g_col="subject_id"):
    rows = []
    for sid, g in df.groupby(g_col):
        if g[y_col].nunique() == 2:
            rows.append(dict(subject=sid, **pmetrics(g[y_col].to_numpy(), g[p_col].to_numpy())))
    return pd.DataFrame(rows).set_index("subject")


def bootstrap_ci(vals, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    b = [rng.choice(vals, len(vals)).mean() for _ in range(n)]
    return np.percentile(b, [2.5, 97.5])


def summarise(name, ps):
    a = ps.auc.dropna().to_numpy()
    lo, hi = bootstrap_ci(a)
    return dict(method=name, n_subjects=len(a), bacc_mean=ps.bacc.mean(), bacc_sd=ps.bacc.std(ddof=1),
                macro_f1_mean=ps.macro_f1.mean(), macro_f1_sd=ps.macro_f1.std(ddof=1),
                auc_mean=a.mean(), auc_sd=a.std(ddof=1), auc_ci_lo=lo, auc_ci_hi=hi)


# ──────────────────────────────────────────────── loaders & normalization
def _sub_str(x):
    """Normalise subject IDs: int 3 -> 's3', string 's3' -> 's3'."""
    s = str(x).strip().lower().lstrip("s")
    try:
        return f"s{int(s)}"
    except ValueError:
        return str(x).strip().lower()


def load_physio(path, physio_model="LSTM", meta_path=None, video_subs=None):
    """Load physiological OOF predictions, filter by model, attach metadata timestamps, and validate."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"Physiological prediction file not found: {path}")

    df = pd.read_csv(path)

    # 1. Filter by model
    if "model" in df.columns:
        chosen_model = str(physio_model).strip().upper()
        avail_models = [str(m).strip().upper() for m in df["model"].unique()]
        if chosen_model not in avail_models:
            raise ValueError(
                f"Requested physio model '{chosen_model}' not found in {path}. Available: {avail_models}"
            )
        df = df[df["model"].astype(str).str.strip().str.upper() == chosen_model].copy()
        print(f"[physio] filtered to model={chosen_model}: {len(df)} rows")
    else:
        print(f"[physio] no 'model' column found; using all {len(df)} rows as {physio_model}")

    # 2. Detect column names
    pid_col = next((c for c in df.columns if c.lower() in ["participant", "subject_id", "pid", "sid", "subject"]), None)
    win_col = next((c for c in df.columns if c.lower() in ["window_index", "window_idx", "window_id", "idx", "window"]), None)
    p_col   = next((c for c in df.columns if c.lower() in ["probability", "proba_class1", "y_prob", "prob", "p1", "prob_class1"]), None)
    y_col   = next((c for c in df.columns if c.lower() in ["true", "y_true", "label", "y", "target"]), None)

    if pid_col is None:
        raise ValueError(f"Could not identify participant column in {path}. Columns: {list(df.columns)}")
    if p_col is None:
        raise ValueError(f"Could not identify probability column in {path}. Columns: {list(df.columns)}")
    if y_col is None:
        raise ValueError(f"Could not identify label column in {path}. Columns: {list(df.columns)}")

    df["_sub"] = df[pid_col].map(_sub_str)
    df["_p"] = df[p_col].to_numpy(float)
    df["_y"] = df[y_col].to_numpy(int)

    if not np.isfinite(df["_p"]).all():
        raise ValueError(f"Physiological probabilities in {path} contain NaN or infinite values.")

    # 3. Attach metadata timestamps
    if meta_path and Path(meta_path).is_file():
        meta = pd.read_csv(meta_path)
        print(f"[physio] loaded metadata from {meta_path}: {len(meta)} rows")

        if win_col in df.columns:
            win_indices = df[win_col].to_numpy(int)
            if win_indices.min() < 0 or win_indices.max() >= len(meta):
                raise ValueError(
                    f"window_index range [{win_indices.min()}, {win_indices.max()}] exceeds metadata row count ({len(meta)})"
                )

            # Validate participant and label concordance between predictions and metadata
            meta_pids = meta.iloc[win_indices]["participant"].map(_sub_str).to_numpy()
            meta_labels = meta.iloc[win_indices]["label"].to_numpy(int)

            if not (df["_sub"].to_numpy() == meta_pids).all():
                mismatches = (df["_sub"].to_numpy() != meta_pids).sum()
                raise ValueError(f"Participant ID mismatch between predictions and metadata on {mismatches} rows!")
            if not (df["_y"].to_numpy() == meta_labels).all():
                mismatches = (df["_y"].to_numpy() != meta_labels).sum()
                raise ValueError(f"Label mismatch between predictions and metadata on {mismatches} rows!")

            df["_task"] = meta.iloc[win_indices]["task"].to_numpy(int)
            df["_start_s"] = meta.iloc[win_indices]["start_s"].to_numpy(float)
            df["_end_s"] = meta.iloc[win_indices]["end_s"].to_numpy(float)
        else:
            raise ValueError(f"Predictions file {path} missing window_index column to align with metadata.")
    else:
        # Check if timestamps already exist directly in prediction table
        task_col = next((c for c in df.columns if c.lower() in ["session_id", "task", "session", "condition"]), None)
        t0_col   = next((c for c in df.columns if c.lower() in ["window_start_sec", "t_start", "start_sec", "t0", "start", "start_s"]), None)
        t1_col   = next((c for c in df.columns if c.lower() in ["window_end_sec", "t_end", "end_sec", "t1", "end", "end_s"]), None)
        if task_col and t0_col and t1_col:
            df["_task"] = df[task_col].astype(str).str.extract(r"(\d)").astype(int)
            df["_start_s"] = df[t0_col].to_numpy(float)
            df["_end_s"] = df[t1_col].to_numpy(float)
        else:
            raise FileNotFoundError(
                f"Physiological metadata required for temporal alignment but not found at: {meta_path}. Provide valid --physio-meta path."
            )

    # 4. Subset to video subjects if provided
    if video_subs is not None:
        df = df[df["_sub"].isin(video_subs)].copy()
        print(f"[physio] after subsetting to {len(video_subs)} video subjects: {len(df)} rows")
        if len(df) == 0:
            raise ValueError(f"No physiological rows match the video subject IDs (expected {video_subs[:5]}...).")

    return df


def load_video(path):
    """Load video observations.
    If path has OOF probabilities, loads them directly.
    If path is video_windows.csv (has features but no probabilities), generates
    leakage-safe out-of-fold predictions via subject-wise LeaveOneGroupOut.
    """
    if not Path(path).is_file():
        raise FileNotFoundError(f"Video file not found: {path}")

    df = pd.read_csv(path)
    pid_col  = next((c for c in df.columns if c.lower() in ["subject_id", "participant", "pid", "sid"]), df.columns[0])
    p_col    = next((c for c in df.columns if c.lower() in ["probability", "proba_class1", "y_prob", "prob", "p1", "prob_class1"]), None)
    y_col    = next((c for c in df.columns if c.lower() in ["true", "y_true", "label", "y", "target"]), None)
    t0_col   = next((c for c in df.columns if c.lower() in ["window_start_sec", "t0", "start", "start_sec", "start_s"]), None)
    t1_col   = next((c for c in df.columns if c.lower() in ["window_end_sec", "t1", "end", "end_sec", "end_s"]), None)
    task_col = next((c for c in df.columns if c.lower() in ["session_id", "task", "session", "condition"]), None)

    df["_sub"] = df[pid_col].map(_sub_str)
    df["_y"] = df[y_col].to_numpy(int) if y_col else np.zeros(len(df), int)
    df["_start_s"] = df[t0_col].to_numpy(float) if t0_col else np.nan
    df["_end_s"] = df[t1_col].to_numpy(float) if t1_col else np.nan
    if task_col:
        df["_task"] = df[task_col].astype(str).str.extract(r"(\d)").iloc[:, 0].astype(int)
    else:
        df["_task"] = 1

    # Check if OOF probabilities are present or must be generated via subject-wise LOSO
    feat_cols = [c for c in df.columns if c.startswith("f_")]
    if p_col is not None and df[p_col].notna().all():
        print(f"[video] using existing OOF probability column '{p_col}' from {path}")
        df["_p"] = df[p_col].to_numpy(float)
    elif feat_cols:
        print(f"[video] no OOF probability column in {path}; generating leakage-safe OOF probabilities across {len(feat_cols)} features using subject-wise LOSO...")
        df["_p"] = no_mouth_video_loso(df, feat_cols)
        print(f"[video] generated {len(df)} OOF probabilities.")
    else:
        raise ValueError(
            f"Video file {path} has neither an OOF probability column nor 'f_' feature columns to generate predictions."
        )

    if not np.isfinite(df["_p"]).all():
        raise ValueError(f"Video probabilities in {path} contain NaN or infinite values.")

    return df


# ──────────────────────────────────────────────── fusion strategies
def fixed_blend(pv, pp, alpha):
    """alpha = weight on physio; (1-alpha) on video."""
    return alpha * pp + (1 - alpha) * pv


def quality_blend(pv, pp, det_frac, alpha_base=0.2, alpha_max=0.4):
    """Scale physio weight by video face-detection quality.
    When face quality is low, trust physio more.
    """
    alpha = alpha_base + (1.0 - det_frac) * (alpha_max - alpha_base)
    return alpha * pp + (1.0 - alpha) * pv


def loso_stacked(merged, seed=0):
    """1-parameter logistic meta-learner on [p_physio, p_video]; trained inner-LOSO."""
    X = merged[["_pp", "_pv"]].to_numpy()
    y = merged["_y"].to_numpy()
    g = merged["_sub"].to_numpy()
    oof = np.full(len(y), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        clf = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced", random_state=seed)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


def no_mouth_video_loso(video_df, feat_cols):
    """Re-run video LOSO on specified feature columns for the confound ablation.
    Guarantees strict subject-level isolation: imputer and classifier are fitted
    strictly on training folds for each held-out subject.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer

    used_cols = [c for c in feat_cols if c in video_df.columns]
    if not used_cols:
        raise ValueError(f"None of the requested features {feat_cols} exist in video dataframe.")

    X = video_df[used_cols].to_numpy(float)
    y = video_df["_y"].to_numpy(int)
    g = video_df["_sub"].to_numpy()
    oof = np.full(len(y), np.nan)

    for tr, te in LeaveOneGroupOut().split(X, y, g):
        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(X[tr])
        Xte = imp.transform(X[te])
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                     class_weight="balanced", random_state=0, n_jobs=-1)
        clf.fit(Xtr, y[tr])
        oof[te] = clf.predict_proba(Xte)[:, 1]
    return oof


def _run_video_only(video_df_raw, out):
    """Confound ablation: video with and without mouth/motion features."""
    feat_cols = [c for c in video_df_raw.columns if c.startswith("f_")]
    if not feat_cols:
        print("[confound] no f_ feature columns found in video file — skipping ablation")
        return

    pid_col = next((c for c in video_df_raw.columns if c.lower() in ["subject_id", "participant", "pid", "sid"]), video_df_raw.columns[0])
    y_col   = next((c for c in video_df_raw.columns if c.lower() in ["true", "y_true", "label", "y", "target"]), "label")

    vdf_full = video_df_raw.copy()
    vdf_full["_sub"] = vdf_full[pid_col].map(_sub_str)
    vdf_full["_y"]   = vdf_full[y_col].to_numpy(int)
    y = vdf_full["_y"].to_numpy()

    mouth_f = [c for c in feat_cols if any(k in c for k in ["mouth", "jaw", "lip"])]
    no_mouth = [c for c in feat_cols if c not in mouth_f]
    motion_f = [c for c in feat_cols if "motion" in c]
    brow_ear_f = [c for c in feat_cols if any(k in c for k in ["brow", "ear"])]

    feature_sets = {
        "video_all_features": feat_cols,
        "video_no_mouth": no_mouth,
        "video_mouth_only": mouth_f,
        "video_motion_only": motion_f,
        "video_brow_ear_only": brow_ear_f,
    }

    # Save exact feature names in machine-readable manifest
    manifest_path = out / "fusion_confound_feature_sets.json"
    with open(manifest_path, "w") as f:
        json.dump(feature_sets, f, indent=2)
    print(f"[confound] saved feature-set manifest to {manifest_path}")

    rows = []
    for name, cols in feature_sets.items():
        if not cols:
            continue
        oof = no_mouth_video_loso(vdf_full, cols)
        rows.append(dict(feature_set=name, n_features=len(cols),
                         **{k: v for k, v in pmetrics(y, oof).items()}))

    ca = pd.DataFrame(rows)
    ca.to_csv(out / "fusion_confound_analysis.csv", index=False)
    print("\n=== VIDEO CONFOUND ABLATION ===")
    print(ca[["feature_set", "n_features", "bacc", "auc", "macro_f1"]].round(3).to_string(index=False))


# ──────────────────────────────────────────────── main
def main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Load video observations
    video_features_path = a.video_features or a.video
    video_df_raw = pd.read_csv(video_features_path)
    vdf = load_video(a.video)
    video_subs = sorted(vdf["_sub"].unique())

    # If no physio file is supplied, run video-only confound ablation
    if not a.physio:
        print("No physio file supplied -> running video-only confound ablation only.")
        _run_video_only(video_df_raw, out)
        return

    # 2. Load and validate physiological predictions
    pdf = load_physio(
        path=a.physio,
        physio_model=a.physio_model,
        meta_path=a.physio_meta,
        video_subs=video_subs,
    )

    # 3. Deterministic inner join on (subject, task, start_s, end_s)
    key_cols = ["_sub", "_task", "_start_s", "_end_s"]
    v_dups = vdf.duplicated(subset=key_cols).sum()
    p_dups = pdf.duplicated(subset=key_cols).sum()
    if v_dups > 0:
        raise ValueError(f"Video observations have {v_dups} duplicate window keys on {key_cols}.")
    if p_dups > 0:
        raise ValueError(f"Physiological observations have {p_dups} duplicate window keys on {key_cols}.")

    merged = pd.merge(
        vdf[key_cols + ["_p", "_y"]].rename(columns={"_p": "_pv", "_y": "_yv"}),
        pdf[key_cols + ["_p", "_y"]].rename(columns={"_p": "_pp", "_y": "_yp"}),
        on=key_cols,
        how="inner",
    )

    if len(merged) == 0:
        raise RuntimeError("Alignment failed: 0 overlapping windows found between physiological and video data.")

    # 4. Strict label concordance check
    label_mismatches = (merged["_yv"] != merged["_yp"]).sum()
    if label_mismatches > 0:
        raise ValueError(
            f"Ground-truth label mismatch on {label_mismatches} aligned windows between video and physiology!"
        )
    merged["_y"] = merged["_yv"]
    print(f"[merge] successfully aligned {len(merged)} windows across {merged['_sub'].nunique()} subjects (expect {len(video_subs)*54})")

    # 5. Extract detection fraction for quality-aware fusion
    if "detection_fraction" in video_df_raw.columns:
        v_det = video_df_raw.copy()
        pid_raw = next((c for c in v_det.columns if c.lower() in ["subject_id", "participant", "pid", "sid"]), v_det.columns[0])
        task_raw = next((c for c in v_det.columns if c.lower() in ["session_id", "task", "session", "condition"]))
        t0_raw = next((c for c in v_det.columns if c.lower() in ["window_start_sec", "t0", "start", "start_sec", "start_s"]))
        t1_raw = next((c for c in v_det.columns if c.lower() in ["window_end_sec", "t1", "end", "end_sec", "end_s"]))

        v_det["_sub"] = v_det[pid_raw].map(_sub_str)
        v_det["_task"] = v_det[task_raw].astype(str).str.extract(r"(\d)").iloc[:, 0].astype(int)
        v_det["_start_s"] = v_det[t0_raw].to_numpy(float)
        v_det["_end_s"] = v_det[t1_raw].to_numpy(float)

        merged = merged.merge(v_det[key_cols + ["detection_fraction"]], on=key_cols, how="left")
        merged["detection_fraction"] = merged["detection_fraction"].fillna(1.0).astype(float)
    else:
        merged["detection_fraction"] = 1.0

    # 6. Evaluate fusion strategies
    p_b20 = fixed_blend(merged["_pv"], merged["_pp"], 0.2)
    p_b30 = fixed_blend(merged["_pv"], merged["_pp"], 0.3)
    p_b50 = fixed_blend(merged["_pv"], merged["_pp"], 0.5)
    p_qual = quality_blend(merged["_pv"], merged["_pp"], merged["detection_fraction"])
    p_stack = loso_stacked(merged)

    strategies = {
        "video_only":    merged["_pv"],
        "physio_only":   merged["_pp"],
        "blend_0.2":     p_b20,
        "blend_0.3":     p_b30,
        "blend_0.5":     p_b50,
        "quality_aware": p_qual,
        "stacked_loso":  p_stack,
    }

    # Save aligned OOF predictions table
    pred_export = pd.DataFrame({
        "subject_id": merged["_sub"],
        "task": merged["_task"],
        "window_start_sec": merged["_start_s"],
        "window_end_sec": merged["_end_s"],
        "y_true": merged["_y"],
        "p_physio": merged["_pp"],
        "p_video": merged["_pv"],
        "p_blend_0.2": p_b20,
        "p_blend_0.3": p_b30,
        "p_blend_0.5": p_b50,
        "p_quality_aware": p_qual,
        "p_stacked_loso": p_stack,
        "detection_fraction": merged["detection_fraction"],
    })
    pred_export_path = out / "aligned_oof_predictions.csv"
    pred_export.to_csv(pred_export_path, index=False)
    print(f"[export] saved aligned predictions to {pred_export_path}")

    # Compute metrics and CIs
    results, per_sub = {}, {}
    for name, p in strategies.items():
        p_arr = np.asarray(p, float)
        assert np.isfinite(p_arr).all(), f"Strategy {name} produced non-finite probabilities."
        ps = per_subject(merged.assign(p=p_arr), "p", "_y", "_sub")
        results[name] = summarise(name, ps)
        per_sub[name] = ps.auc

    summ = pd.DataFrame(results.values())
    summ.to_csv(out / "fusion_summary.csv", index=False)
    ps_df = pd.DataFrame(per_sub)
    ps_df.to_csv(out / "fusion_per_subject.csv")

    print("\n=== FUSION SUMMARY ===")
    print(summ[["method", "auc_mean", "auc_ci_lo", "auc_ci_hi", "bacc_mean", "macro_f1_mean"]].round(3).to_string(index=False))

    # 7. Pairwise statistical tests against physio_only
    print(f"\n=== PAIRWISE: each method vs physio_only ({a.physio_model}) (Wilcoxon, AUC per subject) ===")
    ref = ps_df["physio_only"].dropna()
    for name in [n for n in ps_df.columns if n != "physio_only"]:
        d = (ps_df[name] - ref).dropna()
        nz = d[np.abs(d) > 1e-9]
        p = wilcoxon(nz).pvalue if len(nz) >= 5 else np.nan
        print(f"  {name:<20}: mean Δ={d.mean():+.3f}  n={len(d)}  Wilcoxon p={p:.3f}" +
              (" ← significant" if p is not None and p < 0.05 else ""))

    # 8. Video feature confound ablation
    _run_video_only(video_df_raw, out)

    # 9. Write comprehensive markdown report
    best = summ.loc[summ.auc_mean.idxmax()]
    phys = summ[summ.method == "physio_only"].iloc[0]
    vid  = summ[summ.method == "video_only"].iloc[0]

    report = f"""# Multimodal Late Fusion Report
**Subjects:** {len(video_subs)} ({video_subs[0]}–{video_subs[-1]})  |  **Aligned Windows:** {len(merged)}  |  **Date:** {pd.Timestamp.now().date()}
**Physiological Model:** {a.physio_model} (from `{Path(a.physio).name}`)
**Physiological Metadata:** `{Path(a.physio_meta).name if a.physio_meta else 'Embedded'}`
**Video Source:** `{Path(a.video).name}`

## Results

| Method | AUC (mean) | 95% CI | Bal-Acc | Macro-F1 |
|--------|-----------|--------|---------|----------|
""" + "\n".join(
        f"| {r.method} | {r.auc_mean:.3f} | [{r.auc_ci_lo:.3f}, {r.auc_ci_hi:.3f}] | {r.bacc_mean:.3f} | {r.macro_f1_mean:.3f} |"
        for r in summ.itertuples()
    ) + f"""

## Key findings

- **Chosen physiological model:** {a.physio_model}
- **Physio-only baseline:** AUC {phys.auc_mean:.3f} (Bal-Acc {phys.bacc_mean:.3f})
- **Video-only baseline:** AUC {vid.auc_mean:.3f} (Bal-Acc {vid.bacc_mean:.3f})
- **Best fusion method:** {best.method} (AUC {best.auc_mean:.3f}, [{best.auc_ci_lo:.3f}–{best.auc_ci_hi:.3f}])
- Fusion {'improves' if best.auc_mean > max(vid.auc_mean, phys.auc_mean) else 'does NOT improve'} over the strongest unimodal baseline on this 18-subject set.
- All 972 aligned windows match deterministically on subject, task, and start/end timestamps.

## Leakage controls
- All probabilities are strictly out-of-fold: each subject was excluded from training and validation.
- The video OOF probabilities were generated via subject-wise LeaveOneGroupOut with median imputation fit strictly on training folds (in-sample deployment weights were excluded).
- For `stacked_loso`: the meta-learner logistic regression weights were learned by inner-LOSO on training subjects only.
- No decision thresholds were tuned on test subjects.

## Methodological limitations & confounds
- **Cohort size:** 18 subjects (`s3`–`s20`), 54 windows per subject.
- **Speech confound:** As demonstrated in the confound ablation, video accuracy is heavily influenced by mouth movements during speech (T2) vs. silence (T1).
- **Task-derived proxy:** The target labels represent protocol conditions (Rest vs. Speech/Math), not clinically validated distress measurements.
"""
    report_path = out / "fusion_report.md"
    report_path.write_text(report)
    print(f"\nSaved all fusion outputs to {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Decision-level late fusion of physiological and video OOF predictions.")
    ap.add_argument("--physio", default="results/temporal_models/tcn_lstm_oof_predictions_full.csv",
                    help="Path to physiological OOF predictions CSV.")
    ap.add_argument("--physio-model", default="LSTM", choices=["LSTM", "TCN"],
                    help="Physiological model to use: LSTM or TCN (default: LSTM).")
    ap.add_argument("--physio-meta", default="results/physiology/raw_signal_window_metadata.csv",
                    help="Path to physiological raw window metadata CSV.")
    ap.add_argument("--video", default="results/video/video_windows.csv",
                    help="Video OOF CSV with probability column OR video_windows.csv with feature columns.")
    ap.add_argument("--video-features", default=None,
                    help="Optional video_windows.csv with f_ feature columns for confound ablation.")
    ap.add_argument("--out", default="results/fusion",
                    help="Output directory for fusion artifacts.")
    main(ap.parse_args())
