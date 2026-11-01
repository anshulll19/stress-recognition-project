#!/usr/bin/env python3
"""Exploratory analysis report for the existing UBFC-Phys s1/T1 pilot.

Reads the existing feature table and writes new files to a separate output
directory. It does not modify source data, preprocessing code, or model results.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "ubfc_pipeline" / "out" / "features_full" / "features.csv"
OUT = ROOT / "ubfc_pipeline" / "out" / "analysis_report_s1_T1"
OUT.mkdir(parents=True, exist_ok=True)

def savefig(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)

def main():
    if not INPUT.exists():
        raise SystemExit(f"Feature table not found: {INPUT}")
    df = pd.read_csv(INPUT)
    if df.empty:
        raise SystemExit("Feature table is empty.")

    numeric = df.select_dtypes(include=[np.number])
    phys = [c for c in df.columns if c.startswith("p_")]
    video = [c for c in df.columns if c.startswith("v_")]
    quality = [c for c in df.columns if c.startswith("q_")]

    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percent": (100 * df.isna().mean()).round(2)
    })
    missing.to_csv(OUT / "missing_values.csv")
    numeric.describe().T.to_csv(OUT / "numeric_feature_summary.csv")

    const = pd.DataFrame({
        "feature": df.columns,
        "unique_values": [df[c].nunique(dropna=False) for c in df.columns],
        "missing_values": [int(df[c].isna().sum()) for c in df.columns],
    })
    const["constant_in_this_pilot"] = const["unique_values"] <= 1
    const.to_csv(OUT / "feature_constant_check.csv", index=False)

    hr_cols = [c for c in ["p_hr_mean", "v_rppg_hr", "q_hr_absdiff"] if c in df]
    hr_notes = ["Heart-rate comparison unavailable."]
    if {"p_hr_mean", "v_rppg_hr"}.issubset(df.columns):
        keys = [c for c in ["sid", "task", "win_idx", "t0", "t1"] if c in df]
        hr = df[keys + ["p_hr_mean", "v_rppg_hr"]].copy()
        hr["absolute_difference_bpm"] = (hr["p_hr_mean"] - hr["v_rppg_hr"]).abs()
        hr.to_csv(OUT / "heart_rate_comparison.csv", index=False)
        d = hr["absolute_difference_bpm"].dropna()
        paired = hr[["p_hr_mean", "v_rppg_hr"]].dropna()
        corr = paired["p_hr_mean"].corr(paired["v_rppg_hr"]) if len(paired) >= 2 else np.nan
        hr_notes = [
            f"- Paired windows: {len(paired)}",
            f"- Mean absolute difference: {d.mean():.2f} bpm",
            f"- Median absolute difference: {d.median():.2f} bpm",
            f"- Windows within 5 bpm: {(d <= 5).sum()}/{len(d)} ({100*(d <= 5).mean():.1f}%)",
            f"- Pearson correlation across overlapping windows: {corr:.3f}" if pd.notna(corr) else "- Pearson correlation: undefined",
        ]

        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(hr))
        ax.plot(x, hr["p_hr_mean"], marker="o", label="BVP-derived HR")
        ax.plot(x, hr["v_rppg_hr"], marker="o", label="Video-derived rPPG HR")
        ax.set(xlabel="Window index", ylabel="Heart rate (bpm)",
               title="BVP-derived vs video-derived heart rate")
        ax.grid(True, alpha=.3); ax.legend()
        savefig(fig, "01_heart_rate_by_window.png")

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(hr["p_hr_mean"], hr["v_rppg_hr"])
        lo = np.nanmin([hr["p_hr_mean"].min(), hr["v_rppg_hr"].min()])
        hi = np.nanmax([hr["p_hr_mean"].max(), hr["v_rppg_hr"].max()])
        ax.plot([lo, hi], [lo, hi], "--", label="Identity line")
        ax.set(xlabel="BVP-derived HR (bpm)", ylabel="Video-derived rPPG HR (bpm)",
               title="Heart-rate agreement (exploratory)")
        ax.grid(True, alpha=.3); ax.legend()
        savefig(fig, "02_heart_rate_scatter.png")

    selected = [c for c in [
        "p_hr_mean", "p_sdnn", "p_rmssd", "p_scl_mean", "p_eda_std",
        "v_brow_down_mean", "v_eye_squint_mean", "v_mouth_press_mean",
        "v_mouth_smile_mean", "v_head_speed", "v_rppg_hr",
        "q_ibi_valid_frac", "q_face_frac", "q_hr_absdiff"
    ] if c in df.columns]
    if "t0" in df and selected:
        fig, axes = plt.subplots(len(selected), 1, figsize=(10, max(2.0*len(selected), 7)), sharex=True)
        if len(selected) == 1: axes = [axes]
        x = pd.to_numeric(df["t0"], errors="coerce")
        for ax, col in zip(axes, selected):
            ax.plot(x, pd.to_numeric(df[col], errors="coerce"), marker="o", linewidth=1.1)
            ax.set_ylabel(col, rotation=0, labelpad=48); ax.grid(True, alpha=.25)
        axes[-1].set_xlabel("Window start (seconds)")
        fig.suptitle("Selected feature trajectories across overlapping windows", y=1.002)
        savefig(fig, "03_feature_trajectories.png")

    heat_cols = [c for c in phys + video + quality if c in numeric]
    heat = numeric[heat_cols].copy()
    if not heat.empty:
        heat = heat.loc[:, heat.nunique(dropna=True) > 1]
        heat = heat.loc[:, heat.notna().any()]
        if not heat.empty:
            std = heat.std(ddof=0).replace(0, np.nan)
            z = ((heat - heat.mean()) / std).clip(-3, 3).fillna(0)
            fig, ax = plt.subplots(figsize=(max(10, .42*z.shape[1]), 6))
            im = ax.imshow(z.to_numpy().T, aspect="auto", interpolation="nearest", vmin=-3, vmax=3)
            ax.set_yticks(np.arange(z.shape[1])); ax.set_yticklabels(z.columns, fontsize=7)
            ax.set_xticks(np.arange(len(df))); ax.set_xticklabels(range(len(df)), fontsize=8)
            ax.set_xlabel("Window index")
            ax.set_title("Standardized feature patterns (exploratory)")
            fig.colorbar(im, ax=ax, label="Within-pilot z-score, clipped")
            savefig(fig, "04_feature_heatmap.png")

    corr_cols = [c for c in [
        "p_hr_mean", "p_sdnn", "p_rmssd", "p_scl_mean", "p_eda_std",
        "v_rppg_hr", "v_brow_down_mean", "v_eye_squint_mean",
        "v_mouth_press_mean", "v_mouth_smile_mean", "v_head_speed", "q_hr_absdiff"
    ] if c in numeric and numeric[c].nunique(dropna=True) > 1]
    corr = numeric[corr_cols].corr() if len(corr_cols) >= 2 else pd.DataFrame()
    if not corr.empty:
        corr.to_csv(OUT / "exploratory_correlations.csv")
        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(np.arange(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=70, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(corr.index))); ax.set_yticklabels(corr.index, fontsize=8)
        ax.set_title("Exploratory feature correlations (overlapping windows)")
        fig.colorbar(im, ax=ax, label="Pearson correlation")
        savefig(fig, "05_exploratory_correlations.png")

    constant_names = const.loc[const.constant_in_this_pilot, "feature"].tolist()
    subjects = sorted(df["sid"].dropna().unique().tolist()) if "sid" in df else []
    tasks = sorted(df["task"].dropna().astype(str).unique().tolist()) if "task" in df else []
    t0 = pd.to_numeric(df["t0"], errors="coerce").min() if "t0" in df else np.nan
    t1 = pd.to_numeric(df["t1"], errors="coerce").max() if "t1" in df else np.nan
    lines = [
        "# UBFC-Phys s1/T1 Multimodal Pilot — Exploratory Analysis", "",
        "## Scope", "",
        f"- Input: `{INPUT.relative_to(ROOT)}`",
        f"- Table shape: {df.shape[0]} rows × {df.shape[1]} columns",
        f"- Subject IDs: {subjects}; tasks: {tasks}",
        f"- Window time range: {t0:g}–{t1:g} seconds" if pd.notna(t0) and pd.notna(t1) else "- Window time range unavailable",
        f"- Columns with no missing values: {(df.isna().sum() == 0).sum()}/{df.shape[1]}",
        "",
        "This is an exploratory report from one participant and one resting task. Windows overlap, so rows are correlated and are not independent samples. There are no window-level distress labels; no stress classifier was trained or evaluated here.",
        "", "## Heart-rate comparison", "", *hr_notes, "",
        "Video-derived rPPG heart rate is experimental in this pilot. The comparison does not establish that it is interchangeable with BVP.",
        "", "## Constant features", "",
        (", ".join(f"`{c}`" for c in constant_names) if constant_names else "No constant columns found."),
        "",
        "Features are constant only in this pilot; do not globally remove them based on one participant. Zero EDA SCR count/amplitude may reflect detector settings, signal properties, or the resting condition and requires validation.",
        "", "## Interpretation limits", "",
        "- Facial blendshape/action-like values are observable video features, not direct measurements of distress.",
        "- Brightness and sharpness are visual-quality proxies, not distress markers.",
        "- EDA peak counts are not validated until units, filtering, thresholds and candidate responses are checked.",
        "- Correlations and plots are descriptive only; overlap and the one-participant design prevent generalization claims.",
        "- Do not use these windows for supervised stress training without defensible labels and subject-level evaluation.",
        "", "## Output files", "",
        "- `numeric_feature_summary.csv`, `missing_values.csv`, `feature_constant_check.csv`",
        "- `heart_rate_comparison.csv` (if HR columns exist)",
        "- `exploratory_correlations.csv` (if enough varying features exist)",
        "- `01_heart_rate_by_window.png`, `02_heart_rate_scatter.png`",
        "- `03_feature_trajectories.png`, `04_feature_heatmap.png`, `05_exploratory_correlations.png`",
        "",
        "All outputs are written to this new directory. Existing data, source files and model results are not modified.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Input: {INPUT}")
    print(f"Output directory: {OUT}")
    print(f"Table shape: {df.shape}")
    print(f"Constant columns in this pilot: {constant_names}")
    print("\nGenerated files:")
    for p in sorted(OUT.iterdir()):
        print(f"  {p.name} ({p.stat().st_size:,} bytes)")

if __name__ == "__main__":
    main()
