"""
plot_learning_curve.py
=======================

Turns learning curve results into publication-grade figures, per-subject
aggregations, and paired statistical tests with Holm-Bonferroni multiple-testing correction.

Statistical Methodology:
    1. SUBJECT AS EXPERIMENTAL UNIT: Each subject's metrics are first averaged
       across random subsample seeds before computing cross-subject means and 95% CIs.
    2. PAIRED COMPARISONS: Pretrained vs. Scratch is paired by subject (LOSO fold)
       using both paired Student's t-test and Wilcoxon signed-rank test.
    3. MULTIPLE COMPARISON CORRECTION: Holm-Bonferroni step-down correction is applied
       across all planned pairwise hypotheses to control Family-Wise Error Rate (FWER).
    4. REPRODUCIBLE MANIFEST: Outputs structured JSON summary and per-subject CSVs.
"""

import argparse
import datetime
import json
import os
import subprocess

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

CONDITIONS = ["scratch", "frozen", "finetune"]
LABELS = {
    "scratch": "Scratch (no pretraining)",
    "frozen": "Pretrained, frozen encoder",
    "finetune": "Pretrained, fine-tuned"
}
COLORS = {"scratch": "#6b7280", "frozen": "#2563eb", "finetune": "#dc2626"}


def get_git_commit() -> str:
    """Returns the current Git commit hash if available, otherwise 'unknown'."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return commit
    except Exception:
        return "unknown"


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """
    Computes Holm-Bonferroni step-down adjusted p-values.
    Controls Family-Wise Error Rate (FWER) under arbitrary dependence.
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    inv_order = np.empty(n, dtype=int)
    inv_order[order] = np.arange(n)
    sorted_p = p[order]

    adjusted = np.empty(n, dtype=float)
    cum_max = 0.0
    for j in range(n):
        val = (n - j) * sorted_p[j]
        cum_max = max(cum_max, val)
        adjusted[j] = min(1.0, cum_max)

    return adjusted[inv_order]


def per_subject_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Average over seeds within each (subject, fraction, condition)."""
    return (df.groupby(["held_out_subject", "label_fraction", "condition"])[metric]
              .mean().reset_index())


def summarize_across_subjects(per_subj: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Computes mean and 95% t-distribution CI across subject means."""
    rows = []
    for (frac, cond), g in per_subj.groupby(["label_fraction", "condition"]):
        vals = g[metric].dropna().values
        n = len(vals)
        mean = vals.mean()
        if n > 1:
            sem = vals.std(ddof=1) / np.sqrt(n)
            ci = stats.t.ppf(0.975, n - 1) * sem
        else:
            ci = np.nan
        rows.append({
            "label_fraction": frac,
            "condition": cond,
            "n_subjects": n,
            "mean": mean,
            "ci95": ci
        })
    return pd.DataFrame(rows).sort_values(["label_fraction", "condition"])


def paired_tests(per_subj: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Pretrained vs scratch paired by subject at each label fraction.
    Applies Holm-Bonferroni correction across all planned pairwise tests.
    """
    rows = []
    for frac in sorted(per_subj["label_fraction"].unique()):
        sub = per_subj[per_subj["label_fraction"] == frac]
        wide = sub.pivot(index="held_out_subject", columns="condition", values=metric).dropna()
        for cond in ["frozen", "finetune"]:
            if cond not in wide.columns or "scratch" not in wide.columns or len(wide) < 3:
                continue
            diff = wide[cond] - wide["scratch"]
            t_stat, p_t = stats.ttest_rel(wide[cond], wide["scratch"])
            try:
                _, p_w = stats.wilcoxon(wide[cond], wide["scratch"])
            except ValueError:
                p_w = np.nan
            rows.append({
                "label_fraction": frac,
                "comparison": f"{cond} - scratch",
                "n_subjects": len(wide),
                "mean_diff": float(diff.mean()),
                "subjects_improved": int((diff > 0).sum()),
                "p_ttest_raw": float(p_t),
                "p_wilcoxon_raw": float(p_w)
            })

    if not rows:
        return pd.DataFrame()

    df_res = pd.DataFrame(rows)
    df_res["p_ttest_holm"] = holm_adjust(df_res["p_ttest_raw"].values)

    valid_w = ~df_res["p_wilcoxon_raw"].isna()
    df_res["p_wilcoxon_holm"] = np.nan
    if valid_w.any():
        df_res.loc[valid_w, "p_wilcoxon_holm"] = holm_adjust(df_res.loc[valid_w, "p_wilcoxon_raw"].values)

    return df_res


def plot(df: pd.DataFrame, out_path: str):
    metrics = [("auc", "AUC"), ("class1_f1", "Stress-class F1")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, (metric, title) in zip(axes, metrics):
        per_subj = per_subject_table(df, metric)
        summ = summarize_across_subjects(per_subj, metric)
        for cond in CONDITIONS:
            s = summ[summ["condition"] == cond]
            if s.empty:
                continue
            x = s["label_fraction"].values * 100
            ax.plot(x, s["mean"], marker="o", color=COLORS[cond], label=LABELS[cond])
            ax.fill_between(x, s["mean"] - s["ci95"], s["mean"] + s["ci95"],
                            color=COLORS[cond], alpha=0.15)
        ax.set_xlabel("Labeled training data used (%)")
        ax.set_ylabel(title)
        ax.set_title(f"WESAD LOSO: {title} vs. label fraction")
        ax.set_xticks(sorted(df["label_fraction"].unique() * 100))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    n_subj = df["held_out_subject"].nunique()
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.text(0.5, 0.015, f"Mean across {n_subj} held-out subjects (seeds averaged within subject); shaded = 95% CI",
             ha="center", fontsize=9)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Saved figure to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Statistical analysis and plotting for learning curve experiment")
    parser.add_argument("--csv", default="learning_curve_rigorous_results.csv")
    parser.add_argument("--out", default="learning_curve_rigorous.png")
    parser.add_argument("--per_subject_csv", default=None,
                        help="Optional path to save per-subject aggregated metrics")
    parser.add_argument("--manifest", default=None,
                        help="Optional path to save JSON statistical manifest")
    args = parser.parse_args()

    # Fallback if rigorous csv does not exist yet (e.g. running on historical data)
    csv_to_load = args.csv
    if not os.path.exists(csv_to_load) and os.path.exists("learning_curve_results.csv"):
        print(f"File '{csv_to_load}' not found. Falling back to historical 'learning_curve_results.csv'.")
        csv_to_load = "learning_curve_results.csv"

    df = pd.read_csv(csv_to_load)
    print(f"Loaded {len(df)} runs from {csv_to_load} "
          f"({df['held_out_subject'].nunique()} subjects, "
          f"{df['label_fraction'].nunique()} fractions, "
          f"{df['seed'].nunique()} seeds, {df['condition'].nunique()} conditions)")

    pd.set_option("display.width", 160)
    stat_manifest = {
        "timestamp": datetime.datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "source_csv": csv_to_load,
        "n_subjects": int(df["held_out_subject"].nunique()),
        "metrics": {}
    }

    per_subj_all = []
    for metric in ["auc", "class1_f1"]:
        per_subj = per_subject_table(df, metric)
        per_subj_all.append(per_subj)
        summ = summarize_across_subjects(per_subj, metric)
        tests = paired_tests(per_subj, metric)

        print(f"\n=== {metric}: mean +/- 95% CI across subjects ===")
        print(summ.round(4).to_string(index=False))
        print(f"\n=== {metric}: paired comparison vs scratch (with Holm correction) ===")
        print(tests.round(4).to_string(index=False))

        stat_manifest["metrics"][metric] = {
            "summary": summ.to_dict(orient="records"),
            "paired_tests": tests.to_dict(orient="records")
        }

    if args.per_subject_csv:
        merged = per_subj_all[0]
        for p in per_subj_all[1:]:
            merged = merged.merge(p, on=["held_out_subject", "label_fraction", "condition"])
        os.makedirs(os.path.dirname(args.per_subject_csv) or ".", exist_ok=True)
        merged.to_csv(args.per_subject_csv, index=False)
        print(f"Saved per-subject table to: {args.per_subject_csv}")

    if args.manifest:
        os.makedirs(os.path.dirname(args.manifest) or ".", exist_ok=True)
        with open(args.manifest, "w", encoding="utf-8") as f:
            json.dump(stat_manifest, f, indent=2)
        print(f"Saved statistical manifest to: {args.manifest}")

    plot(df, args.out)


if __name__ == "__main__":
    main()
