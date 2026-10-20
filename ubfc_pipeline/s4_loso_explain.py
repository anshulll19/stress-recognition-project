"""Stage 4: leave-one-subject-out evaluation + modality and feature attribution. CPU-light (sklearn).

python s4_loso_explain.py --features out/features.csv --target stress_vs_rest --norm subject_all --out out/

Leakage controls
  * LeaveOneGroupOut by subject; the test subject never touches fitting, scaling, imputation or tuning.
  * Hyper-parameters chosen by GridSearchCV with GroupKFold *inside the training subjects only*.
  * Window QC filtering uses only q_ columns (no labels) and is applied once so that physio-only,
    video-only and fused models are compared on identical windows.
  * --norm subject_all : per-subject z-score over all of that subject's windows (label-free, but uses the
                         test subject's unlabeled data, i.e. transductive).
    --norm baseline_T1 : z-score against the subject's T1 (rest) windows (needs a labelled rest calibration).
    --norm none        : raw features (hardest setting).

Attribution
  * Modality Shapley (exact, 2 players) from the three LOSO models: v(S) = mean per-subject AUROC, v({})=0.5.
  * Feature SHAP for the fused linear model. For a linear model, interventional SHAP is exactly
    w_j * (x_j - mean_train_j) in standardised space, so no shap package is needed; logit = intercept + sum(phi).
    Swap in shap.Explainer for non-linear models (keep the same per-fold, train-only background).
Why logistic regression, not Keras: ~40 tabular features, ~50 subjects. Use Keras later for raw-signal encoders.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGETS = {"stress_vs_rest": (["T1"], ["T2", "T3"]), "speech_vs_arith": (["T2"], ["T3"]),
           "rest_vs_speech": (["T1"], ["T2"]), "rest_vs_arith": (["T1"], ["T3"])}


def normalise(df, cols, mode):
    if mode == "none":
        return df
    out = df.copy()
    for sid, d in df.groupby("sid"):
        ref = d if mode == "subject_all" else d[d.task == "T1"]
        mu, sd = ref[cols].mean(), ref[cols].std().replace(0, 1.0)   # constant feature -> 0
        out.loc[d.index, cols] = (d[cols] - mu) / sd
    return out


def loso(X, y, g, grid):
    oof = np.full(len(y), np.nan)
    models = {}
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        pipe = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                             LogisticRegression(class_weight="balanced", max_iter=2000))
        gs = GridSearchCV(pipe, {"logisticregression__C": grid}, cv=GroupKFold(5), scoring="roc_auc")
        gs.fit(X.iloc[tr], y[tr], groups=g[tr])
        oof[te] = gs.predict_proba(X.iloc[te])[:, 1]
        models[g[te][0]] = (gs.best_estimator_, tr, te)
    return oof, models


def per_subject(y, p, g):
    rows = []
    for s in np.unique(g):
        m = g == s
        auc = roc_auc_score(y[m], p[m]) if len(set(y[m])) == 2 else np.nan
        rows.append(dict(sid=s, auc=auc, bal_acc=balanced_accuracy_score(y[m], p[m] >= 0.5)))
    return pd.DataFrame(rows).set_index("sid")


def summarise(name, ps, rng, n_boot=1000):
    a = ps.auc.dropna().to_numpy()
    boots = [rng.choice(a, len(a)).mean() for _ in range(n_boot)]
    return dict(model=name, n_subjects=len(a), auc_mean=a.mean(), auc_sd=a.std(ddof=1),
                auc_ci_lo=np.percentile(boots, 2.5), auc_ci_hi=np.percentile(boots, 97.5),
                bal_acc_mean=ps.bal_acc.mean(), frac_subjects_auc_gt_0_5=float(np.mean(a > 0.5)))


def modality_shapley(ap, av, apv, rng, n_boot=1000):
    def phi(i):
        vp, vv, vpv = ap[i].mean(), av[i].mean(), apv[i].mean()
        return 0.5 * ((vp - .5) + (vpv - vv)), 0.5 * ((vv - .5) + (vpv - vp))
    n = len(ap)
    pt = phi(np.arange(n))
    bs = np.array([phi(rng.integers(0, n, n)) for _ in range(n_boot)])
    return pd.DataFrame({"modality": ["physio", "video"], "shapley_auc": pt,
                         "ci_lo": np.percentile(bs, 2.5, axis=0), "ci_hi": np.percentile(bs, 97.5, axis=0)})


def main(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    df = pd.read_csv(a.features)
    neg, pos = TARGETS[a.target]
    df = df[df.task.isin(neg + pos)].copy()
    n0 = len(df)
    df = df[(df.q_ibi_valid_frac >= a.min_ibi) & (df.q_face_frac >= a.min_face)].copy()
    drop = tuple(a.drop or [])
    phys = [c for c in df if c.startswith("p_") and not any(k in c for k in drop)]
    vid = [c for c in df if c.startswith("v_") and not any(k in c for k in drop)]
    allc = phys + vid
    allc = [c for c in allc if df[c].notna().any()]
    phys, vid = [c for c in phys if c in allc], [c for c in vid if c in allc]
    df["y"] = df.task.isin(pos).astype(int)
    cnt = df.groupby("sid").y.agg(["sum", "count"])
    keep = cnt[(cnt["sum"] >= 3) & (cnt["count"] - cnt["sum"] >= 3)].index
    df = df[df.sid.isin(keep)]
    print(f"windows: {n0} -> {len(df)} after QC; subjects kept: {len(keep)}; "
          f"features: {len(phys)} physio + {len(vid)} video; dropped patterns: {drop}")
    df = normalise(df, allc, a.norm).reset_index(drop=True)
    allc = [c for c in allc if df[c].notna().any()]          # re-check after normalisation
    phys, vid = [c for c in phys if c in allc], [c for c in vid if c in allc]
    y, g = df.y.to_numpy(), df.sid.to_numpy()
    grid = [0.01, 0.1, 1.0]

    res, oofs, ps = [], {}, {}
    for name, cols in [("physio", phys), ("video", vid), ("fused", phys + vid)]:
        oofs[name], models = loso(df[cols], y, g, grid)
        if name == "fused":
            fused_models, fused_cols = models, cols
        ps[name] = per_subject(y, oofs[name], g)
        res.append(summarise(name, ps[name], rng))
    summ = pd.DataFrame(res)
    summ.to_csv(out / f"loso_summary_{a.target}.csv", index=False)
    print("\n== LOSO (per-subject AUROC; CI = bootstrap over subjects) ==")
    print(summ.round(3).to_string(index=False))

    common = ps["physio"].auc.dropna().index.intersection(ps["video"].auc.dropna().index) \
        .intersection(ps["fused"].auc.dropna().index)
    sh = modality_shapley(ps["physio"].auc[common].to_numpy(), ps["video"].auc[common].to_numpy(),
                          ps["fused"].auc[common].to_numpy(), rng)
    sh.to_csv(out / f"modality_shapley_{a.target}.csv", index=False)
    print("\n== Modality Shapley (AUROC points above chance, split between modalities) ==")
    print(sh.round(3).to_string(index=False))

    # ---- feature SHAP (analytic, linear model), computed on the held-out subject only
    nphys = len(phys)
    phi_all = np.zeros((len(df), len(fused_cols)))
    base = np.zeros(len(df))
    fold_imp = []
    for sid, (est, tr, te) in fused_models.items():
        imp, sc, clf = (est.named_steps[k] for k in ("simpleimputer", "standardscaler", "logisticregression"))
        Ztr = sc.transform(imp.transform(df.iloc[tr][fused_cols]))
        Zte = sc.transform(imp.transform(df.iloc[te][fused_cols]))
        phi = (Zte - Ztr.mean(0)) * clf.coef_[0]
        phi_all[te], base[te] = phi, clf.intercept_[0]
        fold_imp.append(np.abs(phi).mean(0))
    fold_imp = np.array(fold_imp)
    imp_df = pd.DataFrame({"feature": fused_cols, "block": ["physio"] * nphys + ["video"] * (len(fused_cols) - nphys),
                           "mean_abs_shap": fold_imp.mean(0), "sd_across_subjects": fold_imp.std(0, ddof=1)})
    mean_imp = fold_imp.mean(0)
    rho = [spearmanr(f, mean_imp)[0] for f in fold_imp]
    imp_df = imp_df.sort_values("mean_abs_shap", ascending=False)
    imp_df.to_csv(out / f"shap_features_{a.target}.csv", index=False)
    print("\n== Top features by mean |SHAP| (fused model, held-out subjects) ==")
    print(imp_df.head(12).round(3).to_string(index=False))
    print(f"fold-to-mean rank stability (Spearman): median {np.median(rho):.2f}, min {np.min(rho):.2f}")
    blk = imp_df.groupby("block").mean_abs_shap.sum()
    print("summed mean |SHAP| by block:", (blk / blk.sum()).round(3).to_dict(),
          "(correlated features share credit; read with the Shapley table, not alone)")

    w = df[["sid", "task", "win_idx", "t0", "y"]].copy()
    w["oof_prob"], w["base"] = oofs["fused"], base
    w["shap_physio"], w["shap_video"] = phi_all[:, :nphys].sum(1), phi_all[:, nphys:].sum(1)
    w["logit"] = w.base + w.shap_physio + w.shap_video
    w.to_csv(out / f"windows_scores_{a.target}.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        top = imp_df.head(15)[::-1]
        plt.figure(figsize=(6, 5))
        plt.barh(top.feature, top.mean_abs_shap, color=["C0" if b == "physio" else "C1" for b in top.block])
        plt.xlabel("mean |SHAP| (logit units)"); plt.title(f"{a.target}: blue=physio, orange=video")
        plt.tight_layout(); plt.savefig(out / f"shap_top_{a.target}.png", dpi=150)
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="out/features.csv")
    ap.add_argument("--target", choices=list(TARGETS), default="stress_vs_rest")
    ap.add_argument("--norm", choices=["none", "subject_all", "baseline_T1"], default="subject_all")
    ap.add_argument("--drop", nargs="*", help="substrings of feature names to exclude, e.g. v_mouth v_jaw v_rppg")
    ap.add_argument("--min-ibi", type=float, default=0.6)
    ap.add_argument("--min-face", type=float, default=0.8)
    ap.add_argument("--out", default="out")
    main(ap.parse_args())
