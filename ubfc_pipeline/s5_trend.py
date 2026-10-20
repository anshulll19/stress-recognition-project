"""Stage 5: temporal trend analysis on OUT-OF-FOLD scores from s4 (never in-sample scores).

python s5_trend.py --scores out/windows_scores_stress_vs_rest.csv --out out/ [--selfreport sr.csv --col anxiety_post]

Per subject (recorded time = T1, T2, T3 concatenated; gaps between tasks are NOT recorded, so slopes are
computed within a task only):
  * causal EMA of the logit score
  * causal Theil-Sen slope over the last H windows (within task) -> rising / stable / falling from the slope CI
  * one-sided CUSUM change-point detector; baseline = first K windows of the session (time-based, label-free)
Evaluation against the protocol: false alarms before the T1->T2 boundary, detection rate and delay after it.
k, h, H, K are fixed a priori; if you tune them, do it with nested subject-wise CV.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, theilslopes, wilcoxon

WIN_S = 30.0


def build_timeline(d):
    d = d.sort_values(["task", "t0"]).copy()
    off, t_rec = 0.0, np.zeros(len(d))
    for task in sorted(d.task.unique()):
        m = (d.task == task).to_numpy()
        t_rec[m] = off + d.t0[m] + WIN_S / 2
        off += d.t0[m].max() + WIN_S
        d.loc[m, "task_start_rec"] = t_rec[m].min() - WIN_S / 2
    d["t_rec"] = t_rec
    return d


def slopes(d, H):
    lab, sl = [], []
    for task, dt in d.groupby("task", sort=True):
        x, y = dt.t_rec.to_numpy(), dt.logit_ema.to_numpy()
        for i in range(len(dt)):
            if i + 1 < H:
                lab.append("n/a"); sl.append(np.nan); continue
            s, _, lo, hi = theilslopes(y[i - H + 1:i + 1], x[i - H + 1:i + 1])
            lab.append("rising" if lo > 0 else "falling" if hi < 0 else "stable"); sl.append(s)
    d["slope"], d["trend"] = sl, lab
    return d


def cusum(x, K, k, h, sd_floor):
    mu, sd = x[:K].mean(), max(x[:K].std(ddof=1), sd_floor)
    S, alarm = 0.0, None
    path = np.zeros(len(x))
    for i in range(K, len(x)):
        S = max(0.0, S + (x[i] - mu) / sd - k)
        path[i] = S
        if alarm is None and S > h:
            alarm = i
    return path, alarm


def main(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.scores)
    subs, ev = [], []
    for sid, d in df.groupby("sid"):
        if not {"T1", "T2"} <= set(d.task):
            continue
        d = build_timeline(d).reset_index(drop=True)
        d["logit_ema"] = d.logit.ewm(alpha=a.alpha, adjust=False).mean()
        d = slopes(d, a.H)
        path, al = cusum(d.logit.to_numpy(), a.K, a.k, a.h, a.sd_floor)
        d["cusum"] = path
        b = d[d.task == "T2"].t_rec.min() - WIN_S / 2     # protocol T1->T2 boundary in recorded time
        t_al = d.t_rec[al] if al is not None else np.nan
        ev.append(dict(sid=sid, boundary=b, alarm_t=t_al,
                       false_alarm=bool(al is not None and t_al < b),
                       detected=bool(al is not None and t_al >= b),
                       delay_s=(t_al - b) if (al is not None and t_al >= b) else np.nan))
        subs.append(d.assign(alarm_t=t_al))
    tl = pd.concat(subs, ignore_index=True)
    tl.to_csv(out / "trend_timeline.csv", index=False)
    ev = pd.DataFrame(ev)
    ev.to_csv(out / "trend_events.csv", index=False)
    print(f"subjects: {len(ev)}")
    print(f"false alarm in rest (T1): {ev.false_alarm.mean():.0%}   detected after onset: {ev.detected.mean():.0%}   "
          f"median delay: {ev.delay_s.median():.0f} s  (IQR {ev.delay_s.quantile(.25):.0f}-{ev.delay_s.quantile(.75):.0f})")
    print("\nshare of windows labelled by slope direction, per task:")
    print(pd.crosstab(tl.task, tl.trend, normalize="index").round(2).to_string())

    tm = tl.groupby(["sid", "task"]).logit.mean().unstack()
    for t in ("T2", "T3"):
        if t in tm:
            dd = (tm[t] - tm["T1"]).dropna()
            print(f"\nmean score {t} - T1: median {dd.median():.2f}, higher in {np.mean(dd > 0):.0%} of subjects, "
                  f"Wilcoxon p={wilcoxon(dd).pvalue:.2g}")
    if a.selfreport:
        sr = pd.read_csv(a.selfreport).set_index("sid")[a.col]
        stress = tm[[c for c in ("T2", "T3") if c in tm]].mean(1) - tm["T1"]
        j = pd.concat([stress.rename("score"), sr], axis=1).dropna()
        r, p = spearmanr(j.score, j[a.col])
        print(f"\nsubject-level validity: Spearman(score increase, {a.col}) = {r:.2f}, p={p:.2g}, n={len(j)}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pick = np.random.default_rng(a.seed).choice(ev.sid.to_numpy(), min(6, len(ev)), replace=False)
        fig, ax = plt.subplots(2, len(pick), figsize=(3.2 * len(pick), 5), sharex="col", squeeze=False)
        for j, sid in enumerate(pick):
            d = tl[tl.sid == sid]
            ax[0, j].plot(d.t_rec, d.logit, alpha=.4, label="logit"); ax[0, j].plot(d.t_rec, d.logit_ema, label="EMA")
            if d.alarm_t.iloc[0] == d.alarm_t.iloc[0]:
                ax[0, j].axvline(d.alarm_t.iloc[0], color="r", ls="--")
            for s in d.task_start_rec.unique():
                ax[0, j].axvline(s, color="k", lw=.5)
            ax[0, j].set_title(f"s{sid}")
            ax[1, j].plot(d.t_rec, d.shap_physio, label="physio"); ax[1, j].plot(d.t_rec, d.shap_video, label="video")
            ax[1, j].axhline(0, color="k", lw=.5); ax[1, j].set_xlabel("recorded time (s)")
        ax[0, 0].legend(fontsize=7); ax[1, 0].legend(fontsize=7)
        ax[0, 0].set_ylabel("score (logit)"); ax[1, 0].set_ylabel("modality contribution (SHAP sum)")
        plt.tight_layout(); plt.savefig(out / "trend_examples.png", dpi=150)
        print("saved trend_examples.png (subjects chosen by random seed, not by appearance)")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="out/windows_scores_stress_vs_rest.csv")
    ap.add_argument("--out", default="out")
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--H", type=int, default=6, help="windows in slope horizon (6 x 10 s stride = 60 s)")
    ap.add_argument("--K", type=int, default=6, help="baseline windows at session start")
    ap.add_argument("--k", type=float, default=0.5)
    ap.add_argument("--h", type=float, default=5.0)
    ap.add_argument("--sd-floor", type=float, default=1.0, help="min baseline SD in logit units")
    ap.add_argument("--selfreport", help="csv with columns sid,<col>")
    ap.add_argument("--col", default="score")
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
