"""Stage 3: which labels can support a valid supervised target?

python s3_label_audit.py --features out/features.csv --info out/subjects_info.csv --out out/

Prints (1) what label-like information exists, (2) whether physiology actually separates the protocol
conditions in the expected direction, and (3) whether the test/control groups differ in that contrast.
This is a *descriptive* check on the label design. Do not tune thresholds or pick features from it.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

EXPECT = {"p_hr_mean": +1, "p_rmssd": -1, "p_scl_mean": +1, "p_scr_n": +1}
PAIRS = [("T1", "T2"), ("T1", "T3"), ("T2", "T3")]


def parse_group(txt):
    t = str(txt).lower()
    if "control" in t:
        return "control"
    if re.search(r"\btest\b", t):
        return "test"
    return np.nan


def main(features, info, out):
    df = pd.read_csv(features)
    med = df.groupby(["sid", "task"])[list(EXPECT)].median().unstack("task")
    rows = []
    for feat, sign in EXPECT.items():
        for a, b in PAIRS:
            d = (med[(feat, b)] - med[(feat, a)]).dropna()
            if len(d) < 5:
                continue
            p = wilcoxon(d).pvalue if (d != 0).any() else np.nan
            rows.append(dict(feature=feat, contrast=f"{b}-{a}", n_subjects=len(d), median_diff=d.median(),
                             frac_expected_direction=float(np.mean(np.sign(d) == sign)), wilcoxon_p=p))
    res = pd.DataFrame(rows)
    print("== Subject-level physiological contrasts (expected: HR up, RMSSD down, SCL up, SCR up under stress) ==")
    print(res.round(3).to_string(index=False))

    if Path(info).exists():
        inf = pd.read_csv(info)
        inf["group"] = inf.info_raw.map(parse_group)
        print("\n== Parsed group from info files (VERIFY against the dataset paper; parsing is a regex guess) ==")
        print(inf.group.value_counts(dropna=False).to_string())
        print("example raw info:", inf.info_raw.iloc[0] if len(inf) else "n/a")
        g = inf.set_index("sid").group
        gr = []
        for feat in EXPECT:
            for a, b in PAIRS[:2]:
                d = (med[(feat, b)] - med[(feat, a)]).dropna()
                x, y = d[d.index.map(g) == "test"], d[d.index.map(g) == "control"]
                if len(x) >= 5 and len(y) >= 5:
                    gr.append(dict(feature=feat, contrast=f"{b}-{a}", n_test=len(x), n_control=len(y),
                                   median_test=x.median(), median_control=y.median(),
                                   mwu_p=mannwhitneyu(x, y).pvalue))
        if gr:
            print("\n== Do test and control groups differ in the task contrast? (group x task check) ==")
            print(pd.DataFrame(gr).round(3).to_string(index=False))
        res.to_csv(Path(out) / "label_audit.csv", index=False)

    print("""
== Interpretation guide ==
 * stress_vs_rest (T1 vs T2+T3): a PROTOCOL-CONDITION label. Defensible if the contrasts above point the
   expected way in most subjects; call it 'task-induced stress condition', not 'stress'.
 * speech_vs_arith (T2 vs T3): both are stressors; differences reflect task demand (speaking vs calculating),
   and mouth/jaw features will separate them via speech, not stress. Use for confound analysis only.
 * Self-report questionnaire scores: usually one value per subject/session, NOT per window. Use them as an
   external validity check of the subject-level score (s5_trend.py --selfreport), not as window labels.
 * Test/control group: an experimental-design factor, not a stress label.
 * Task order is fixed (T1,T2,T3) if the paper says so, so condition is confounded with time-in-session and
   fatigue; there is no recovery segment, so 'recovery' trends cannot be validated on UBFC-Phys.
""")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="out/features.csv")
    ap.add_argument("--info", default="out/subjects_info.csv")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    main(a.features, a.info, a.out)
