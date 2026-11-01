"""Stage 2: window features (physio + video) and synchronisation / quality checks.

python s2_windows.py --root /path/to/UBFC-Phys --ts out/ts --out out/
Writes out/features.csv (one row per 30 s window, stride 10 s) and out/sync_qc.csv (one row per subject-task).

Feature prefixes: p_ physiology, v_ video, q_ quality (never used as model inputs).
No labels are touched here except the protocol task id (T1/T2/T3) carried as metadata.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import (BVP_FS, EDA_FS, STRIDE_S, WIN_S, bvp_window, eda_window, index_files, prepare_bvp,
                    prepare_eda, prepare_video, read_signal, video_window)


def main(root, ts, out, win=WIN_S, stride=STRIDE_S):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    rows, qc = [], []
    for (sid, task), f in sorted(index_files(root).items()):
        npz = Path(ts) / f"s{sid}_{task}.npz"
        if not ("bvp" in f and "eda" in f and npz.exists()):
            print(f"skip s{sid} {task}: missing bvp/eda/video-timeseries")
            continue
        bx, ex = read_signal(f["bvp"]), read_signal(f["eda"])
        if np.isnan(bx).all() or np.isnan(ex).all():
            print(f"skip s{sid} {task}: all-NaN signal")
            continue
        bp, ep, v = prepare_bvp(bx), prepare_eda(ex), prepare_video(npz)
        d_b, d_e, d_v = len(bx) / BVP_FS, len(ex) / EDA_FS, v["dur"]
        D = min(d_b, d_e, d_v)          # common duration, assumes recordings start together
        n = 0
        t0 = 0.0
        while t0 + win <= D + 1e-6:
            r = dict(sid=sid, task=task, win_idx=n, t0=t0, t1=t0 + win)
            r.update(bvp_window(bp, t0, t0 + win))
            r.update(eda_window(ep, t0, t0 + win))
            r.update(video_window(v, t0, t0 + win))
            rows.append(r)
            n += 1
            t0 += stride
        qc.append(dict(sid=sid, task=task, dur_bvp=d_b, dur_eda=d_e, dur_vid=d_v,
                       dur_spread=max(d_b, d_e, d_v) - min(d_b, d_e, d_v), n_windows=n))
    df = pd.DataFrame(rows)
    df["q_hr_absdiff"] = (df.p_hr_mean - df.v_rppg_hr).abs()
    df.to_csv(out / "features.csv", index=False)
    pd.DataFrame(qc).to_csv(out / "sync_qc.csv", index=False)

    print(f"{len(df)} windows, {df.sid.nunique()} subjects")
    ok = df.dropna(subset=["p_hr_mean", "v_rppg_hr"])
    print("\nrPPG (video) vs BVP heart rate agreement per task (window level):")
    for t, d in ok.groupby("task"):
        print(f"  {t}: n={len(d)}  MAE={d.q_hr_absdiff.mean():.1f} bpm  "
              f"median AE={d.q_hr_absdiff.median():.1f}  r={d.p_hr_mean.corr(d.v_rppg_hr):.2f}  "
              f"within 5 bpm={np.mean(d.q_hr_absdiff <= 5):.0%}")
    per_sub = ok.groupby("sid").apply(lambda d: d.q_hr_absdiff.median())
    print("subjects with median |HR diff| > 10 bpm (check sync/video quality):",
          per_sub[per_sub > 10].index.tolist())
    print("\nquality summary: ibi_valid_frac<0.6:", f"{(df.q_ibi_valid_frac < 0.6).mean():.1%}",
          " face_frac<0.8:", f"{(df.q_face_frac < 0.8).mean():.1%}")
    print("duration spread >2 s:", pd.DataFrame(qc).query("dur_spread > 2")[["sid", "task", "dur_spread"]]
          .round(1).values.tolist())
    print("Note: 30 s windows are insensitive to sub-second offsets; waveform cross-correlation is ambiguous "
          "modulo the cardiac period, so only gross offsets (>~10 s) can be detected this way.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--ts", default="out/ts")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    main(a.root, a.ts, a.out)
