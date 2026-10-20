"""Stage 0: inventory + integrity checks. Reads only file headers / CSVs (no video decoding).

python s0_inventory.py --root /path/to/UBFC-Phys --out out/
Writes out/inventory.csv and out/subjects_info.csv and prints what to look at.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import BVP_FS, EDA_FS, _PAT, index_files, read_signal


def video_props(p):
    import cv2
    cap = cv2.VideoCapture(str(p))
    fps, n = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    return fps, n, w, h


def main(root, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    idx = index_files(root)
    rows = []
    for (sid, task), f in sorted(idx.items()):
        r = dict(sid=sid, task=task, has_vid="vid" in f, has_bvp="bvp" in f, has_eda="eda" in f)
        durs = []
        if "bvp" in f:
            x = read_signal(f["bvp"])
            r.update(bvp_n=len(x), bvp_nan=np.isnan(x).mean(), bvp_std=np.nanstd(x))
            durs.append(len(x) / BVP_FS)
        if "eda" in f:
            x = read_signal(f["eda"])
            r.update(eda_n=len(x), eda_nan=np.isnan(x).mean(), eda_min=np.nanmin(x), eda_max=np.nanmax(x),
                     eda_std=np.nanstd(x))
            durs.append(len(x) / EDA_FS)
        if "vid" in f:
            fps, n, w, h = video_props(f["vid"])
            r.update(vid_fps=fps, vid_frames=n, vid_w=w, vid_h=h)
            if fps > 0:
                durs.append(n / fps)
                for k in ("bvp_n", "eda_n"):
                    if k in r:
                        r[f"implied_{k[:3]}_hz"] = r[k] / (n / fps)
        r["dur_spread_s"] = (max(durs) - min(durs)) if len(durs) > 1 else np.nan
        flags = []
        if not (r["has_vid"] and r["has_bvp"] and r["has_eda"]): flags.append("missing_modality")
        if r.get("bvp_nan", 0) > 0.05: flags.append("bvp_nan>5%")
        if r.get("eda_nan", 0) > 0.05: flags.append("eda_nan>5%")
        if r.get("bvp_std", 1) < 1e-6: flags.append("bvp_flat")
        if r.get("eda_max", 1) < 0.05 or r.get("eda_std", 1) < 1e-4: flags.append("eda_flat")
        if r["dur_spread_s"] > 2: flags.append("duration_mismatch>2s")
        r["issues"] = ";".join(flags)
        rows.append(r)
    inv = pd.DataFrame(rows)
    inv.to_csv(out / "inventory.csv", index=False)

    n_sub = inv.sid.nunique()
    complete = inv.groupby("sid").apply(lambda d: len(d) == 3 and d.issues.eq("").all())
    print(f"subjects found: {n_sub}; tasks per subject: {inv.groupby('sid').size().value_counts().to_dict()}")
    print(f"subjects with 3 tasks, all modalities, no flags: {int(complete.sum())}")
    print("\nflag counts:\n", inv.issues.str.split(";").explode().replace("", np.nan).value_counts().to_string())
    print("\ndurations (s) per task:")
    inv["dur_bvp"] = inv.get("bvp_n", np.nan) / BVP_FS
    print(inv.groupby("task")["dur_bvp"].describe()[["count", "min", "50%", "max"]].round(1).to_string())
    for k in ("implied_bvp_hz", "implied_eda_hz"):
        if k in inv:
            print(f"median {k}: {inv[k].median():.2f}  (expect {BVP_FS if 'bvp' in k else EDA_FS})")
    if "vid_fps" in inv:
        print("video fps:", inv.vid_fps.round(2).value_counts().to_dict(),
              " size:", inv[["vid_w", "vid_h"]].drop_duplicates().values.tolist())

    # subject info files + other files (questionnaires etc.) for the label audit
    info = []
    for p in sorted(Path(root).rglob("info_s*.txt")):
        info.append(dict(file=p.name, sid=int("".join(c for c in p.stem if c.isdigit())),
                         info_raw=" | ".join(p.read_text(errors="ignore").split())))
    pd.DataFrame(info).to_csv(out / "subjects_info.csv", index=False)
    print(f"\n{len(info)} info files -> subjects_info.csv (open it: what do these fields contain?)")
    others = [p for p in Path(root).rglob("*")
              if p.is_file() and not _PAT.fullmatch(p.name) and not p.name.startswith("info_s")
              and p.suffix.lower() in (".csv", ".xlsx", ".xls", ".txt", ".md", ".pdf")]
    print("other candidate label/metadata files:")
    for p in others[:20]:
        print("  ", p.relative_to(root))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    main(a.root, a.out)
