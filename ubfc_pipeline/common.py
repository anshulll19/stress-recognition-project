"""Shared config, loaders and feature functions for the UBFC-Phys pipeline.

Physio feature functions take *pre-processed whole-task signals* (filtered once per task,
so windows do not suffer filter edge effects) and then slice windows.
They are written so the same functions can be reused on WESAD wrist BVP/EDA (also E4, 64/4 Hz).
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt, welch

BVP_FS, EDA_FS = 64.0, 4.0          # nominal; s0_inventory.py checks the implied rates
TASKS = ["T1", "T2", "T3"]          # rest, speech, arithmetic (verify against the dataset paper)
TASK_NAMES = {"T1": "rest", "T2": "speech", "T3": "arithmetic"}
WIN_S, STRIDE_S = 30.0, 10.0

_PAT = re.compile(r"(vid|bvp|eda)_s(\d+)_T(\d)\.(avi|csv)")


def index_files(root):
    """{(subject_id, 'T1'): {'vid': Path, 'bvp': Path, 'eda': Path}} regardless of folder layout."""
    idx = {}
    for p in Path(root).rglob("*"):
        m = _PAT.fullmatch(p.name)
        if m:
            kind, sid, t, _ = m.groups()
            idx.setdefault((int(sid), f"T{t}"), {})[kind] = p
    return idx


def read_signal(p):
    x = pd.read_csv(p, header=None).iloc[:, 0]
    return pd.to_numeric(x, errors="coerce").to_numpy(float)


def clean(x):
    return pd.Series(x).interpolate(limit_direction="both").to_numpy(float)


def bandpass(x, fs, lo, hi, order=3):
    return sosfiltfilt(butter(order, [lo, hi], btype="band", fs=fs, output="sos"), x)


def lowpass(x, fs, fc, order=2):
    return sosfiltfilt(butter(order, fc, btype="low", fs=fs, output="sos"), x)


# ----------------------------------------------------------------- BVP (pulse rate variability)
def prepare_bvp(x, fs=BVP_FS):
    x = clean(x)
    x = x - x.mean()
    y = bandpass(x, fs, 0.7, 3.5)
    pk, _ = find_peaks(y, distance=int(0.33 * fs), prominence=0.4 * np.std(y))
    return dict(y=y, peak_t=pk / fs, fs=fs)


def bvp_window(prep, t0, t1):
    names = ["hr_mean", "sdnn", "rmssd", "pnn50", "bvp_amp"]
    out = {f"p_{n}": np.nan for n in names}
    out["q_ibi_valid_frac"] = 0.0
    fs = prep["fs"]
    y = prep["y"][int(t0 * fs):int(t1 * fs)]
    if len(y) > 0:
        out["p_bvp_amp"] = np.percentile(y, 95) - np.percentile(y, 5)
    pt = prep["peak_t"]
    sel = pt[(pt >= t0) & (pt < t1)]
    if len(sel) < 8:
        return out
    ibi = np.diff(sel) * 1000.0
    med = pd.Series(ibi).rolling(5, center=True, min_periods=1).median().to_numpy()
    ok = (ibi > 300) & (ibi < 1500) & (np.abs(ibi - med) < 0.25 * med)
    out["q_ibi_valid_frac"] = float(ok.mean())
    if ok.sum() < 8:
        return out
    d = np.diff(ibi)
    okd = ok[:-1] & ok[1:]
    out["p_hr_mean"] = 60000.0 / ibi[ok].mean()
    out["p_sdnn"] = ibi[ok].std(ddof=1)
    if okd.sum() >= 5:
        out["p_rmssd"] = float(np.sqrt(np.mean(d[okd] ** 2)))
        out["p_pnn50"] = float(100 * np.mean(np.abs(d[okd]) > 50))
    return out


# ------------------------------------------------------------------------------------------ EDA
def prepare_eda(x, fs=EDA_FS):
    x = clean(x)
    tonic = lowpass(x, fs, 0.05)
    phasic = x - tonic
    noise = 1.4826 * np.median(np.abs(np.diff(x) - np.median(np.diff(x)))) / np.sqrt(2)   # robust sample noise
    prom = max(0.02, 4 * noise)            # uS; floor ~ common SCR criterion, raised for noisy recordings
    pk, _ = find_peaks(lowpass(phasic, fs, 1.0), prominence=prom, distance=int(fs))
    return dict(x=x, tonic=tonic, phasic=phasic, peak_i=pk, fs=fs)


def eda_window(prep, t0, t1):
    fs = prep["fs"]
    a, b = int(t0 * fs), int(t1 * fs)
    ton, ph, x = prep["tonic"][a:b], prep["phasic"][a:b], prep["x"][a:b]
    pk = prep["peak_i"][(prep["peak_i"] >= a) & (prep["peak_i"] < b)]
    slope = np.polyfit(np.arange(len(ton)) / fs, ton, 1)[0] if len(ton) > 3 else np.nan
    return {"p_scl_mean": ton.mean(), "p_scl_slope": slope, "p_eda_std": x.std(),
            "p_scr_n": float(len(pk)), "p_scr_amp": float(ph[pk - a].sum()) if len(pk) else 0.0}


# ------------------------------------------------------------------------------------ video
BS_GROUPS = {   # MediaPipe blendshape groups -> AU-like facial descriptors (L/R averaged)
    "brow_down": ["browDownLeft", "browDownRight"],
    "brow_inner_up": ["browInnerUp"],
    "brow_outer_up": ["browOuterUpLeft", "browOuterUpRight"],
    "eye_squint": ["eyeSquintLeft", "eyeSquintRight"],
    "eye_wide": ["eyeWideLeft", "eyeWideRight"],
    "nose_sneer": ["noseSneerLeft", "noseSneerRight"],
    "mouth_press": ["mouthPressLeft", "mouthPressRight"],
    "mouth_frown": ["mouthFrownLeft", "mouthFrownRight"],
    "mouth_smile": ["mouthSmileLeft", "mouthSmileRight"],
    "mouth_stretch": ["mouthStretchLeft", "mouthStretchRight"],
    "jaw_open": ["jawOpen"],
}
BLINK = ["eyeBlinkLeft", "eyeBlinkRight"]


def pos_rppg(rgb, fs, win_s=1.6):
    """POS (Wang et al. 2017) on an (n,3) RGB trace -> pulse signal (overlap-add)."""
    n, L = len(rgb), int(win_s * fs)
    h = np.zeros(n)
    for s in range(0, n - L + 1):
        C = rgb[s:s + L].T
        Cn = C / (C.mean(axis=1, keepdims=True) + 1e-8)
        S1, S2 = Cn[1] - Cn[2], Cn[1] + Cn[2] - 2 * Cn[0]
        p = S1 + (S1.std() / (S2.std() + 1e-8)) * S2
        h[s:s + L] += p - p.mean()
    return h


def prepare_video(npz_path):
    d = np.load(npz_path, allow_pickle=False)
    fps = float(d["fps"])
    rgb = d["rgb"].astype(float)
    valid = ~np.isnan(rgb).any(1)
    rppg = np.full(len(rgb), np.nan)
    if valid.mean() > 0.5 and len(rgb) > 10 * fps:
        r = pd.DataFrame(rgb).interpolate(limit=int(fps), limit_direction="both").to_numpy().copy()
        ok = ~np.isnan(r).any(1)
        r[~ok] = np.nanmean(r[ok], 0)
        h = bandpass(pos_rppg(r, fps), fps, 0.7, 3.5)
        h[~ok] = np.nan
        rppg = h
    return dict(fps=fps, dur=float(d["n_frames"]) / fps, t_lm=d["t_lm"], bs=d["bs"], pose=d["pose"],
                names=[str(s) for s in d["bs_names"]], rppg=rppg)


def count_blinks(s, hi=0.5, lo=0.3):
    n, closed = 0, False
    for v in s:
        if np.isnan(v):
            closed = False
        elif not closed and v > hi:
            n, closed = n + 1, True
        elif closed and v < lo:
            closed = False
    return n


def _wrap(a):
    return (a + 180) % 360 - 180


def rppg_window(h, fps, t0, t1):
    seg = h[int(t0 * fps):int(t1 * fps)]
    if len(seg) < 10 * fps or np.isnan(seg).mean() > 0.2:
        return np.nan, np.nan
    f, P = welch(np.nan_to_num(seg), fs=fps, nperseg=min(len(seg), int(10 * fps)), nfft=8192)
    band = (f >= 0.7) & (f <= 3.0)
    fpk = f[band][np.argmax(P[band])]
    near = band & (np.abs(f - fpk) < 0.15)
    return 60 * fpk, P[near].sum() / (P[band].sum() + 1e-12)


def video_window(v, t0, t1):
    out = {}
    m = (v["t_lm"] >= t0) & (v["t_lm"] < t1)
    bs, pose, tl = v["bs"][m], v["pose"][m], v["t_lm"][m]
    ok = ~np.isnan(bs[:, 0]) if m.any() else np.zeros(0, bool)
    out["q_face_frac"] = float(ok.mean()) if m.any() else 0.0
    keys = [f"v_{g}_{s}" for g in BS_GROUPS for s in ("mean", "std")] + \
           ["v_blink_rate", "v_head_pitch_std", "v_head_yaw_std", "v_head_roll_std", "v_head_speed"]
    out.update({k: np.nan for k in keys})
    if ok.sum() >= 10:
        b = bs[ok]
        for g, nm in BS_GROUPS.items():
            s = b[:, [v["names"].index(n) for n in nm]].mean(1)
            out[f"v_{g}_mean"], out[f"v_{g}_std"] = s.mean(), s.std()
        dt = np.median(np.diff(tl)) if len(tl) > 1 else np.nan
        bl = bs[:, [v["names"].index(n) for n in BLINK]].mean(1)
        out["v_blink_rate"] = count_blinks(bl) / (ok.sum() * dt) * 60.0
        q = _wrap(pose[ok] - pose[ok][0])
        q = q - np.median(q, 0)
        out["v_head_pitch_std"], out["v_head_yaw_std"], out["v_head_roll_std"] = q.std(0)
        out["v_head_speed"] = float(np.mean(np.linalg.norm(_wrap(np.diff(pose[ok], axis=0)), axis=1)) / dt)
    hr, snr = rppg_window(v["rppg"], v["fps"], t0, t1)
    out["v_rppg_hr"], out["v_rppg_snr"] = hr, snr
    return out
