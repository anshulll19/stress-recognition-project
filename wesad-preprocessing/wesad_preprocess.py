#!/usr/bin/env python3
"""
wesad_preprocess.py
====================

WESAD (Wearable Stress and Affect Detection) — Phase 0 preprocessing script.

WHAT THIS SCRIPT DOES (and does NOT do)
----------------------------------------
This script ONLY does data engineering. It does not train a model and it
does not compute accuracy. Its job is:
  1. Load one subject's raw .pkl file (or loop over all subjects).
  2. Validate the wrist BVP channel (present, non-empty, ~64Hz, no NaNs).
  3. Align the 700Hz label array to the 64Hz BVP timeline BY TIME, not by
     raw index (this is the #1 known WESAD bug — see README section below).
  4. Run the PPGE-paper preprocessing recipe:
       - Butterworth bandpass filter, 0.7-3.7 Hz
       - 60-second windows, 5-second overlap (55s step)
       - Z-score normalization per window
  5. Assign a label to each window using a documented, explicit rule
     (majority-vote or discard) for windows that straddle a label change.
  6. Produce a plain data-quality report: which subjects loaded, final
     array shapes, and any corrupted/missing/misaligned data. No accuracy,
     no model selection, no subject cherry-picking based on how "good" the
     data looks for a future classifier.

Subjects are NEVER excluded because of how they might affect a future
accuracy number. They are only ever excluded for genuine data-integrity
reasons (file missing/corrupted, required channel missing, unreadable
label array), and every exclusion is written into the report with a
reason, so your team can audit the decision.

USAGE
-----
Single-subject test mode (do this first, e.g. on S2):
    python wesad_preprocess.py --data_dir /path/to/WESAD --mode single --subject S2

All-subjects mode (only after single-subject mode looks correct):
    python wesad_preprocess.py --data_dir /path/to/WESAD --mode all

Choosing the label-transition rule (default: majority):
    python wesad_preprocess.py --data_dir /path/to/WESAD --mode all --label_rule discard

Output:
    - One .npz file per successfully processed subject, in --output_dir,
      containing X (windows, 3840), y (window labels), and raw_label_frac
      (fraction of the window covered by the assigned label, for auditing).
    - A single report.json and report.txt in --output_dir summarizing
      everything (Step 4 deliverable).
"""

import argparse
import json
import os
import pickle
import sys
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, filtfilt

# ----------------------------------------------------------------------
# Constants (from the WESAD documentation / PPGE paper recipe)
# ----------------------------------------------------------------------
CHEST_FS = 700          # Hz — chest device AND the label array
WRIST_BVP_FS = 64       # Hz — Empatica E4 BVP channel
BANDPASS_LOW = 0.7      # Hz
BANDPASS_HIGH = 3.7     # Hz
BUTTER_ORDER = 4
WINDOW_SEC = 60
STEP_SEC = 55           # 60s window, 5s overlap -> 55s step
WINDOW_SAMPLES = WINDOW_SEC * WRIST_BVP_FS      # 3840
STEP_SAMPLES = STEP_SEC * WRIST_BVP_FS          # 3520

# WESAD label codes (from the dataset readme)
LABEL_NAMES = {
    0: "transient/undefined",
    1: "baseline",
    2: "stress",
    3: "amusement",
    4: "meditation",
    5: "ignore/other",
    6: "ignore/other",
    7: "ignore/other",
}

# WESAD ships subjects S2-S17, but S1 and S12 do not exist. This is normal.
EXPECTED_SUBJECTS = [f"S{i}" for i in range(2, 18) if i != 12]


@dataclass
class SubjectResult:
    subject: str
    status: str                     # "ok", "excluded"
    reason: str = ""                # only set if excluded / warning
    n_windows: int = 0
    x_shape: tuple = None
    bvp_len: int = 0
    bvp_duration_sec: float = 0.0
    label_len: int = 0
    label_duration_sec: float = 0.0
    unique_labels_raw: list = field(default_factory=list)
    windows_discarded_for_transition: int = 0
    warnings: list = field(default_factory=list)


# ----------------------------------------------------------------------
# Step 2: Load + validate a single subject
# ----------------------------------------------------------------------
def load_subject_pickle(data_dir: str, subject: str) -> dict:
    """Load one subject's .pkl file. Raises a clear exception on failure."""
    pkl_path = os.path.join(data_dir, subject, f"{subject}.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Expected file not found: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    return data


def validate_and_extract(data: dict, subject: str, result: SubjectResult):
    """
    Confirms the 3 things the task spec asks for:
      1. BVP channel present and readable
      2. Sampling rate matches ~64Hz (checked via known constant + duration sanity check)
      3. Label channel can be aligned to BVP correctly
    Returns (bvp_signal, raw_labels_700hz) or raises ValueError with a clear reason.
    """
    if "signal" not in data or "wrist" not in data.get("signal", {}):
        raise ValueError("Missing data['signal']['wrist'] — malformed subject file")

    wrist = data["signal"]["wrist"]
    if "BVP" not in wrist:
        raise ValueError("BVP channel missing entirely from wrist signal dict")

    bvp = np.asarray(wrist["BVP"]).squeeze()
    if bvp.ndim != 1:
        # Some subjects store as (N,1); squeeze should fix it, but guard anyway
        bvp = bvp.reshape(-1)

    if bvp.size == 0:
        raise ValueError("BVP channel is empty")

    if np.isnan(bvp).any():
        nan_frac = float(np.isnan(bvp).mean())
        if nan_frac > 0.01:
            raise ValueError(f"BVP channel is {nan_frac:.1%} NaN — treating as corrupted")
        else:
            result.warnings.append(f"BVP contains {nan_frac:.3%} NaN values; interpolated")
            # Interpolate small gaps rather than discard the whole subject
            idx = np.arange(len(bvp))
            good = ~np.isnan(bvp)
            bvp = np.interp(idx, idx[good], bvp[good])

    if "label" not in data:
        raise ValueError("data['label'] missing — cannot align labels to BVP")
    labels = np.asarray(data["label"]).squeeze()
    if labels.size == 0:
        raise ValueError("Label array is empty")

    result.bvp_len = int(bvp.size)
    result.bvp_duration_sec = bvp.size / WRIST_BVP_FS
    result.label_len = int(labels.size)
    result.label_duration_sec = labels.size / CHEST_FS
    result.unique_labels_raw = sorted(int(v) for v in np.unique(labels))

    # Sanity check: BVP duration and label duration should roughly agree
    # (both devices were recording the same session). Large mismatch is a
    # red flag for a misaligned / truncated file, not something to silently
    # ignore.
    duration_diff = abs(result.bvp_duration_sec - result.label_duration_sec)
    if duration_diff > 5.0:  # more than 5 seconds off -> flag it
        result.warnings.append(
            f"BVP duration ({result.bvp_duration_sec:.1f}s) and label duration "
            f"({result.label_duration_sec:.1f}s) differ by {duration_diff:.1f}s "
            f"— possible truncation/misalignment, inspect this subject manually."
        )

    return bvp, labels


# ----------------------------------------------------------------------
# The 700Hz -> 64Hz label alignment trap
# ----------------------------------------------------------------------
def align_labels_to_bvp(labels_700hz: np.ndarray, bvp_len: int) -> np.ndarray:
    """
    Aligns the 700Hz label array to the 64Hz BVP timeline BY TIME.

    WRONG (the #1 mistake): labels[::int(700/64)] or labels[:bvp_len] —
    this assumes both arrays start and advance in lockstep index-for-index,
    which they do NOT because the sampling rates differ.

    RIGHT: compute the real-world timestamp of every BVP sample and every
    label sample, then for each BVP timestamp pick the label that was
    active at that moment in time.
    """
    bvp_time = np.arange(bvp_len) / WRIST_BVP_FS
    label_time = np.arange(len(labels_700hz)) / CHEST_FS

    # For each BVP timestamp, find the index of the most recent label
    # sample at or before that time.
    idx = np.searchsorted(label_time, bvp_time, side="right") - 1
    idx = np.clip(idx, 0, len(labels_700hz) - 1)
    return labels_700hz[idx]


# ----------------------------------------------------------------------
# Step 3: Preprocessing pipeline
# ----------------------------------------------------------------------
def bandpass_filter(signal: np.ndarray, fs: int = WRIST_BVP_FS,
                     low: float = BANDPASS_LOW, high: float = BANDPASS_HIGH,
                     order: int = BUTTER_ORDER) -> np.ndarray:
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="bandpass")
    return filtfilt(b, a, signal)


def make_windows(signal: np.ndarray, labels_aligned: np.ndarray,
                  label_rule: str, result: SubjectResult):
    """
    Slides a 60s window / 55s step over the filtered BVP signal.
    For each window, decides the label using `label_rule`:
        "majority" -> label that covers >50% of the window's samples
        "discard"  -> drop the window entirely if it isn't 100% one label
    Returns X (n_windows, WINDOW_SAMPLES), y (n_windows,), label_frac (n_windows,)
    """
    n = len(signal)
    starts = list(range(0, n - WINDOW_SAMPLES + 1, STEP_SAMPLES))

    X, y, frac = [], [], []
    discarded = 0

    for s in starts:
        e = s + WINDOW_SAMPLES
        seg = signal[s:e]
        seg_labels = labels_aligned[s:e]

        vals, counts = np.unique(seg_labels, return_counts=True)
        majority_label = int(vals[np.argmax(counts)])
        majority_frac = counts.max() / counts.sum()

        if label_rule == "discard" and majority_frac < 1.0:
            discarded += 1
            continue
        # "majority" rule (default): always keep, use the majority label,
        # regardless of how small the majority is, but we record the
        # fraction so it's auditable.

        std = seg.std()
        if std == 0:
            # Flat/constant segment (e.g. sensor dropout) -> skip, not a
            # real physiological signal, would just become all-zeros/NaN
            # after z-scoring.
            result.warnings.append(
                f"Skipped a window at sample {s} with zero variance (likely sensor dropout)"
            )
            continue
        seg_norm = (seg - seg.mean()) / std

        X.append(seg_norm)
        y.append(majority_label)
        frac.append(majority_frac)

    result.windows_discarded_for_transition = discarded

    if len(X) == 0:
        return (np.empty((0, WINDOW_SAMPLES)), np.empty((0,), dtype=int),
                np.empty((0,)))

    return np.stack(X), np.array(y, dtype=int), np.array(frac)


# ----------------------------------------------------------------------
# Per-subject pipeline
# ----------------------------------------------------------------------
def process_subject(data_dir: str, subject: str, label_rule: str,
                     output_dir: str) -> SubjectResult:
    result = SubjectResult(subject=subject, status="ok")
    try:
        data = load_subject_pickle(data_dir, subject)
        bvp_raw, labels_700 = validate_and_extract(data, subject, result)

        labels_aligned = align_labels_to_bvp(labels_700, len(bvp_raw))

        bvp_filtered = bandpass_filter(bvp_raw)

        X, y, frac = make_windows(bvp_filtered, labels_aligned, label_rule, result)

        result.n_windows = int(X.shape[0])
        result.x_shape = tuple(X.shape)

        if result.n_windows == 0:
            result.warnings.append(
                "No windows produced (recording too short, or all windows discarded)."
            )

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{subject}_windows.npz")
        np.savez_compressed(out_path, X=X, y=y, label_fraction=frac)

    except Exception as e:
        result.status = "excluded"
        result.reason = f"{type(e).__name__}: {e}"

    return result


# ----------------------------------------------------------------------
# Step 4: Report
# ----------------------------------------------------------------------
def build_report(results: list, label_rule: str, data_dir: str) -> dict:
    present = sorted({r.subject for r in results})
    ok = [r for r in results if r.status == "ok"]
    excluded = [r for r in results if r.status == "excluded"]
    missing_expected = [s for s in EXPECTED_SUBJECTS if s not in present]

    report = {
        "data_dir": data_dir,
        "label_transition_rule": label_rule,
        "expected_subjects": EXPECTED_SUBJECTS,
        "subjects_found_on_disk": present,
        "subjects_missing_from_disk": missing_expected,
        "note_on_missing_subjects": (
            "S1 and S12 are expected to be absent from WESAD itself (dataset "
            "quirk, not corruption). Any OTHER subject missing from "
            "'subjects_found_on_disk' above was not present in --data_dir "
            "and should be checked (extraction issue / incomplete download)."
        ),
        "subjects_processed_ok": [r.subject for r in ok],
        "subjects_excluded": [
            {"subject": r.subject, "reason": r.reason} for r in excluded
        ],
        "per_subject_detail": [
            {
                "subject": r.subject,
                "status": r.status,
                "reason": r.reason,
                "n_windows": r.n_windows,
                "output_array_shape": r.x_shape,
                "bvp_length_samples": r.bvp_len,
                "bvp_duration_sec": round(r.bvp_duration_sec, 2),
                "label_length_samples": r.label_len,
                "label_duration_sec": round(r.label_duration_sec, 2),
                "unique_raw_label_values": r.unique_labels_raw,
                "windows_discarded_for_label_transition": r.windows_discarded_for_transition,
                "warnings": r.warnings,
            }
            for r in results
        ],
        "total_windows_all_subjects": int(sum(r.n_windows for r in ok)),
        "label_code_reference": LABEL_NAMES,
    }
    return report


def report_to_text(report: dict) -> str:
    lines = []
    lines.append("WESAD PREPROCESSING REPORT (Phase 0 — data engineering only, no accuracy)")
    lines.append("=" * 75)
    lines.append(f"Data directory: {report['data_dir']}")
    lines.append(f"Label-transition rule used: {report['label_transition_rule']}")
    lines.append("")
    lines.append(f"Subjects expected (WESAD spec, S1/S12 don't exist): {report['expected_subjects']}")
    lines.append(f"Subjects found on disk:                            {report['subjects_found_on_disk']}")
    if report["subjects_missing_from_disk"]:
        lines.append(f"Subjects MISSING from disk (check download!):     {report['subjects_missing_from_disk']}")
    lines.append("")
    lines.append(f"Subjects processed successfully: {report['subjects_processed_ok']}")
    if report["subjects_excluded"]:
        lines.append("Subjects EXCLUDED (data-integrity reasons only):")
        for item in report["subjects_excluded"]:
            lines.append(f"   - {item['subject']}: {item['reason']}")
    else:
        lines.append("Subjects excluded: none")
    lines.append("")
    lines.append(f"Total windows across all processed subjects: {report['total_windows_all_subjects']}")
    lines.append("")
    lines.append("Per-subject detail:")
    for d in report["per_subject_detail"]:
        lines.append(f"  {d['subject']}: status={d['status']}"
                      + (f" ({d['reason']})" if d["reason"] else ""))
        if d["status"] == "ok":
            lines.append(f"      output shape: {d['output_array_shape']}  "
                          f"(n_windows, {WINDOW_SAMPLES})")
            lines.append(f"      BVP: {d['bvp_length_samples']} samples "
                          f"(~{d['bvp_duration_sec']}s)   "
                          f"labels: {d['label_length_samples']} samples "
                          f"(~{d['label_duration_sec']}s)")
            lines.append(f"      raw label values present: {d['unique_raw_label_values']}")
            lines.append(f"      windows discarded for straddling a label transition: "
                          f"{d['windows_discarded_for_label_transition']}")
        if d["warnings"]:
            for w in d["warnings"]:
                lines.append(f"      WARNING: {w}")
    lines.append("")
    lines.append("Label code reference: " + json.dumps(report["label_code_reference"]))
    return "\n".join(lines)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="WESAD Phase-0 preprocessing (no accuracy, data-engineering only)")
    parser.add_argument("--data_dir", required=True, help="Path to the extracted WESAD folder (contains S2/, S3/, ...)")
    parser.add_argument("--mode", choices=["single", "all"], default="single")
    parser.add_argument("--subject", default="S2", help="Subject ID for --mode single, e.g. S2")
    parser.add_argument("--label_rule", choices=["majority", "discard"], default="majority",
                         help="How to label windows that straddle a transition. "
                              "majority = keep window, use the label covering >50%%. "
                              "discard = drop any window that isn't 100%% one label.")
    parser.add_argument("--output_dir", default="./wesad_processed")
    args = parser.parse_args()

    if args.mode == "single":
        subjects = [args.subject]
    else:
        # scan the data_dir for anything matching S<number>
        found = []
        if os.path.isdir(args.data_dir):
            for name in sorted(os.listdir(args.data_dir)):
                if name.startswith("S") and name[1:].isdigit():
                    if os.path.exists(os.path.join(args.data_dir, name, f"{name}.pkl")):
                        found.append(name)
        subjects = sorted(found, key=lambda s: int(s[1:])) if found else EXPECTED_SUBJECTS

    results = []
    for subj in subjects:
        print(f"Processing {subj} ...")
        r = process_subject(args.data_dir, subj, args.label_rule, args.output_dir)
        results.append(r)
        if r.status == "ok":
            print(f"  OK -> {r.x_shape} windows saved")
        else:
            print(f"  EXCLUDED -> {r.reason}")

    report = build_report(results, args.label_rule, args.data_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    text_report = report_to_text(report)
    with open(os.path.join(args.output_dir, "report.txt"), "w") as f:
        f.write(text_report)

    print("\n" + text_report)
    print(f"\nSaved: {os.path.join(args.output_dir, 'report.json')}")
    print(f"Saved: {os.path.join(args.output_dir, 'report.txt')}")


if __name__ == "__main__":
    main()
