#!/usr/bin/env python3
"""
inspect_empathicschool.py
=========================

Robust, zero-assumption diagnostic inspection script for the EmpathicSchool dataset.
Designed to be run immediately after the dataset archive is unpacked.

Usage:
    python scripts/inspect_empathicschool.py <DATASET_PATH> [--output report.md] [--max-files 20]

This script reports what physically exists:
- Directory hierarchy and file extension distribution
- Detected subject and session/trial candidate identifiers
- Inspection of structured files (.csv, .tsv, .npz, .npy, .mat, .json, .txt)
- Signal channel names, estimated sampling rates, and missing values
- Video files (counts, formats, file sizes, and metadata if available)
- Metadata and timestamp synchronization markers
"""

import argparse
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def format_bytes(size: int) -> str:
    """Format byte size into human readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def inspect_directory_tree(dataset_path: Path, max_depth: int = 4):
    """Summarizes directory structure up to max_depth."""
    tree_lines = []
    base_depth = len(dataset_path.parts)

    folder_count = 0
    file_count = 0

    for root, dirs, files in os.walk(dataset_path):
        current_depth = len(Path(root).parts) - base_depth
        if current_depth > max_depth:
            continue
        indent = "  " * current_depth
        rel_path = os.path.relpath(root, dataset_path)
        display_name = "." if rel_path == "." else os.path.basename(root)
        tree_lines.append(f"{indent}📁 {display_name}/ ({len(files)} files, {len(dirs)} subdirs)")
        folder_count += len(dirs)
        file_count += len(files)

    return tree_lines, folder_count, file_count


def scan_files_and_extensions(dataset_path: Path):
    """Categorizes all files by extension and finds potential subject/session IDs."""
    extension_counts = Counter()
    total_size = 0
    all_files = []

    subject_pattern = re.compile(r"(?:sub[_-]?|p|s|participant[_-]?|subject[_-]?)(\d+|[a-zA-Z0-9]+)", re.IGNORECASE)
    session_pattern = re.compile(r"(?:ses[_-]?|session[_-]?|trial[_-]?|task[_-]?|block[_-]?)(\d+|[a-zA-Z0-9]+)", re.IGNORECASE)

    detected_subjects = set()
    detected_sessions = set()

    for root, _, files in os.walk(dataset_path):
        for f in files:
            if f.startswith("."):
                continue  # skip hidden/OS files
            file_path = Path(root) / f
            ext = file_path.suffix.lower()
            extension_counts[ext] += 1
            try:
                size = file_path.stat().st_size
                total_size += size
            except OSError:
                size = 0
            all_files.append((file_path, ext, size))

            # Pattern search in relative paths
            rel_str = str(file_path.relative_to(dataset_path))
            sub_matches = subject_pattern.findall(rel_str)
            for m in sub_matches:
                detected_subjects.add(str(m))

            ses_matches = session_pattern.findall(rel_str)
            for m in ses_matches:
                detected_sessions.add(str(m))

    return extension_counts, total_size, all_files, detected_subjects, detected_sessions


def inspect_csv_tsv(file_path: Path, max_rows: int = 5):
    """Inspects CSV/TSV table structure, headers, potential signals, and time columns."""
    summary = {
        "path": str(file_path),
        "columns": [],
        "num_rows": 0,
        "sample_rows": [],
        "time_col": None,
        "estimated_fs": None,
        "potential_physio_cols": [],
        "potential_label_cols": [],
        "null_counts": {},
    }

    try:
        import pandas as pd
        sep = "\t" if file_path.suffix.lower() == ".tsv" else None
        df = pd.read_csv(file_path, sep=sep, nrows=1000, engine="python")
        summary["columns"] = list(df.columns)
        summary["null_counts"] = df.isnull().sum().to_dict()

        # Identify candidate columns
        col_lower = [str(c).lower() for c in df.columns]
        for col, cl in zip(df.columns, col_lower):
            if any(term in cl for term in ["time", "timestamp", "sec", "ms"]):
                summary["time_col"] = col
            if any(term in cl for term in ["ppg", "bvp", "pulse", "pleth", "hr", "ibi"]):
                summary["potential_physio_cols"].append(col)
            if any(term in cl for term in ["label", "stress", "affect", "condition", "state", "target"]):
                summary["potential_label_cols"].append(col)

        # Estimate sampling rate if time column exists and numeric
        if summary["time_col"] and pd.api.types.is_numeric_dtype(df[summary["time_col"]]):
            time_vals = df[summary["time_col"]].dropna().values
            if len(time_vals) > 1:
                dt = np.diff(time_vals)
                median_dt = np.median(dt)
                if median_dt > 0:
                    # Heuristic check for milliseconds vs seconds
                    if median_dt > 0.5:
                        summary["estimated_fs"] = 1.0 / (median_dt / 1000.0) if median_dt > 10 else 1.0 / median_dt
                    else:
                        summary["estimated_fs"] = 1.0 / median_dt

        # Total line count via simple iteration
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            summary["num_rows"] = sum(1 for _ in f) - 1

    except Exception as e:
        summary["error"] = str(e)

    return summary


def inspect_npz_npy(file_path: Path):
    """Inspects NPY/NPZ contents, shapes, and dtypes."""
    summary = {"path": str(file_path), "keys": {}, "error": None}
    try:
        if file_path.suffix.lower() == ".npz":
            data = np.load(file_path, allow_pickle=True)
            for k in data.files:
                arr = data[k]
                summary["keys"][k] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
        else:
            arr = np.load(file_path, allow_pickle=True)
            summary["keys"]["array"] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    except Exception as e:
        summary["error"] = str(e)
    return summary


def inspect_mat_file(file_path: Path):
    """Inspects MATLAB .mat file keys and array shapes."""
    summary = {"path": str(file_path), "keys": {}, "error": None}
    try:
        from scipy.io import loadmat
        mat = loadmat(file_path, struct_as_record=False, squeeze_me=True)
        for k, v in mat.items():
            if not k.startswith("__"):
                shape = getattr(v, "shape", "non-array")
                summary["keys"][k] = {"shape": list(shape) if hasattr(shape, "__iter__") else str(shape), "type": type(v).__name__}
    except Exception as e:
        summary["error"] = str(e)
    return summary


def inspect_video_file(file_path: Path):
    """Inspects video metadata (dimensions, fps, duration) if cv2 or basic probing is available."""
    summary = {"path": str(file_path), "size": file_path.stat().st_size, "metadata": None}
    try:
        import cv2
        cap = cv2.VideoCapture(str(file_path))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration_sec = frame_count / fps if fps > 0 else 0
            summary["metadata"] = {
                "fps": fps,
                "frame_count": frame_count,
                "resolution": f"{width}x{height}",
                "duration_seconds": duration_sec,
            }
            cap.release()
    except Exception as e:
        summary["metadata"] = f"OpenCV inspection unavailable: {e}"
    return summary


def run_inspection(dataset_dir: str, max_samples_per_type: int = 5, output_file: str = None):
    dataset_path = Path(dataset_dir).expanduser().resolve()
    print("=" * 80)
    print(f"EMPATHICSCHOOL ZERO-ASSUMPTION DATASET INSPECTOR")
    print(f"Target Path: {dataset_path}")
    print("=" * 80)

    if not dataset_path.exists():
        print(f"\n[STATUS] Target directory does NOT exist yet: {dataset_path}")
        print("Note: Access to EmpathicSchool is currently pending on Zenodo.")
        print("Once the dataset is downloaded, rerun this script to perform full discovery.\n")
        return

    report_lines = []
    def log(msg=""):
        print(msg)
        report_lines.append(msg)

    log(f"# EmpathicSchool Physical Dataset Inspection Report")
    log(f"**Root Path:** `{dataset_path}`\n")

    # 1. Directory Tree Summary
    log("## 1. Directory Hierarchy Overview")
    tree_lines, folder_count, file_count = inspect_directory_tree(dataset_path)
    for line in tree_lines[:30]:
        log(line)
    if len(tree_lines) > 30:
        log(f"  ... [truncated, {len(tree_lines) - 30} deeper folders omitted]")
    log(f"\nTotal directories: {folder_count}, Total files: {file_count}\n")

    # 2. Extension Distribution & Storage
    log("## 2. File Format Breakdown & Total Size")
    extension_counts, total_size, all_files, detected_subjects, detected_sessions = scan_files_and_extensions(dataset_path)
    log(f"**Total Footprint:** {format_bytes(total_size)}")
    log("| Extension | Count |")
    log("| :--- | :--- |")
    for ext, count in sorted(extension_counts.items(), key=lambda x: -x[1]):
        log(f"| `{ext if ext else '[none]'}` | {count} |")
    log("")

    # 3. Subject & Session Extraction
    log("## 3. Discovered Subject & Session / Block Identifiers")
    log(f"- Potential Candidate Subjects Discovered ({len(detected_subjects)}): {sorted(list(detected_subjects))[:20]}")
    if len(detected_subjects) > 20:
        log(f"  ... (+ {len(detected_subjects) - 20} more)")
    log(f"- Potential Candidate Sessions/Blocks ({len(detected_sessions)}): {sorted(list(detected_sessions))[:20]}")
    log("")

    # 4. Deep Inspection of Structured Data Files
    files_by_ext = defaultdict(list)
    for p, ext, sz in all_files:
        files_by_ext[ext].append(p)

    log("## 4. Structured Data File Inspection")
    for ext in [".csv", ".tsv", ".npz", ".npy", ".mat", ".h5", ".json"]:
        sample_files = files_by_ext.get(ext, [])[:max_samples_per_type]
        if not sample_files:
            continue
        log(f"### Format: `{ext}` (Sample inspection of {len(sample_files)} files)")
        for fp in sample_files:
            rel = fp.relative_to(dataset_path)
            if ext in [".csv", ".tsv"]:
                res = inspect_csv_tsv(fp)
                log(f"- **File:** `{rel}` (~{res.get('num_rows', 0)} rows)")
                log(f"  - Columns ({len(res['columns'])}): `{res['columns'][:15]}`")
                if res.get("time_col"):
                    log(f"  - Detected Time Column: `{res['time_col']}` (Estimated fs: {res.get('estimated_fs', 'N/A')})")
                if res.get("potential_physio_cols"):
                    log(f"  - Candidate Physio Signals: `{res['potential_physio_cols']}`")
                if res.get("potential_label_cols"):
                    log(f"  - Candidate Label / State Fields: `{res['potential_label_cols']}`")
            elif ext in [".npz", ".npy"]:
                res = inspect_npz_npy(fp)
                log(f"- **File:** `{rel}`")
                for k, v in res.get("keys", {}).items():
                    log(f"  - Array `{k}`: shape={v['shape']}, dtype={v['dtype']}")
            elif ext == ".mat":
                res = inspect_mat_file(fp)
                log(f"- **File:** `{rel}`")
                for k, v in res.get("keys", {}).items():
                    log(f"  - Key `{k}`: shape={v.get('shape')}, type={v.get('type')}")
        log("")

    # 5. Video File Inspection
    video_exts = [".mp4", ".avi", ".mkv", ".mov", ".wmv"]
    video_files = []
    for ve in video_exts:
        video_files.extend(files_by_ext.get(ve, []))

    log("## 5. Video Recordings Inspection")
    log(f"Total video files discovered: {len(video_files)}")
    if video_files:
        for vf in video_files[:max_samples_per_type]:
            vres = inspect_video_file(vf)
            rel = vf.relative_to(dataset_path)
            log(f"- **Video:** `{rel}` ({format_bytes(vres['size'])})")
            if isinstance(vres["metadata"], dict):
                m = vres["metadata"]
                log(f"  - Resolution: {m['resolution']}, FPS: {m['fps']:.2f}, Frames: {m['frame_count']}, Duration: {m['duration_seconds']:.1f}s")
            elif vres["metadata"]:
                log(f"  - Probing note: {vres['metadata']}")
    else:
        log("No standard video files found at this root.")
    log("")

    # 6. Synchronization & Metadata Markers
    log("## 6. Timestamp & Synchronization Findings")
    metadata_files = [f for f in all_files if any(term in f[0].name.lower() for term in ["sync", "time", "marker", "align", "info", "readme", "meta"])]
    log(f"Files matching synchronization/metadata keywords ({len(metadata_files)}):")
    for mf, _, sz in metadata_files[:10]:
        log(f"- `{mf.relative_to(dataset_path)}` ({format_bytes(sz)})")
    log("")

    # Summary
    log("=" * 80)
    log("Inspection completed without hardcoded assumptions.")
    log("=" * 80)

    if output_file:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        print(f"\nReport written to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect EmpathicSchool physical files without assumptions.")
    parser.add_argument("dataset_path", help="Path to unpacked EmpathicSchool dataset directory")
    parser.add_argument("--output", default=None, help="Optional output Markdown report path")
    parser.add_argument("--max-samples", type=int, default=5, help="Max files to inspect per extension")
    args = parser.parse_args()

    run_inspection(args.dataset_path, max_samples_per_type=args.max_samples, output_file=args.output)
