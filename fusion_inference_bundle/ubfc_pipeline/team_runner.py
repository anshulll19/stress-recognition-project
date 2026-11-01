"""Team entry point for reproducible Stage 1 video processing."""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import mediapipe
import matplotlib
import numpy as np
import pandas
import scipy
import sklearn

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))

from common import index_files


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(
        description="Run Stage 1 and produce a reproducibility manifest and QC report."
    )
    ap.add_argument("--root", required=True, help="Dataset root")
    ap.add_argument("--out", required=True, help="New output directory for this run")
    ap.add_argument("--model", default=str(PROJECT / "models/face_landmarker.task"))
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--subjects", type=int, nargs="*")
    ap.add_argument("--limit-frames", type=int, default=0)
    ap.add_argument("--no-rppg", action="store_true")
    args = ap.parse_args()

    if args.stride < 1:
        ap.error("--stride must be >= 1")
    if args.limit_frames < 0:
        ap.error("--limit-frames must be >= 0")

    root = Path(args.root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    model = Path(args.model).expanduser().resolve()

    if not root.is_dir():
        ap.error(f"Dataset root does not exist: {root}")
    if not model.is_file():
        ap.error(f"Model file does not exist: {model}")

    files = index_files(root)
    selected = {
        (sid, task): f
        for (sid, task), f in sorted(files.items())
        if "vid" in f and (not args.subjects or sid in args.subjects)
    }
    if not selected:
        ap.error("No matching videos found. Check --root and --subjects.")

    out.mkdir(parents=True, exist_ok=True)
    expected_names = [f"s{sid}_{task}.npz" for sid, task in selected]
    existing = [name for name in expected_names if (out / name).exists()]
    if existing:
        ap.error(
            "Refusing to reuse existing outputs. Choose a fresh --out directory. "
            f"Existing files: {existing}"
        )

    command = [
        sys.executable, str(HERE / "s1_video_ts.py"),
        "--root", str(root),
        "--out", str(out),
        "--model", str(model),
        "--stride", str(args.stride),
    ]
    if args.subjects:
        command += ["--subjects", *map(str, args.subjects)]
    if args.limit_frames:
        command += ["--limit-frames", str(args.limit_frames)]
    if args.no_rppg:
        command.append("--no-rppg")

    started = time.time()
    print("Running shared Stage 1 pipeline:")
    print(" ".join(command))
    subprocess.run(command, cwd=HERE, check=True)
    elapsed = round(time.time() - started, 2)

    manifest, qc = [], []
    for (sid, task), f in selected.items():
        name = f"s{sid}_{task}.npz"
        path = out / name
        if not path.is_file():
            raise RuntimeError(f"Expected output was not created: {path}")

        with np.load(path, allow_pickle=False) as d:
            fps = float(d["fps"])
            n_frames = int(d["n_frames"])
            t = d["t_lm"]
            bs = d["bs"]
            pose = d["pose"]
            rgb = d["rgb"]
            names = d["bs_names"]

            expected_samples = (n_frames + args.stride - 1) // args.stride
            timestamp_increasing = bool(
                len(t) > 0 and np.all(np.diff(t) > 0)
            )
            timestamp_step_ok = bool(
                len(t) < 2
                or np.allclose(
                    np.diff(t), args.stride / fps,
                    rtol=0.02, atol=0.002
                )
            )
            bs_finite = float(np.isfinite(bs).mean()) if bs.size else 0.0
            pose_finite = float(np.isfinite(pose).mean()) if pose.size else 0.0
            rgb_finite = float(np.isfinite(rgb).mean()) if rgb.size else None
            face_fraction = (
                float(np.isfinite(bs[:, 0]).mean()) if bs.ndim == 2 and bs.shape[0]
                else 0.0
            )

            structural_ok = (
                n_frames > 0
                and len(t) == expected_samples
                and bs.shape == (len(t), len(names))
                and pose.shape == (len(t), 3)
                and timestamp_increasing
                and timestamp_step_ok
                and (rgb.shape == (n_frames, 3) if not args.no_rppg
                     else rgb.shape == (0, 3))
            )

            warnings = []
            if face_fraction < 0.80:
                warnings.append("low_face_valid_fraction")
            if not args.no_rppg and (rgb_finite is None or rgb_finite < 0.95):
                warnings.append("rgb_has_missing_values")
            if bs_finite < 0.95 or pose_finite < 0.95:
                warnings.append("landmark_or_pose_missing_values")

            manifest.append({
                "subject_id": sid,
                "task_id": task,
                "video_path": str(f["vid"]),
                "output_npz": str(path),
                "video_fps_reported": fps,
                "frames_processed": n_frames,
                "landmark_samples": len(t),
                "blendshape_count": len(names),
                "stride": args.stride,
                "window_seconds": 30,
                "window_stride_seconds": 10,
                "rppg_enabled": not args.no_rppg,
                "limit_frames": args.limit_frames,
                "model_path": str(model),
                "pipeline_script": "s1_video_ts.py",
                "pipeline_version": "team-runner-v1",
            })
            qc.append({
                "subject_id": sid,
                "task_id": task,
                "structural_status": "PASS" if structural_ok else "FAIL",
                "frames_processed": n_frames,
                "landmark_samples": len(t),
                "expected_landmark_samples": expected_samples,
                "blendshape_finite_fraction": round(bs_finite, 5),
                "pose_finite_fraction": round(pose_finite, 5),
                "face_valid_fraction": round(face_fraction, 5),
                "rgb_finite_fraction": (
                    round(rgb_finite, 5) if rgb_finite is not None else "disabled"
                ),
                "timestamps_increasing": timestamp_increasing,
                "timestamp_spacing_ok": timestamp_step_ok,
                "warnings": ";".join(warnings),
            })

    write_csv(out / "manifest.csv", manifest)
    write_csv(out / "qc.csv", qc)

    def git_value(*git_args):
        try:
            result = subprocess.run(
                ["git", *git_args],
                cwd=PROJECT,
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    model_hash = hashlib.sha256()
    with model.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            model_hash.update(chunk)

    package_versions = {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pandas.__version__,
        "scikit-learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "opencv-python": cv2.__version__,
        "mediapipe": mediapipe.__version__,
    }

    metadata = {
        "pipeline_version": "team-runner-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "package_versions": package_versions,
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_worktree_dirty": (
            bool(git_value("status", "--porcelain"))
            if git_value("rev-parse", "--is-inside-work-tree") == "true"
            else None
        ),
        "model_path": str(model),
        "model_sha256": model_hash.hexdigest(),
        "root": str(root),
        "stride": args.stride,
        "limit_frames": args.limit_frames,
        "rppg_enabled": not args.no_rppg,
        "window_seconds": 30,
        "window_stride_seconds": 10,
        "elapsed_seconds": elapsed,
        "subjects_filter": args.subjects,
        "outputs": [row["output_npz"] for row in manifest],
    }
    (out / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"\nFinished in {elapsed}s")
    print(f"Manifest: {out / 'manifest.csv'}")
    print(f"QC report: {out / 'qc.csv'}")
    print(f"Run metadata: {out / 'run_metadata.json'}")
    for row in qc:
        print(
            f"{row['subject_id']} {row['task_id']}: "
            f"{row['structural_status']}; "
            f"face_valid_fraction={row['face_valid_fraction']}; "
            f"warnings={row['warnings'] or 'none'}"
        )

    if any(row["structural_status"] != "PASS" for row in qc):
        raise SystemExit("QC FAILED: inspect qc.csv before using these outputs.")


if __name__ == "__main__":
    main()
