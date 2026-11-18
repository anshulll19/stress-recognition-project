import importlib.util
import json
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import neurokit2

ROOT = Path("data/ubfc_test_upload")
FEATURE_FILE = Path("results/neurokit2_v2/fused_features_v2.csv")
MODULE_FILE = Path(
    "physiological_pipeline/03_NeuroKit_V2/neurokit_eda_v2.py"
)
OUTPUT = Path("results/neurokit2_v2/validation")

FEATURES = [
    "eda_mean", "eda_std", "eda_median", "eda_iqr",
    "eda_range", "eda_tonic_mean", "eda_tonic_std",
    "eda_tonic_slope", "eda_phasic_mean", "eda_phasic_std",
    "eda_scr_valid_event_count",
    "eda_scr_onsets_diagnostic",
    "eda_scr_peaks_diagnostic",
    "eda_raw_std", "eda_unique_fraction",
    "eda_flat_step_fraction",
    "eda_median_abs_difference",
    "eda_max_abs_jump", "eda_jump_ratio",
]

RTOL = 1e-5
ATOL = 1e-8


def load_extractor():
    spec = importlib.util.spec_from_file_location(
        "eda_module", MODULE_FILE
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    module = load_extractor()
    expected_df = pd.read_csv(FEATURE_FILE)

    files = sorted(ROOT.rglob("eda_s*_T*.csv"))
    records = []
    feature_errors = []

    seen = set()

    for path in files:
        match = re.search(r"eda_s(\d+)_T(\d+)\.csv$", path.name)
        if not match:
            continue

        participant = int(match.group(1))
        task = int(match.group(2))
        key = (participant, task)

        status = "PASS"
        reason = ""
        max_recording_diff = 0.0
        actual_count = 0

        if key in seen:
            status = "FAIL"
            reason = "Duplicate raw recording"
        seen.add(key)

        expected = expected_df[
            (expected_df["participant"] == participant)
            & (expected_df["task"] == task)
        ].sort_values("start_s").reset_index(drop=True)

        if expected.empty:
            status = "FAIL"
            reason = "No reference feature rows"

        try:
            raw = pd.read_csv(
                path, header=None
            ).iloc[:, 0].to_numpy(dtype=float)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                actual = pd.DataFrame(
                    module.extract_eda_features(raw)
                )

            actual_count = len(actual)

            if status == "PASS" and len(actual) != len(expected):
                status = "FAIL"
                reason = (
                    f"Window mismatch: actual={len(actual)}, "
                    f"expected={len(expected)}"
                )

            if status == "PASS":
                for feature in FEATURES:
                    a = pd.to_numeric(
                        actual[feature], errors="coerce"
                    ).to_numpy(dtype=float)

                    e = pd.to_numeric(
                        expected[feature], errors="coerce"
                    ).to_numpy(dtype=float)

                    if not np.array_equal(np.isnan(a), np.isnan(e)):
                        status = "FAIL"
                        reason = f"NaN pattern mismatch: {feature}"
                        break

                    valid = np.isfinite(a) & np.isfinite(e)

                    if valid.any():
                        diff = float(
                            np.max(np.abs(a[valid] - e[valid]))
                        )
                        max_recording_diff = max(
                            max_recording_diff, diff
                        )

                        if not np.allclose(
                            a[valid], e[valid],
                            rtol=RTOL, atol=ATOL
                        ):
                            status = "FAIL"
                            reason = f"Numerical mismatch: {feature}"
                            feature_errors.append({
                                "participant": participant,
                                "task": task,
                                "feature": feature,
                                "max_absolute_difference": diff,
                            })
                            break

            warning_messages = sorted({
                str(w.message) for w in caught
            })

        except Exception as exc:
            status = "FAIL"
            reason = str(exc)
            warning_messages = []

        records.append({
            "participant": participant,
            "task": task,
            "file": str(path),
            "expected_windows": len(expected),
            "actual_windows": actual_count,
            "max_absolute_difference": max_recording_diff,
            "status": status,
            "reason": reason,
            "warnings": " | ".join(warning_messages),
        })

    audit = pd.DataFrame(records)
    audit.to_csv(
        OUTPUT / "eda_feature_parity_recordings.csv",
        index=False
    )

    pd.DataFrame(feature_errors).to_csv(
        OUTPUT / "eda_feature_parity_errors.csv",
        index=False
    )

    passed = int((audit["status"] == "PASS").sum())
    failed = int((audit["status"] == "FAIL").sum())

    summary = {
        "validation": "EDA feature parity",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "raw_recordings_found": len(files),
        "unique_participant_task_pairs": len(seen),
        "passed_recordings": passed,
        "failed_recordings": failed,
        "total_extracted_windows": int(
            audit["actual_windows"].sum()
        ),
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "relative_tolerance": RTOL,
        "absolute_tolerance": ATOL,
        "neurokit2_version": neurokit2.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "result": "PASS" if failed == 0 and passed == 168 else "FAIL",
    }

    (OUTPUT / "eda_feature_parity_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print(json.dumps(summary, indent=2))
    print("\nAudit saved to:", OUTPUT)


if __name__ == "__main__":
    main()
