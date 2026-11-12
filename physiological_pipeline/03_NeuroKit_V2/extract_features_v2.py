
"""
NeuroKit2 Physiological Feature Extraction - Version 2

Dataset: UBFC-Phys
Participants: 56
Tasks: T1, T2, T3
Expected recordings: 168
Window length: 30 seconds
Window stride: 5 seconds

Outputs:
    results/neurokit2_v2/bvp_features_v2.csv
    results/neurokit2_v2/eda_features_v2.csv
    results/neurokit2_v2/fused_features_v2.csv
    results/neurokit2_v2/extraction_quality_v2.csv
    results/neurokit2_v2/extraction_summary_v2.json
"""

import sys
import json
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = PROJECT_ROOT / "data" / "ubfc_test_upload"

OUTPUT_ROOT = PROJECT_ROOT / "results" / "neurokit2_v2"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------
# LOAD V2 MODULES
# --------------------------------------------------

def load_module(name):

    path = Path(__file__).parent / f"{name}.py"

    if not path.exists():
        raise FileNotFoundError(
            f"Required module not found: {path}"
        )

    spec = importlib.util.spec_from_file_location(name, path)

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module


bvp_module = load_module("neurokit_bvp_v2")

eda_module = load_module("neurokit_eda_v2")


# --------------------------------------------------
# LOAD BASELINE WINDOW METADATA
# --------------------------------------------------

baseline_path = (
    PROJECT_ROOT
    / "results"
    / "physiology"
    / "physiology_features_labeled.csv"
)

if not baseline_path.exists():
    raise FileNotFoundError(baseline_path)

baseline = pd.read_csv(baseline_path)

required_columns = {
    "participant",
    "task",
    "start_s",
    "end_s",
    "label",
}

missing_columns = required_columns - set(baseline.columns)

if missing_columns:
    raise ValueError(
        f"Baseline metadata missing columns: {missing_columns}"
    )

baseline["participant"] = baseline["participant"].astype(int)
baseline["task"] = baseline["task"].astype(int)

baseline = baseline.sort_values(
    ["participant", "task", "start_s"]
).reset_index(drop=True)

baseline_keys = [
    "participant",
    "task",
    "start_s",
    "end_s",
    "label",
]

if baseline.duplicated(baseline_keys).any():
    raise ValueError("Duplicate baseline window metadata found.")


# --------------------------------------------------
# INDEX RAW FILES
# --------------------------------------------------

bvp_file_index = {}
eda_file_index = {}

for path in DATA_ROOT.rglob("bvp_*.csv"):
    bvp_file_index.setdefault(path.name, []).append(path)

for path in DATA_ROOT.rglob("eda_*.csv"):
    eda_file_index.setdefault(path.name, []).append(path)


# --------------------------------------------------
# FEATURE EXTRACTION
# --------------------------------------------------

bvp_rows = []
eda_rows = []
quality_rows = []

expected_recordings = 56 * 3

for participant in range(1, 57):

    for task in range(1, 4):

        prefix = f"s{participant}_T{task}"

        bvp_filename = f"bvp_{prefix}.csv"
        eda_filename = f"eda_{prefix}.csv"

        bvp_matches = bvp_file_index.get(bvp_filename, [])
        eda_matches = eda_file_index.get(eda_filename, [])

        # Check missing and duplicate recordings.
        if len(bvp_matches) != 1 or len(eda_matches) != 1:

            quality_rows.append({
                "participant": participant,
                "task": task,
                "status": "missing_or_duplicate_file",
                "bvp_matches": len(bvp_matches),
                "eda_matches": len(eda_matches),
            })

            continue

        try:

            # ------------------------------------------
            # LOAD RAW SIGNALS
            # ------------------------------------------

            bvp = pd.read_csv(
                bvp_matches[0],
                header=None
            ).iloc[:, 0].to_numpy(dtype=float)

            eda = pd.read_csv(
                eda_matches[0],
                header=None
            ).iloc[:, 0].to_numpy(dtype=float)

            # ------------------------------------------
            # EXTRACT FEATURES
            # ------------------------------------------

            bvp_features = bvp_module.extract_bvp_features(bvp)

            eda_features = eda_module.extract_eda_features(eda)

            # ------------------------------------------
            # VALIDATE BASELINE METADATA
            # ------------------------------------------

            expected = baseline.loc[
                (baseline["participant"] == participant)
                & (baseline["task"] == task),
                ["start_s", "end_s", "label"]
            ].copy()

            expected = expected.sort_values(
                "start_s"
            ).reset_index(drop=True)

            if len(expected) == 0:
                raise ValueError("No baseline metadata found.")

            if (
                len(bvp_features) != len(expected)
                or len(eda_features) != len(expected)
            ):
                raise ValueError(
                    f"Window mismatch: baseline={len(expected)}, "
                    f"BVP={len(bvp_features)}, "
                    f"EDA={len(eda_features)}"
                )

            # ------------------------------------------
            # ATTACH METADATA
            # ------------------------------------------

            for i, (bvp_row, eda_row) in enumerate(
                zip(bvp_features, eda_features)
            ):

                meta = expected.iloc[i]

                common = {
                    "participant": participant,
                    "task": task,
                    "start_s": float(meta["start_s"]),
                    "end_s": float(meta["end_s"]),
                    "label": int(meta["label"]),
                }

                bvp_rows.append({
                    **common,
                    **bvp_row,
                })

                eda_rows.append({
                    **common,
                    **eda_row,
                })

            # ------------------------------------------
            # RECORDING QUALITY SUMMARY
            # ------------------------------------------

            eda_df = pd.DataFrame(eda_features)

            quality_rows.append({

                "participant": participant,
                "task": task,
                "status": "success",

                "bvp_samples": len(bvp),
                "eda_samples": len(eda),

                "bvp_windows": len(bvp_features),
                "eda_windows": len(eda_features),

                "eda_mean_unique_fraction": float(
                    eda_df["eda_unique_fraction"].mean()
                ),

                "eda_mean_flat_step_fraction": float(
                    eda_df["eda_flat_step_fraction"].mean()
                ),

                "eda_mean_jump_ratio": float(
                    eda_df["eda_jump_ratio"].mean()
                ),

                "eda_scr_event_windows": int(
                    (eda_df["eda_scr_valid_event_count"] > 0).sum()
                ),

            })

        except Exception as exc:

            quality_rows.append({
                "participant": participant,
                "task": task,
                "status": "error",
                "error": str(exc),
            })

        completed = len(quality_rows)

        if completed % 10 == 0 or completed == expected_recordings:

            print(
                f"Processed {completed}/{expected_recordings} recordings"
            )


# --------------------------------------------------
# CREATE DATAFRAMES
# --------------------------------------------------

bvp_df = pd.DataFrame(bvp_rows)
eda_df = pd.DataFrame(eda_rows)
quality_df = pd.DataFrame(quality_rows)


# --------------------------------------------------
# SAVE INDIVIDUAL FEATURES AND QUALITY
# --------------------------------------------------

if not bvp_df.empty:

    bvp_df.to_csv(
        OUTPUT_ROOT / "bvp_features_v2.csv",
        index=False
    )

if not eda_df.empty:

    eda_df.to_csv(
        OUTPUT_ROOT / "eda_features_v2.csv",
        index=False
    )

quality_df.to_csv(
    OUTPUT_ROOT / "extraction_quality_v2.csv",
    index=False
)


# --------------------------------------------------
# FUSE BVP AND EDA
# --------------------------------------------------

fused = pd.DataFrame()

if not bvp_df.empty and not eda_df.empty:

    fused = bvp_df.merge(
        eda_df,
        on=baseline_keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_bvp", "_eda"),
    )

    fused.to_csv(
        OUTPUT_ROOT / "fused_features_v2.csv",
        index=False
    )


# --------------------------------------------------
# VALIDATE COMPLETE EXTRACTION
# --------------------------------------------------

successful = int(
    (quality_df["status"] == "success").sum()
)

failed = expected_recordings - successful

expected_windows = len(baseline)

summary = {

    "pipeline": "NeuroKit2_V2",

    "expected_participants": 56,

    "actual_participants": int(
        fused["participant"].nunique()
    ) if not fused.empty else 0,

    "expected_recordings": expected_recordings,

    "successful_recordings": successful,

    "failed_recordings": failed,

    "bvp_windows": len(bvp_df),

    "eda_windows": len(eda_df),

    "fused_windows": len(fused),

    "baseline_windows": expected_windows,

    "feature_count_bvp": (
        len([
            c for c in bvp_df.columns
            if c not in baseline_keys
        ])
        if not bvp_df.empty else 0
    ),

    "feature_count_eda": (
        len([
            c for c in eda_df.columns
            if c not in baseline_keys
        ])
        if not eda_df.empty else 0
    ),

    "complete_extraction": bool(
        successful == expected_recordings
        and len(fused) == expected_windows
        and fused["participant"].nunique() == 56
    ) if not fused.empty else False,

}

with open(
    OUTPUT_ROOT / "extraction_summary_v2.json",
    "w"
) as file:

    json.dump(summary, file, indent=4)


# --------------------------------------------------
# FINAL REPORT
# --------------------------------------------------

print("\n" + "=" * 65)

print("NEUROKIT2 VERSION 2 EXTRACTION SUMMARY")

print("=" * 65)

print(json.dumps(summary, indent=4))

print("\nRecording status:")

print(
    quality_df["status"]
    .value_counts()
    .to_string()
)

if not fused.empty:

    print("\nFused dataset shape:", fused.shape)

    print(
        "\nTask and label distribution:"
    )

    print(
        fused.groupby(["task", "label"]).size()
    )

print("\nOutput directory:", OUTPUT_ROOT)

if not summary["complete_extraction"]:

    print(
        "\nWARNING: Extraction is incomplete. "
        "Inspect extraction_quality_v2.csv."
    )

else:

    print("\nALL EXTRACTION VALIDATION CHECKS PASSED.")

