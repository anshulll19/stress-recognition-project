"""
check_clas_block_adjacency.py
===============================

Answers the open question from the CLAS windowing fix: are Baseline/Neutral
blocks within a subject's session temporally CONTIGUOUS (one continuous
rest period, just logged as separate short blocks), or SCATTERED (brief
standalone rest breaks interspersed between task blocks)?

This determines whether stitching same-label blocks together to reach 60s
produces a genuinely continuous recording (clean) or a composite of
non-adjacent segments (needs an explicit "composite window" caveat).

Usage:
    python check_clas_block_adjacency.py --block_details_dir <path to Block_details folder>

Expects each subject's Block_details file to have columns including at
least: block_type (or similar, e.g. 'Baseline', 'Neutral', 'Math Test'),
start_time, end_time (in seconds, or any consistent unit).
ADJUST THE COLUMN NAMES BELOW to match your actual Block_details schema
once you've confirmed it (run inspect_clas.py first if unsure).
"""

import argparse
import glob
import os

import pandas as pd
import numpy as np

# --- Adjust these to match your actual Block_details column names ---
COL_TYPE = "block_type"
COL_START = "start_time"
COL_END = "end_time"
GAP_THRESHOLD_SEC = 2.0  # gaps smaller than this count as "contiguous" (allows for tiny logging jitter)

REST_LABELS = {"Baseline", "Neutral"}


def analyze_subject(df: pd.DataFrame, subject_id: str):
    df = df.sort_values(COL_START).reset_index(drop=True)
    rest_blocks = df[df[COL_TYPE].isin(REST_LABELS)].reset_index(drop=True)

    if len(rest_blocks) < 2:
        return {"subject": subject_id, "n_rest_blocks": len(rest_blocks),
                "n_contiguous_pairs": 0, "n_scattered_pairs": 0, "verdict": "too few blocks to assess"}

    gaps = []
    for i in range(len(rest_blocks) - 1):
        gap = rest_blocks.loc[i + 1, COL_START] - rest_blocks.loc[i, COL_END]
        gaps.append(gap)

    gaps = np.array(gaps)
    n_contiguous = int((gaps <= GAP_THRESHOLD_SEC).sum())
    n_scattered = int((gaps > GAP_THRESHOLD_SEC).sum())

    return {
        "subject": subject_id,
        "n_rest_blocks": len(rest_blocks),
        "n_contiguous_pairs": n_contiguous,
        "n_scattered_pairs": n_scattered,
        "median_gap_sec": float(np.median(gaps)),
        "verdict": "mostly contiguous" if n_contiguous > n_scattered else "mostly scattered"
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block_details_dir", required=True)
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.block_details_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {args.block_details_dir}")

    results = []
    for f in files:
        subject_id = os.path.basename(f).replace(".csv", "")
        df = pd.read_csv(f)
        if COL_TYPE not in df.columns:
            print(f"WARNING: {f} missing expected column '{COL_TYPE}' -- "
                  f"actual columns: {list(df.columns)}. Update COL_TYPE/COL_START/COL_END at top of script.")
            continue
        results.append(analyze_subject(df, subject_id))

    print(f"\n{'Subject':<12}{'RestBlocks':<12}{'Contiguous':<12}{'Scattered':<12}{'MedianGap(s)':<14}{'Verdict'}")
    for r in results:
        if 'median_gap_sec' in r:
            print(f"{r['subject']:<12}{r['n_rest_blocks']:<12}{r['n_contiguous_pairs']:<12}"
                  f"{r['n_scattered_pairs']:<12}{r['median_gap_sec']:<14.1f}{r['verdict']}")
        else:
            print(f"{r['subject']:<12}{r['n_rest_blocks']:<12}{'--':<12}{'--':<12}{'--':<14}{r['verdict']}")

    total_contiguous = sum(r.get('n_contiguous_pairs', 0) for r in results)
    total_scattered = sum(r.get('n_scattered_pairs', 0) for r in results)
    print(f"\n=== OVERALL: {total_contiguous} contiguous pairs vs {total_scattered} scattered pairs ===")
    if total_contiguous > total_scattered:
        print("-> Blocks are MOSTLY CONTIGUOUS. Stitching same-label blocks produces genuinely")
        print("   continuous recordings in most cases -- safe to proceed with Option 2 (stitching)")
        print("   with minimal caveating needed.")
    else:
        print("-> Blocks are MOSTLY SCATTERED. Stitching will produce COMPOSITE windows (non-adjacent")
        print("   segments concatenated). This is still usable, but must be explicitly disclosed as")
        print("   a methodology limitation -- composite windows are not equivalent to a native")
        print("   continuous 60s recording.")


if __name__ == '__main__':
    main()
