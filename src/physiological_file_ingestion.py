"""
physiological_file_ingestion.py
================================

Robust file and buffer ingestion module for physiological signals (BVP @ 64 Hz, EDA @ 4 Hz).
Parses raw CSV/TXT uploads, validates signal length, numerical stability, and aligns sampling rates.

Preserves research pipeline duration requirements:
- Task 1 (Rest Baseline): minimum 180.0 seconds (11,520 BVP samples, 720 EDA samples -> 31 overlapping windows).
- Task 2/3 (Active Tasks): minimum 40.0 seconds (2,560 BVP samples, 160 EDA samples -> at least 3 sliding windows).
"""

from dataclasses import dataclass, field
import io
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


BVP_NOMINAL_FS: float = 64.0
EDA_NOMINAL_FS: float = 4.0

MIN_TASK_DURATION_S: float = 40.0         # 40s active task minimum (produces >= 3 windows @ 30s/5s stride)
MIN_BASELINE_DURATION_S: float = 180.0     # 180s resting baseline protocol (produces 31 overlapping windows)


class IngestionError(ValueError):
    """Raised when uploaded physiological file cannot be parsed or validated."""
    pass


@dataclass
class IngestedPhysiologyData:
    """Standardized representation of ingested multimodal physiological signals."""
    bvp: np.ndarray
    eda: np.ndarray
    bvp_fs: float = BVP_NOMINAL_FS
    eda_fs: float = EDA_NOMINAL_FS
    duration_s: float = 0.0
    n_bvp_samples: int = 0
    n_eda_samples: int = 0
    participant_id: Optional[str] = None
    task_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        return {
            "participant_id": self.participant_id,
            "task_id": self.task_id,
            "duration_s": round(self.duration_s, 2),
            "bvp_samples": self.n_bvp_samples,
            "eda_samples": self.n_eda_samples,
            "bvp_fs": self.bvp_fs,
            "eda_fs": self.eda_fs,
            "metadata": self.metadata,
        }


def parse_raw_signal(
    raw_content: Union[bytes, str, io.StringIO, io.BytesIO, List[float], np.ndarray],
    signal_name: str = "signal",
) -> np.ndarray:
    """
    Parse arbitrary raw signal source into 1D float64 numpy array.

    Supports:
    - Raw CSV/TXT strings or bytes (single column, comma/tab/space/newline separated)
    - CSV with or without header
    - Lists of floats or numpy arrays
    """
    if isinstance(raw_content, np.ndarray):
        arr = np.asarray(raw_content, dtype=np.float64).ravel()
    elif isinstance(raw_content, list):
        if len(raw_content) == 0:
            raise IngestionError(f"{signal_name} list is empty.")
        try:
            arr = np.asarray(raw_content, dtype=np.float64).ravel()
        except (ValueError, TypeError) as e:
            raise IngestionError(f"Could not convert {signal_name} list to floats: {e}")
    elif isinstance(raw_content, (bytes, str, io.StringIO, io.BytesIO)):
        if isinstance(raw_content, bytes):
            text = raw_content.decode("utf-8", errors="replace")
        elif isinstance(raw_content, (io.StringIO, io.BytesIO)):
            content = raw_content.read()
            text = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else content
        else:
            text = raw_content

        text = text.strip()
        if not text:
            raise IngestionError(f"{signal_name} data is empty.")

        # Try pandas read_csv first
        try:
            df = pd.read_csv(io.StringIO(text), header=None, comment="#")
            # If header row had column names (strings), first row might be string
            first_val = str(df.iloc[0, 0]).strip().lower()
            if not re.match(r"^-?\d+(\.\d+)?([eE][-+]?\d+)?$", first_val):
                df = pd.read_csv(io.StringIO(text), header=0, comment="#")
            col = df.iloc[:, 0]
            arr = pd.to_numeric(col, errors="coerce").to_numpy(dtype=np.float64)
        except Exception:
            # Fallback to regex tokenization of floating point values
            tokens = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", text)
            if not tokens:
                raise IngestionError(f"Could not parse any numeric values from {signal_name} content.")
            arr = np.array([float(t) for t in tokens], dtype=np.float64)
    else:
        raise IngestionError(f"Unsupported data format for {signal_name}: {type(raw_content)}")

    if arr.size == 0:
        raise IngestionError(f"{signal_name} parsed array contains 0 samples.")

    # Check NaN proportion
    nan_mask = np.isnan(arr) | np.isinf(arr)
    nan_count = int(np.sum(nan_mask))
    if nan_count > 0:
        if nan_count == arr.size:
            raise IngestionError(f"{signal_name} contains only NaN or infinite values.")
        if nan_count / arr.size > 0.05:
            raise IngestionError(
                f"{signal_name} contains excessive non-finite values ({nan_count}/{arr.size} = {nan_count/arr.size:.1%})."
            )
        # Interpolate minor NaNs
        arr = pd.Series(arr).interpolate(limit_direction="both").to_numpy(dtype=np.float64)

    return arr


def ingest_bvp_eda(
    bvp_data: Union[bytes, str, List[float], np.ndarray],
    eda_data: Union[bytes, str, List[float], np.ndarray],
    participant_id: Optional[str] = None,
    task_id: Optional[str] = None,
    bvp_fs: float = BVP_NOMINAL_FS,
    eda_fs: float = EDA_NOMINAL_FS,
    min_duration_s: Optional[float] = None,
    max_duration_diff_s: float = 15.0,
) -> IngestedPhysiologyData:
    """
    Ingest and align BVP and EDA signals.

    Duration enforcement:
    - If task_id == "T1": defaults to 180.0 seconds (original UBFC resting baseline protocol).
    - Other tasks (T2/T3): defaults to 40.0 seconds (preserves earlier physiological minimum,
      ensuring at least 3 sliding windows @ 30s window / 5s stride).
    """
    tid = task_id.upper().strip() if task_id else None

    if min_duration_s is None:
        min_duration_s = MIN_BASELINE_DURATION_S if tid == "T1" else MIN_TASK_DURATION_S

    min_bvp_samples = int(np.floor(bvp_fs * min_duration_s))
    min_eda_samples = int(np.floor(eda_fs * min_duration_s))

    bvp = parse_raw_signal(bvp_data, signal_name="BVP")
    eda = parse_raw_signal(eda_data, signal_name="EDA")

    if bvp.size < min_bvp_samples:
        raise IngestionError(
            f"BVP recording too short for {task_id or 'task'}: got {bvp.size} samples ({bvp.size / bvp_fs:.1f}s), "
            f"minimum required is {min_bvp_samples} samples ({min_duration_s:.0f}s @ {bvp_fs} Hz)."
        )

    if eda.size < min_eda_samples:
        raise IngestionError(
            f"EDA recording too short for {task_id or 'task'}: got {eda.size} samples ({eda.size / eda_fs:.1f}s), "
            f"minimum required is {min_eda_samples} samples ({min_duration_s:.0f}s @ {eda_fs} Hz)."
        )

    # Check overall variance
    if float(np.std(bvp)) < 1e-7:
        raise IngestionError("BVP signal is a flatline (zero variance across entire recording).")
    if float(np.std(eda)) < 1e-7:
        raise IngestionError("EDA signal is a flatline (zero variance across entire recording).")

    dur_bvp = bvp.size / bvp_fs
    dur_eda = eda.size / eda_fs
    duration_spread = abs(dur_bvp - dur_eda)

    if duration_spread > max_duration_diff_s:
        raise IngestionError(
            f"BVP and EDA duration mismatch exceeds threshold: "
            f"BVP={dur_bvp:.1f}s, EDA={dur_eda:.1f}s (difference={duration_spread:.1f}s > {max_duration_diff_s}s)."
        )

    # Truncate to common duration
    common_duration = min(dur_bvp, dur_eda)
    n_bvp = int(np.floor(common_duration * bvp_fs))
    n_eda = int(np.floor(common_duration * eda_fs))

    bvp_aligned = bvp[:n_bvp]
    eda_aligned = eda[:n_eda]

    return IngestedPhysiologyData(
        bvp=bvp_aligned,
        eda=eda_aligned,
        bvp_fs=bvp_fs,
        eda_fs=eda_fs,
        duration_s=common_duration,
        n_bvp_samples=n_bvp,
        n_eda_samples=n_eda,
        participant_id=participant_id,
        task_id=task_id,
        metadata={
            "raw_bvp_duration_s": round(dur_bvp, 2),
            "raw_eda_duration_s": round(dur_eda, 2),
            "duration_spread_s": round(duration_spread, 2),
            "enforced_min_duration_s": min_duration_s,
        },
    )
