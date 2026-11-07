"""
physiological_streaming.py
===========================

Sliding window generator and real-time buffer management for physiological signals.
Enforces the standardized 30-second window and 5-second stride.
Conducts per-window signal quality and dropout diagnostics, guaranteeing that
sensor dropouts and unreliable windows are never misclassified as non-stress.
"""

from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional, Tuple, Union

import numpy as np

try:
    from .physiology_inference import UBFC_FEATURES, UBFCInferenceEngine
except ImportError:
    from physiology_inference import UBFC_FEATURES, UBFCInferenceEngine


DEFAULT_WINDOW_SECONDS: float = 30.0
DEFAULT_STRIDE_SECONDS: float = 5.0
BVP_FS: float = 64.0
EDA_FS: float = 4.0

BVP_WINDOW_SAMPLES: int = int(BVP_FS * DEFAULT_WINDOW_SECONDS)   # 1920
EDA_WINDOW_SAMPLES: int = int(EDA_FS * DEFAULT_WINDOW_SECONDS)   # 120
BVP_STRIDE_SAMPLES: int = int(BVP_FS * DEFAULT_STRIDE_SECONDS)   # 320
EDA_STRIDE_SAMPLES: int = int(EDA_FS * DEFAULT_STRIDE_SECONDS)   # 20


@dataclass
class PhysiologicalWindow:
    """A synchronous 30-second multimodal physiological window with quality metadata."""
    window_index: int
    start_s: float
    end_s: float
    bvp_samples: np.ndarray
    eda_samples: np.ndarray
    is_reliable: bool
    quality_flag: str                           # 'OK', 'DROPOUT', 'UNRELIABLE'
    quality_reason: Optional[str] = None
    features: Optional[Dict[str, float]] = None

    def to_dict(self, include_raw: bool = False) -> dict:
        data = {
            "window_index": self.window_index,
            "start_s": round(self.start_s, 2),
            "end_s": round(self.end_s, 2),
            "is_reliable": self.is_reliable,
            "quality_flag": self.quality_flag,
            "quality_reason": self.quality_reason,
            "features": self.features,
        }
        if include_raw:
            data["bvp_samples"] = self.bvp_samples.tolist()
            data["eda_samples"] = self.eda_samples.tolist()
        return data


def assess_window_reliability(
    bvp_w: np.ndarray,
    eda_w: np.ndarray,
    min_bvp_std: float = 1e-6,
    min_eda_std: float = 1e-6,
) -> Tuple[bool, str, Optional[str]]:
    """
    Evaluate sensor quality and contact integrity for a single physiological window.

    Returns:
    - is_reliable (bool): False if sensor dropout, flatline, or non-finite values.
    - quality_flag (str): 'OK', 'DROPOUT', or 'UNRELIABLE'.
    - reason (Optional[str]): Diagnostic message.
    """
    if not np.isfinite(bvp_w).all() or not np.isfinite(eda_w).all():
        return False, "UNRELIABLE", "Window contains NaN or infinite values."

    b_std = float(np.std(bvp_w))
    e_std = float(np.std(eda_w))

    if b_std < min_bvp_std and e_std < min_eda_std:
        return False, "DROPOUT", "Both BVP and EDA signals are flatline (simultaneous sensor detachment)."
    if b_std < min_bvp_std:
        return False, "DROPOUT", f"BVP flatline detected (std={b_std:.2e} < {min_bvp_std:.0e})."
    if e_std < min_eda_std:
        return False, "DROPOUT", f"EDA flatline detected (std={e_std:.2e} < {min_eda_std:.0e})."

    # Amplitude range check (sensor saturation or disconnection artifact)
    b_ptp = float(np.ptp(bvp_w))
    e_ptp = float(np.ptp(eda_w))
    if b_ptp < 1e-5:
        return False, "DROPOUT", "BVP peak-to-peak amplitude near zero."

    return True, "OK", None


def slice_physiological_windows(
    bvp_signal: np.ndarray,
    eda_signal: np.ndarray,
    bvp_fs: float = BVP_FS,
    eda_fs: float = EDA_FS,
    window_s: float = DEFAULT_WINDOW_SECONDS,
    stride_s: float = DEFAULT_STRIDE_SECONDS,
    extract_features: bool = True,
) -> List[PhysiologicalWindow]:
    """
    Slice continuous BVP and EDA recordings into synchronous sliding windows.

    Parameters
    ----------
    bvp_signal : np.ndarray
        Continuous BVP signal @ bvp_fs (nominal 64 Hz).
    eda_signal : np.ndarray
        Continuous EDA signal @ eda_fs (nominal 4 Hz).
    window_s : float
        Window duration in seconds (default 30.0).
    stride_s : float
        Stride duration in seconds (default 5.0).
    extract_features : bool
        If True, extracts the 12 UBFC features for reliable windows.
    """
    b = np.asarray(bvp_signal, dtype=np.float64).ravel()
    e = np.asarray(eda_signal, dtype=np.float64).ravel()

    dur_b = b.size / bvp_fs
    dur_e = e.size / eda_fs
    max_duration = min(dur_b, dur_e)

    w_bvp = int(round(window_s * bvp_fs))
    w_eda = int(round(window_s * eda_fs))
    s_bvp = int(round(stride_s * bvp_fs))
    s_eda = int(round(stride_s * eda_fs))

    windows: List[PhysiologicalWindow] = []
    window_idx = 0
    t0 = 0.0

    while t0 + window_s <= max_duration + 1e-5:
        t1 = t0 + window_s
        b_start = int(round(t0 * bvp_fs))
        b_end = b_start + w_bvp
        e_start = int(round(t0 * eda_fs))
        e_end = e_start + w_eda

        if b_end > b.size or e_end > e.size:
            break

        b_sub = b[b_start:b_end]
        e_sub = e[e_start:e_end]

        is_rel, q_flag, reason = assess_window_reliability(b_sub, e_sub)

        feats = None
        if is_rel and extract_features:
            try:
                feats = UBFCInferenceEngine.extract_features(b_sub, e_sub)
            except Exception as ex:
                is_rel = False
                q_flag = "UNRELIABLE"
                reason = f"Feature extraction failed: {str(ex)}"

        win = PhysiologicalWindow(
            window_index=window_idx,
            start_s=round(t0, 2),
            end_s=round(t1, 2),
            bvp_samples=b_sub,
            eda_samples=e_sub,
            is_reliable=is_rel,
            quality_flag=q_flag,
            quality_reason=reason,
            features=feats,
        )
        windows.append(win)
        window_idx += 1
        t0 += stride_s

    return windows

