"""Bridge: one row of results/fusion_lstm/aligned_oof_predictions.csv ->
(physiology BranchPrediction, video BranchPrediction) for the decision engine.

The aligned file holds out-of-fold, time-synchronised predictions for 18
participants (prerecorded UBFC-Phys), so every prediction is REPLAY.
Physiology uses ``p_physio``; video uses the clean upper-face model by default
(``p_video_upperface``), not the speech-confounded full-face model.
"""
from __future__ import annotations

from typing import Any, Tuple

from src.adapters import _probs, unavailable
from src.prediction_schema import Branch, BranchPrediction, DataSource, utc_now

# Maps the physiology inference quality flag (src/physiology_inference.py) to a score.
QUALITY_FLAG_SCORES = {"OK": 1.0, "SUSPECT": 0.4, "DROPOUT": 0.0}
LOW_DETECTION_FRACTION = 0.8


def window_ref(row: Any) -> str:
    return (f"{row['subject_id']}/T{int(row['task'])}/"
            f"{float(row['window_start_sec']):g}-{float(row['window_end_sec']):g}s")


def predictions_from_aligned_row(row: Any, video_column: str = "p_video_upperface",
                                 simulate_physiology_dropout: bool = False,
                                 simulate_video_loss: bool = False
                                 ) -> Tuple[BranchPrediction, BranchPrediction]:
    ref = window_ref(row)
    stamp = utc_now().isoformat()

    if simulate_physiology_dropout:
        phys = unavailable(Branch.PHYSIOLOGY, ref, "signal_lost")
    else:
        phys = BranchPrediction(
            branch=Branch.PHYSIOLOGY, available=True, source=DataSource.REPLAY,
            probabilities=_probs(row["p_physio"]),
            quality=QUALITY_FLAG_SCORES["OK"],
            # The aligned file has no per-window physiology quality, so this is
            # an assumption of nominal quality, stated explicitly in the flag.
            quality_flags=["quality_not_measured_in_replay"],
            timestamp=stamp, model_version="lstm_oof", window_ref=ref)

    if simulate_video_loss:
        vid = unavailable(Branch.VIDEO, ref, "face_not_detected")
    else:
        frac = float(row.get("detection_fraction", 1.0))
        flags = []
        if frac <= 0.0:
            flags.append("face_not_detected")
        elif frac < LOW_DETECTION_FRACTION:
            flags.append("low_face_detection")
        vid = BranchPrediction(
            branch=Branch.VIDEO, available=True, source=DataSource.REPLAY,
            probabilities=_probs(row[video_column]), quality=frac, quality_flags=flags,
            timestamp=stamp, model_version=video_column, window_ref=ref)
    return phys, vid
