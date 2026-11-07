"""Physiology-first decision policy.

Rules
1. Reliable physiology        -> physiology is the primary prediction.
2. Reliable video (as well)   -> shown as supporting evidence only; never overrides.
3. Physiology unreliable/absent, video reliable
                              -> explicitly labelled video-only fallback.
4. Neither reliable           -> insufficient data (no label is produced).

The engine is pure: no I/O, no Streamlit, no model code. ``now`` can be
injected so tests are deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

from src.prediction_schema import (
    Branch, BranchAssessment, BranchPrediction, DataSource, DecisionMode,
    FinalDecision, parse_timestamp, utc_now,
)


@dataclass(frozen=True)
class DecisionPolicy:
    min_physiology_quality: float = 0.5
    min_video_quality: float = 0.5
    min_confidence: float = 0.0          # optional floor on the top-class probability
    max_live_age_s: float = 15.0         # staleness limit; applies to LIVE inputs only
    # Placeholder flag names: agree the real vocabulary with the model owners.
    blocking_flags: Tuple[str, ...] = ("signal_lost", "sensor_disconnected", "face_not_detected")


def assess_branch(pred: Optional[BranchPrediction], branch: Branch,
                  policy: DecisionPolicy, now: datetime) -> BranchAssessment:
    reasons: List[str] = []
    if pred is None:
        return BranchAssessment(branch, False, ["no prediction supplied"])
    if pred.branch != branch:
        return BranchAssessment(branch, False, [f"prediction is for the {pred.branch.value} branch"])
    if not pred.available:
        return BranchAssessment(branch, False, ["branch unavailable"])

    problems = pred.validate()
    if problems:
        return BranchAssessment(branch, False, [f"invalid output: {p}" for p in problems])

    min_q = policy.min_physiology_quality if branch == Branch.PHYSIOLOGY else policy.min_video_quality
    if pred.quality < min_q:
        reasons.append(f"signal quality {pred.quality:.2f} below threshold {min_q:.2f}")
    if pred.confidence < policy.min_confidence:
        reasons.append(f"confidence {pred.confidence:.2f} below minimum {policy.min_confidence:.2f}")
    hit = [f for f in pred.quality_flags if f in policy.blocking_flags]
    if hit:
        reasons.append("blocking quality flags: " + ", ".join(hit))
    if pred.source == DataSource.LIVE:
        ts = parse_timestamp(pred.timestamp)
        if ts is None:
            reasons.append("live prediction has no timestamp")
        elif (now - ts).total_seconds() > policy.max_live_age_s:
            reasons.append(f"live prediction is stale (>{policy.max_live_age_s:g}s old)")
    return BranchAssessment(branch, not reasons, reasons)


def _combined_source(*preds: BranchPrediction) -> Tuple[DataSource, List[str]]:
    sources = {p.source for p in preds}
    if sources == {DataSource.LIVE}:
        return DataSource.LIVE, []
    warnings = []
    if len(sources) > 1:
        warnings.append("Inputs mix live and replay data; the result is labelled REPLAY.")
    return DataSource.REPLAY, warnings


def decide(physiology: Optional[BranchPrediction], video: Optional[BranchPrediction],
           policy: Optional[DecisionPolicy] = None, now: Optional[datetime] = None) -> FinalDecision:
    policy = policy or DecisionPolicy()
    now = now or utc_now()
    phys_a = assess_branch(physiology, Branch.PHYSIOLOGY, policy, now)
    vid_a = assess_branch(video, Branch.VIDEO, policy, now)
    stamp = now.isoformat()
    common = dict(timestamp=stamp, physiology_assessment=phys_a, video_assessment=vid_a)

    if phys_a.reliable:
        used = [physiology] + ([video] if vid_a.reliable else [])
        source, warnings = _combined_source(*used)
        text = (f"Physiology determined the result ({physiology.label}, "
                f"{physiology.confidence:.0%}).")
        supporting = None
        if vid_a.reliable:
            supporting = video
            if video.label == physiology.label:
                text += f" Video agrees ({video.label}, {video.confidence:.0%}) as supporting evidence."
            else:
                text += (f" Video suggests '{video.label}' ({video.confidence:.0%}); "
                         "it is supporting evidence only and does not change the result.")
                warnings.append("Physiology and video disagree.")
        else:
            text += " Video was not used: " + "; ".join(vid_a.reasons) + "."
        return FinalDecision(
            mode=DecisionMode.PHYSIOLOGY_PRIMARY, determining_branch=Branch.PHYSIOLOGY,
            primary_label=physiology.label, primary_probabilities=dict(physiology.probabilities),
            source=source, explanation=text, supporting_video=supporting,
            warnings=warnings, **common)

    if vid_a.reliable:
        source, warnings = _combined_source(video)
        text = ("Physiology is not usable (" + "; ".join(phys_a.reasons) + "). "
                f"Result comes from the video branch only ({video.label}, {video.confidence:.0%}).")
        return FinalDecision(
            mode=DecisionMode.VIDEO_FALLBACK, determining_branch=Branch.VIDEO,
            primary_label=video.label, primary_probabilities=dict(video.probabilities),
            source=source, explanation=text,
            banner="VIDEO-ONLY FALLBACK: physiology unavailable or unreliable, so the result comes from video alone.",
            warnings=warnings, **common)

    text = ("Insufficient data. Physiology: " + "; ".join(phys_a.reasons) +
            ". Video: " + "; ".join(vid_a.reasons) + ".")
    return FinalDecision(
        mode=DecisionMode.INSUFFICIENT_DATA, determining_branch=None, primary_label=None,
        primary_probabilities={}, source=DataSource.REPLAY if any(
            p is not None and p.source == DataSource.REPLAY for p in (physiology, video)
        ) else DataSource.LIVE, explanation=text, **common)
