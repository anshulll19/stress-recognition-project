"""Common prediction output format shared by the physiology branch, the video
branch and the decision engine.

Both model owners return a ``BranchPrediction``; nothing else in the pipeline
needs to know how a model works internally. The label is always derived from
the probabilities, so the two can never disagree.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

PROB_SUM_TOLERANCE = 0.02


class Branch(str, Enum):
    PHYSIOLOGY = "physiology"
    VIDEO = "video"


class DataSource(str, Enum):
    """Where the input signal came from. Replay must never be shown as live."""
    LIVE = "live"
    REPLAY = "replay"  # prerecorded data (e.g. UBFC-Phys recordings)


class DecisionMode(str, Enum):
    PHYSIOLOGY_PRIMARY = "physiology_primary"
    VIDEO_FALLBACK = "video_only_fallback"
    INSUFFICIENT_DATA = "insufficient_data"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-8601 string; naive values are treated as UTC."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


@dataclass
class BranchPrediction:
    branch: Branch
    available: bool
    source: DataSource
    probabilities: Dict[str, float] = field(default_factory=dict)
    quality: Optional[float] = None          # 0..1, higher is better
    quality_flags: List[str] = field(default_factory=list)
    timestamp: Optional[str] = None          # ISO-8601; capture time of the input
    model_version: Optional[str] = None
    window_ref: Optional[str] = None

    @property
    def label(self) -> Optional[str]:
        if not self.probabilities:
            return None
        return max(self.probabilities, key=self.probabilities.get)

    @property
    def confidence(self) -> Optional[float]:
        return self.probabilities.get(self.label) if self.label else None

    def validate(self) -> List[str]:
        """Return a list of schema problems (empty list = valid)."""
        problems: List[str] = []
        if not self.available:
            return problems
        if not self.probabilities:
            problems.append("available but no probabilities supplied")
        else:
            for k, v in self.probabilities.items():
                if not isinstance(v, (int, float)) or not math.isfinite(v) or not 0.0 <= v <= 1.0:
                    problems.append(f"probability for '{k}' is not a finite value in [0, 1]")
            vals = [v for v in self.probabilities.values()
                    if isinstance(v, (int, float)) and math.isfinite(v)]
            if vals and abs(sum(vals) - 1.0) > PROB_SUM_TOLERANCE:
                problems.append(f"probabilities sum to {sum(vals):.3f}, expected 1.0")
        if self.quality is None or not math.isfinite(self.quality) or not 0.0 <= self.quality <= 1.0:
            problems.append("quality must be a finite value in [0, 1]")
        if self.timestamp and parse_timestamp(self.timestamp) is None:
            problems.append("timestamp is not valid ISO-8601")
        return problems

    def to_dict(self) -> Dict[str, Any]:
        return {
            "branch": self.branch.value,
            "available": self.available,
            "source": self.source.value,
            "label": self.label,
            "confidence": self.confidence,
            "probabilities": dict(self.probabilities),
            "quality": self.quality,
            "quality_flags": list(self.quality_flags),
            "timestamp": self.timestamp,
            "model_version": self.model_version,
            "window_ref": self.window_ref,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BranchPrediction":
        return cls(
            branch=Branch(d["branch"]),
            available=bool(d.get("available", True)),
            source=DataSource(d.get("source", "replay")),
            probabilities=dict(d.get("probabilities") or {}),
            quality=d.get("quality"),
            quality_flags=list(d.get("quality_flags") or []),
            timestamp=d.get("timestamp"),
            model_version=d.get("model_version"),
            window_ref=d.get("window_ref"),
        )


@dataclass
class BranchAssessment:
    """Result of the reliability check for one branch."""
    branch: Branch
    reliable: bool
    reasons: List[str] = field(default_factory=list)  # why it failed (empty if reliable)

    def to_dict(self) -> Dict[str, Any]:
        return {"branch": self.branch.value, "reliable": self.reliable, "reasons": list(self.reasons)}


@dataclass
class FinalDecision:
    mode: DecisionMode
    determining_branch: Optional[Branch]
    primary_label: Optional[str]
    primary_probabilities: Dict[str, float]
    source: DataSource                       # LIVE only if every branch used is live
    timestamp: str                           # when the decision was made
    explanation: str
    banner: Optional[str] = None             # e.g. video-only fallback warning
    supporting_video: Optional[BranchPrediction] = None
    physiology_assessment: Optional[BranchAssessment] = None
    video_assessment: Optional[BranchAssessment] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def is_fallback(self) -> bool:
        return self.mode == DecisionMode.VIDEO_FALLBACK

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "determining_branch": self.determining_branch.value if self.determining_branch else None,
            "primary_label": self.primary_label,
            "primary_probabilities": dict(self.primary_probabilities),
            "source": self.source.value,
            "timestamp": self.timestamp,
            "explanation": self.explanation,
            "banner": self.banner,
            "supporting_video": self.supporting_video.to_dict() if self.supporting_video else None,
            "physiology_assessment": self.physiology_assessment.to_dict() if self.physiology_assessment else None,
            "video_assessment": self.video_assessment.to_dict() if self.video_assessment else None,
            "warnings": list(self.warnings),
        }
