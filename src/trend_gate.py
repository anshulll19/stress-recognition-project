"""Link the decision engine to the existing temporal trend detector.

``demo.adapters.trend_adapter.TemporalTrendAdapter`` already refuses to label a
trend during cold start (< K windows) or when fewer than ``min_valid_frac`` of
the last K windows are valid. It accepts a ``valid_mask``. This module builds
that mask from the decision engine, so a window only counts as valid when the
physiology branch was reliable for it.
"""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

from src.decision_engine import DecisionPolicy, decide
from src.prediction_schema import BranchPrediction, DecisionMode


def valid_mask_from_predictions(pairs: Iterable[Tuple[BranchPrediction, BranchPrediction]],
                                policy: DecisionPolicy | None = None) -> np.ndarray:
    """True where physiology determined the decision (reliable physiology)."""
    return np.array([decide(p, v, policy=policy).mode == DecisionMode.PHYSIOLOGY_PRIMARY
                     for p, v in pairs], dtype=bool)
