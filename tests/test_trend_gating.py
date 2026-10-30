"""Trends must be withheld unless enough valid (physiology-reliable) windows exist."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.aligned_replay import predictions_from_aligned_row
from src.trend_gate import valid_mask_from_predictions

adapter_mod = pytest.importorskip("demo.adapters.trend_adapter")
CSV = Path(__file__).resolve().parents[1] / "results/fusion_lstm/aligned_oof_predictions.csv"
pytestmark = pytest.mark.skipif(not CSV.exists(), reason="aligned predictions missing")

INSUFF = adapter_mod.INSUFF


@pytest.fixture(scope="module")
def subject_rows():
    df = pd.read_csv(CSV)
    return df[df["subject_id"] == "s3"].reset_index(drop=True)


def pairs_for(rows, dropout_idx=()):
    return [predictions_from_aligned_row(r, simulate_physiology_dropout=(i in dropout_idx))
            for i, (_, r) in enumerate(rows.iterrows())]


def test_mask_marks_dropout_windows_invalid(subject_rows):
    mask = valid_mask_from_predictions(pairs_for(subject_rows, dropout_idx={3, 4}))
    assert not mask[3] and not mask[4] and mask[0] and mask[5]


def test_clean_series_reaches_sufficient_data(subject_rows):
    trend = adapter_mod.TemporalTrendAdapter()
    probs = subject_rows["p_physio"].to_numpy()
    mask = valid_mask_from_predictions(pairs_for(subject_rows))
    out = trend.evaluate_series(probs, valid_mask=mask)
    assert not out["is_sufficient"][trend.cfg.K - 2]       # cold start
    assert out["is_sufficient"][trend.cfg.K + 2]


def test_physiology_dropout_makes_trend_insufficient_not_stale(subject_rows):
    trend = adapter_mod.TemporalTrendAdapter()
    probs = subject_rows["p_physio"].to_numpy()
    n = len(probs)
    # Drop physiology for the 8 most recent windows of the first 40 -> too few valid
    drop = set(range(20, 30))
    mask = valid_mask_from_predictions(pairs_for(subject_rows, dropout_idx=drop))
    out = trend.evaluate_series(probs, valid_mask=mask)
    assert out["displayed"][29] == INSUFF
    assert not out["is_sufficient"][29]
    assert n > 40 and out["is_sufficient"][12]
