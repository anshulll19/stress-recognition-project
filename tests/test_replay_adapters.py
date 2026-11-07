from pathlib import Path

import pandas as pd
import pytest

from src.adapters import (CLASS_NAMES, load_oof_predictions, physiology_from_oof,
                          physiology_from_probability, signal_quality_flatline,
                          video_from_window_row, video_quality_from_window)
from src.decision_engine import decide
from src.prediction_schema import Branch, DataSource, DecisionMode

ROOT = Path(__file__).resolve().parents[1]
OOF = ROOT / "results/temporal_models/tcn_lstm_oof_predictions_full.csv"
VIDEO = ROOT / "results/video/video_windows.csv"


class FakeModel:
    def __init__(self, p): self.p = p
    def predict_proba(self, X): return [[1 - self.p, self.p]] * len(X)


class BrokenModel:
    def predict_proba(self, X): raise RuntimeError("boom")


def video_row(**over):
    base = dict(subject_id="s3", session_id="T1", window_start_sec=0.0, window_end_sec=10.0,
                detection_fraction=1.0, window_complete=True, window_reliable=True, f_a=1.0, f_b=2.0)
    base.update(over)
    return pd.Series(base)


FEATS = ["f_a", "f_b"]


def test_flatline_and_nonfinite():
    assert signal_quality_flatline([1.0] * 100)[0] == 0.0
    assert signal_quality_flatline([1.0, float("nan")])[0] == 0.0
    assert signal_quality_flatline([1.0, 2.0, 3.0])[0] == 1.0


def test_physiology_probability_is_valid_and_replay():
    p = physiology_from_probability(0.7, quality=1.0)
    assert p.validate() == [] and p.source == DataSource.REPLAY
    assert p.label == CLASS_NAMES[1]


def test_video_row_to_prediction_and_decision():
    v = video_from_window_row(video_row(), FakeModel(0.9), FEATS)
    assert v.validate() == [] and v.window_ref == "s3/T1/0-10s"
    d = decide(physiology_from_probability(0.8, 1.0), v)
    assert d.mode == DecisionMode.PHYSIOLOGY_PRIMARY and d.source == DataSource.REPLAY


def test_broken_video_model_is_unavailable_not_a_crash():
    v = video_from_window_row(video_row(), BrokenModel(), FEATS)
    assert not v.available
    assert decide(physiology_from_probability(0.8, 1.0), v).supporting_video is None


@pytest.mark.parametrize("over,expect_flag", [
    (dict(detection_fraction=0.0, window_reliable=False), "face_not_detected"),
    (dict(detection_fraction=0.5), "low_face_detection"),
    (dict(window_complete=False), "window_incomplete"),
])
def test_video_quality_flags(over, expect_flag):
    q, flags = video_quality_from_window(video_row(**over))
    assert expect_flag in flags


def test_unreliable_video_window_cannot_support():
    v = video_from_window_row(video_row(window_reliable=False), FakeModel(0.9), FEATS)
    d = decide(physiology_from_probability(0.8, 1.0), v)
    assert d.supporting_video is None


@pytest.mark.skipif(not OOF.exists(), reason="OOF predictions not in checkout")
def test_real_oof_rows_convert():
    df = load_oof_predictions(OOF, "LSTM")
    r = df.iloc[0]
    p = physiology_from_oof(df, int(r["participant"]), int(r["window_index"]), quality=1.0)
    assert p.validate() == []
    missing = physiology_from_oof(df, 999, 0, quality=1.0)
    assert not missing.available


@pytest.mark.skipif(not VIDEO.exists(), reason="video windows not in checkout")
def test_real_video_windows_quality_columns():
    df = pd.read_csv(VIDEO)
    q, flags = video_quality_from_window(df.iloc[0])
    assert 0.0 <= q <= 1.0
