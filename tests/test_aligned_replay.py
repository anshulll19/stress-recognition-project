from pathlib import Path

import pandas as pd
import pytest

from src.aligned_replay import predictions_from_aligned_row, window_ref
from src.decision_engine import decide
from src.prediction_schema import DataSource, DecisionMode

CSV = Path(__file__).resolve().parents[1] / "results/fusion_lstm/aligned_oof_predictions.csv"
pytestmark = pytest.mark.skipif(not CSV.exists(), reason="aligned predictions missing")


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(CSV)


def test_every_row_converts_to_valid_replay_predictions(df):
    for _, row in df.iterrows():
        phys, vid = predictions_from_aligned_row(row)
        assert phys.validate() == [] and vid.validate() == []
        assert phys.source == vid.source == DataSource.REPLAY


def test_all_windows_decide_physiology_first_unless_face_lost(df):
    for _, row in df.iterrows():
        d = decide(*predictions_from_aligned_row(row))
        assert d.source == DataSource.REPLAY
        assert d.mode == DecisionMode.PHYSIOLOGY_PRIMARY
        assert d.primary_probabilities["stress_task_proxy"] == pytest.approx(row["p_physio"], abs=1e-9)


def test_simulated_failures(df):
    row = df.iloc[0]
    assert decide(*predictions_from_aligned_row(row, simulate_physiology_dropout=True)
                  ).mode == DecisionMode.VIDEO_FALLBACK
    assert decide(*predictions_from_aligned_row(row, simulate_physiology_dropout=True,
                                                simulate_video_loss=True)
                  ).mode == DecisionMode.INSUFFICIENT_DATA
    d = decide(*predictions_from_aligned_row(row, simulate_video_loss=True))
    assert d.mode == DecisionMode.PHYSIOLOGY_PRIMARY and d.supporting_video is None


def test_low_detection_window_video_not_used_as_support(df):
    row = df.iloc[0].copy()
    row["detection_fraction"] = 0.3
    d = decide(*predictions_from_aligned_row(row))
    assert d.supporting_video is None


def test_window_ref_format(df):
    assert window_ref(df.iloc[0]) == "s3/T1/0-10s"
