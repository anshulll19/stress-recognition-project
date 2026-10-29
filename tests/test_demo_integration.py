"""Lightweight smoke tests. They need no models, webcam or Streamlit server."""
import json
import py_compile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.decision_engine import decide
from src.prediction_schema import Branch, BranchPrediction, DataSource, DecisionMode

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)


def test_mock_pipeline_end_to_end():
    phys = BranchPrediction.from_dict({"branch": "physiology", "available": True, "source": "replay",
        "probabilities": {"rest": 0.3, "stress": 0.7}, "quality": 0.8, "timestamp": NOW.isoformat()})
    video = BranchPrediction.from_dict({"branch": "video", "available": True, "source": "replay",
        "probabilities": {"rest": 0.4, "stress": 0.6}, "quality": 0.7, "timestamp": NOW.isoformat()})
    out = decide(phys, video, now=NOW).to_dict()
    json.loads(json.dumps(out))
    assert out["mode"] == DecisionMode.PHYSIOLOGY_PRIMARY.value
    assert out["source"] == "replay"


@pytest.mark.parametrize("rel", ["app.py", "ubfc_pipeline/s8_late_fusion.py"])
def test_existing_files_still_compile(rel):
    path = ROOT / rel
    if not path.exists():
        pytest.skip(f"{rel} not present in this checkout")
    py_compile.compile(str(path), doraise=True)


@pytest.mark.parametrize("rel", ["models/deployment_candidate/manifest.json",
                                 "models/video_model_metadata.json"])
def test_model_metadata_is_valid_json(rel):
    path = ROOT / rel
    if not path.exists():
        pytest.skip(f"{rel} not present in this checkout")
    json.loads(path.read_text(encoding="utf-8"))
