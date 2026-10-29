import json
from datetime import datetime, timedelta, timezone

import pytest

from src.decision_engine import DecisionPolicy, decide
from src.prediction_schema import Branch, BranchPrediction, DataSource, DecisionMode

NOW = datetime(2026, 10, 30, 12, 0, 0, tzinfo=timezone.utc)


def mk(branch, label_p=0.8, quality=0.9, available=True, source=DataSource.REPLAY,
       ts=None, flags=None, classes=("rest", "stress")):
    probs = {classes[0]: round(1 - label_p, 6), classes[1]: label_p}
    return BranchPrediction(branch=branch, available=available, source=source,
                            probabilities=probs if available else {}, quality=quality,
                            quality_flags=flags or [], timestamp=ts or NOW.isoformat())


def phys(**kw): return mk(Branch.PHYSIOLOGY, **kw)
def vid(**kw): return mk(Branch.VIDEO, **kw)


def test_reliable_physiology_is_primary():
    d = decide(phys(), vid(), now=NOW)
    assert d.mode == DecisionMode.PHYSIOLOGY_PRIMARY
    assert d.determining_branch == Branch.PHYSIOLOGY
    assert d.primary_label == "stress"
    assert d.supporting_video is not None


def test_video_never_overrides_physiology_on_disagreement():
    d = decide(phys(label_p=0.9), vid(label_p=0.1), now=NOW)
    assert d.primary_label == "stress"
    assert d.supporting_video.label == "rest"
    assert any("disagree" in w for w in d.warnings)


def test_unreliable_video_not_shown_as_support():
    d = decide(phys(), vid(quality=0.1), now=NOW)
    assert d.mode == DecisionMode.PHYSIOLOGY_PRIMARY
    assert d.supporting_video is None
    assert "Video was not used" in d.explanation


@pytest.mark.parametrize("p", [None, phys(available=False), phys(quality=0.2),
                               phys(flags=["signal_lost"])])
def test_physiology_missing_or_bad_falls_back_to_labelled_video(p):
    d = decide(p, vid(), now=NOW)
    assert d.mode == DecisionMode.VIDEO_FALLBACK
    assert d.is_fallback and d.determining_branch == Branch.VIDEO
    assert d.banner and "VIDEO-ONLY" in d.banner


def test_neither_reliable_is_insufficient():
    d = decide(phys(quality=0.1), vid(available=False), now=NOW)
    assert d.mode == DecisionMode.INSUFFICIENT_DATA
    assert d.primary_label is None and d.primary_probabilities == {}
    assert d.determining_branch is None


def test_both_missing():
    assert decide(None, None, now=NOW).mode == DecisionMode.INSUFFICIENT_DATA


def test_invalid_probabilities_are_unreliable():
    bad = phys()
    bad.probabilities = {"rest": 0.9, "stress": 0.9}
    assert decide(bad, vid(), now=NOW).mode == DecisionMode.VIDEO_FALLBACK


def test_wrong_branch_object_is_rejected():
    d = decide(vid(), None, now=NOW)  # video object passed as physiology
    assert d.mode == DecisionMode.INSUFFICIENT_DATA


def test_stale_live_prediction_rejected_but_stale_replay_ok():
    old = (NOW - timedelta(seconds=60)).isoformat()
    assert decide(phys(source=DataSource.LIVE, ts=old), None, now=NOW).mode == DecisionMode.INSUFFICIENT_DATA
    assert decide(phys(source=DataSource.REPLAY, ts=old), None, now=NOW).mode == DecisionMode.PHYSIOLOGY_PRIMARY


def test_live_requires_timestamp():
    p = phys(source=DataSource.LIVE)
    p.timestamp = None
    assert decide(p, None, now=NOW).mode == DecisionMode.INSUFFICIENT_DATA


def test_replay_never_labelled_live():
    d = decide(phys(source=DataSource.LIVE), vid(source=DataSource.REPLAY), now=NOW)
    assert d.source == DataSource.REPLAY
    assert any("mix" in w.lower() for w in d.warnings)
    assert decide(phys(source=DataSource.LIVE), vid(source=DataSource.LIVE), now=NOW).source == DataSource.LIVE


def test_custom_policy_threshold():
    strict = DecisionPolicy(min_physiology_quality=0.95)
    assert decide(phys(quality=0.9), vid(), policy=strict, now=NOW).mode == DecisionMode.VIDEO_FALLBACK


def test_decision_is_json_serialisable_and_roundtrips_prediction():
    d = decide(phys(), vid(), now=NOW)
    json.dumps(d.to_dict())
    p = phys()
    assert BranchPrediction.from_dict(p.to_dict()).to_dict() == p.to_dict()
